"""
Reward functions for j1-nano-causal training.

This extends the original j1-micro rewards with a CAUSAL SENSITIVITY reward
that encourages criteria to actually affect the scores.

R_total = R_format + R_argmax + lambda * R_causal

Where R_causal measures: || d(score) / d(criteria) ||
"""

import re
import torch
import torch.nn.functional as F
from config import Config
from typing import List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Global counters for logging
_reward_stats = {
    'format_rewards': [],
    'argmax_rewards': [],
    'causal_rewards': [],
    'causal_coherence': [],
    'causal_score_diff': [],
    'causal_specificity': [],
    'step': 0
}

def log_reward_stats():
    """Log aggregated reward stats to wandb."""
    global _reward_stats
    if not WANDB_AVAILABLE or not wandb.run:
        return

    if len(_reward_stats['format_rewards']) > 0:
        wandb.log({
            'rewards/R_format_mean': sum(_reward_stats['format_rewards']) / len(_reward_stats['format_rewards']),
            'rewards/R_argmax_mean': sum(_reward_stats['argmax_rewards']) / len(_reward_stats['argmax_rewards']) if _reward_stats['argmax_rewards'] else 0,
            'rewards/R_causal_mean': sum(_reward_stats['causal_rewards']) / len(_reward_stats['causal_rewards']) if _reward_stats['causal_rewards'] else 0,
            'causal/coherence_mean': sum(_reward_stats['causal_coherence']) / len(_reward_stats['causal_coherence']) if _reward_stats['causal_coherence'] else 0,
            'causal/score_diff_mean': sum(_reward_stats['causal_score_diff']) / len(_reward_stats['causal_score_diff']) if _reward_stats['causal_score_diff'] else 0,
            'causal/specificity_mean': sum(_reward_stats['causal_specificity']) / len(_reward_stats['causal_specificity']) if _reward_stats['causal_specificity'] else 0,
            'step': _reward_stats['step']
        })

        # Reset for next batch
        _reward_stats = {
            'format_rewards': [],
            'argmax_rewards': [],
            'causal_rewards': [],
            'causal_coherence': [],
            'causal_score_diff': [],
            'causal_specificity': [],
            'step': _reward_stats['step'] + 1
        }

# Global encoder for computing criteria embeddings
_encoder = None
_tokenizer = None


class FormatError(Exception):
    pass


def get_encoder():
    """Lazy load the encoder model for criteria embedding."""
    global _encoder, _tokenizer
    if _encoder is None:
        # Use a small encoder model for efficiency
        # This runs on CPU to not interfere with training
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _encoder.eval()
        # Keep on CPU to avoid GPU memory issues during training
    return _encoder, _tokenizer


def mean_pooling(model_output, attention_mask):
    """Mean pooling for sentence embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_text(text: str) -> torch.Tensor:
    """Encode text to embedding vector."""
    encoder, tokenizer = get_encoder()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = encoder(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)


def extract_scores(raw_response: str) -> List[float]:
    """
    Extract Judge scores from the raw response.
    Expects the following format:
        ---
        <specific_criteria>...</specific_criteria>
        <analysis>...</analysis>
        <scores>\boxed{x, y}</scores>
        ---
    """
    match = re.search(r"<scores>(.*?)</scores>", raw_response, re.DOTALL)
    if not match:
        raise FormatError("No Judge scores found in response")

    boxed_match = re.search(r"\\boxed{([\d.]+),\s*([\d.]+)}", match.group(1))
    if not boxed_match:
        raise FormatError("No boxed scores found in scores tag")

    try:
        return [float(boxed_match.group(1)), float(boxed_match.group(2))]
    except ValueError:
        raise FormatError("Invalid score format in boxed response")


def extract_criteria(raw_response: str) -> Optional[str]:
    """Extract criteria text from the response."""
    match = re.search(r"<specific_criteria>(.*?)</specific_criteria>", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_analysis(raw_response: str) -> Optional[str]:
    """Extract analysis text from the response."""
    match = re.search(r"<analysis>(.*?)</analysis>", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_causal_sensitivity(
    criteria_text: str,
    analysis_text: str,
    scores: List[float],
    num_perturbations: int = 5,
    perturbation_std: float = 0.1
) -> Tuple[float, float, float, float]:
    """
    Compute causal sensitivity of criteria.

    Method: Measure how much the criteria embedding influences the analysis.

    High sensitivity = criteria strongly correlate with analysis
    Low sensitivity = criteria are ignored (decorative)

    We use three metrics:
    1. Semantic similarity between criteria and analysis
    2. Score spread (how differentiated are the scores)
    3. Criteria specificity

    Returns: (sensitivity, coherence, score_diff, specificity)
    """
    if not criteria_text or not analysis_text:
        return 0.0, 0.0, 0.0, 0.0

    try:
        # Encode criteria and analysis
        criteria_emb = encode_text(criteria_text)
        analysis_emb = encode_text(analysis_text)

        # Metric 1: Semantic coherence between criteria and analysis
        # If analysis actually uses criteria, they should be semantically related
        coherence = F.cosine_similarity(criteria_emb, analysis_emb).item()
        coherence = max(0.0, coherence)  # Ensure non-negative

        # Metric 2: Score differentiation
        # If criteria are useful, scores should be different (not both 5.0)
        score_diff = abs(scores[0] - scores[1]) / 10.0  # Normalize to [0, 1]

        # Metric 3: Criteria specificity (longer, more detailed = better)
        # Count distinct criteria points
        criteria_lines = [l.strip() for l in criteria_text.split('\n') if l.strip()]
        specificity = min(len(criteria_lines) / 5.0, 1.0)  # Cap at 5 criteria

        # Combined sensitivity score
        # High coherence + high score diff + specific criteria = high sensitivity
        sensitivity = (coherence + score_diff + specificity) / 3.0

        return max(0.0, sensitivity), coherence, score_diff, specificity

    except Exception as e:
        return 0.0, 0.0, 0.0, 0.0


def format_reward_func(completions: List[dict[str, str]], **kwargs) -> List[float]:
    """
    Judge format reward for the n=2 (pairwise preference) case

    > Range: [0, 0.2]
    > Boxed contributes 1/4 of the total score
    > Each of the 6 tags contributes 1/8 of the total score
    """
    global _reward_stats
    scores = []
    for completion in completions:
        boxed_fmted = True
        raw_response = completion[0]["content"]
        try:
            extract_scores(raw_response)
        except FormatError:
            boxed_fmted = False
        finally:
            required_tags = [
                "<specific_criteria>",
                "</specific_criteria>",
                "<analysis>",
                "</analysis>",
                "<scores>",
                "</scores>",
            ]
            denominator = len(required_tags) + 2
            total_score = 0.0
            for tag in required_tags:
                if tag in raw_response:
                    total_score += 1.0 / denominator

            if boxed_fmted:
                total_score += 2.0 / denominator

            reward = total_score / 5
            scores.append(reward)
            _reward_stats['format_rewards'].append(reward)

    return scores


def argmax_reward_func(
    completions: List[dict[str, str]], chosen_positions: List[str], **kwargs
) -> List[float]:
    """
    For the n=2 (pairwise preference) case, the `chosen` response should score higher than the `rejected` response.

    > Range: [0, 1]
    > completions: list of rollouts
    > chosen_positions: list of positions (A or B) of the `chosen` response
    """
    global _reward_stats
    scores = []
    for completion, chosen_position in zip(completions, chosen_positions):
        raw_response = completion[0]["content"]
        try:
            extracted_score_box = extract_scores(raw_response)
        except FormatError:
            scores.append(0.0)
            _reward_stats['argmax_rewards'].append(0.0)
            continue
        else:
            position_to_score = {
                Config.CHOSEN_POSITION_A: lambda scores: scores[0] > scores[1],
                Config.CHOSEN_POSITION_B: lambda scores: scores[1] > scores[0],
            }
            if chosen_position not in position_to_score:
                raise ValueError(f"Invalid chosen position: {chosen_position}")
            reward = 1.0 if position_to_score[chosen_position](extracted_score_box) else 0.0
            scores.append(reward)
            _reward_stats['argmax_rewards'].append(reward)

    return scores


def causal_reward_func(
    completions: List[dict[str, str]], **kwargs
) -> List[float]:
    """
    NEW: Causal sensitivity reward.

    Measures whether the generated criteria actually influence the scores.

    > Range: [0, lambda] where lambda = Config.CAUSAL_REWARD_WEIGHT

    High reward = criteria are causally relevant (used in analysis, affect scores)
    Low reward = criteria are decorative (ignored)

    This encourages:
      d(score) / d(criteria) != 0
    """
    global _reward_stats
    rewards = []

    for completion in completions:
        raw_response = completion[0]["content"]

        try:
            # Extract all components
            scores = extract_scores(raw_response)
            criteria_text = extract_criteria(raw_response)
            analysis_text = extract_analysis(raw_response)

            if criteria_text and analysis_text:
                # Compute causal sensitivity (returns all components)
                sensitivity, coherence, score_diff, specificity = compute_causal_sensitivity(
                    criteria_text,
                    analysis_text,
                    scores,
                    num_perturbations=Config.NUM_PERTURBATIONS,
                    perturbation_std=Config.PERTURBATION_STD
                )

                # Scale by weight and convert to reward
                reward = Config.CAUSAL_REWARD_WEIGHT * sensitivity

                # Track all components for wandb
                _reward_stats['causal_rewards'].append(reward)
                _reward_stats['causal_coherence'].append(coherence)
                _reward_stats['causal_score_diff'].append(score_diff)
                _reward_stats['causal_specificity'].append(specificity)
            else:
                reward = 0.0
                _reward_stats['causal_rewards'].append(0.0)

        except (FormatError, Exception):
            reward = 0.0
            _reward_stats['causal_rewards'].append(0.0)

        rewards.append(reward)

    # Log stats to wandb after processing batch
    log_reward_stats()

    return rewards


# =============================================================================
# ALTERNATIVE: Gradient-based causal reward (more principled but slower)
# =============================================================================

def causal_reward_func_gradient(
    completions: List[dict[str, str]],
    model=None,
    tokenizer=None,
    **kwargs
) -> List[float]:
    """
    More principled causal reward using actual gradients.

    This computes || d(score) / d(criteria_embedding) || directly.

    Requires access to the model during reward computation.
    Use this if you want true gradient-based sensitivity.
    """
    if model is None:
        # Fall back to heuristic version
        return causal_reward_func(completions, **kwargs)

    rewards = []

    for completion in completions:
        raw_response = completion[0]["content"]

        try:
            criteria_text = extract_criteria(raw_response)

            if criteria_text and model is not None:
                # Tokenize criteria
                inputs = tokenizer(criteria_text, return_tensors="pt", truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Get embeddings with gradient tracking
                embeddings = model.get_input_embeddings()(inputs['input_ids'])
                embeddings.requires_grad_(True)

                # Forward pass through model (simplified)
                # In practice, you'd need to run the full scoring computation
                outputs = model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
                logits = outputs.logits

                # Use mean logit as proxy for "score influence"
                score_proxy = logits.mean()

                # Compute gradient
                grad = torch.autograd.grad(score_proxy, embeddings, retain_graph=False)[0]
                sensitivity = grad.norm().item()

                reward = Config.CAUSAL_REWARD_WEIGHT * min(sensitivity, 1.0)
            else:
                reward = 0.0

        except Exception:
            reward = 0.0

        rewards.append(reward)

    return rewards
