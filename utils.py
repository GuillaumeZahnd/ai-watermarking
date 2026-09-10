import scipy.stats as stats
import math


def print_results(
    nb_green_tokens: int,
    nb_tokens: int,
    z_score: float,
    nb_tokens_min: int
) -> None:

    if nb_tokens < nb_tokens_min:
        print(f"Not enough tokens: expected at least {nb_tokens_min}, got {nb_tokens}")
        return

    print(f"Tokens: {nb_tokens}, Green tokens: {nb_green_tokens}, Ratio: {(nb_green_tokens/nb_tokens):.2f}")
    print(f"z-score: {z_score:.2f}")

    p_value, confidence = z_score_to_p_value(z_score=z_score)
    print(f"p-value: {p_value:.6f} (confidence: {(confidence*100):.2f})")


def z_score_to_p_value(z_score: float) -> tuple[float, float]:
    """
    Frequentist approach.
    Right-tailed p-value (probability of observing a score greater or equal to this z-score by pure chance)
    and corresponding confidence.
    """

    p_value = stats.norm.sf(z_score)
    confidence = 1.0 - p_value
    return p_value, confidence
