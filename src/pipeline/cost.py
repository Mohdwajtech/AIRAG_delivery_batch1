"""src/pipeline/cost.py — real cost from token usage.

RATES are $ per 1,000,000 tokens as (input_rate, output_rate).
VERIFY AGAINST TODAY'S PRICING before trusting these numbers.
"""
RATES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o":      (2.50, 10.00),
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """cost = (input*in_rate + output*out_rate) / 1e6. Unknown model -> 0.0."""
    if model not in RATES:
        return 0.0
    in_rate, out_rate = RATES[model]
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000
