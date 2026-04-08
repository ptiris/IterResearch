"""
Model Pricing Data for Cost Estimation.

Pricing data sourced from LiteLLM model_prices_and_context_window.json.
All prices are in USD per 1 million tokens.

Note: For Chinese models like Qwen via DashScope, prices may vary.
      These are estimated values based on publicly available information.
"""

USD_TO_CNY = 7.0

MODEL_PRICING = {
    # Qwen Series (via DashScope/Aliyun)
    "qwen-flash": {
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.000002,
        "cache_read_input_token_cost": 0.0000001,
        "cache_creation_input_token_cost": 0.000003,
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-plus": {
        "input_cost_per_token": 0.000004,
        "output_cost_per_token": 0.000012,
        "cache_read_input_token_cost": 0.000001,
        "cache_creation_input_token_cost": 0.000015,
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-long": {
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "cache_read_input_token_cost": 0.0000002,
        "cache_creation_input_token_cost": 0.00001,
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-coder-plus": {
        "input_cost_per_token": 0.000007,
        "output_cost_per_token": 0.00002,
        "cache_read_input_token_cost": 0.0000015,
        "cache_creation_input_token_cost": 0.00002,
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    "qwen-max": {
        "input_cost_per_token": 0.00002,
        "output_cost_per_token": 0.00006,
        "cache_read_input_token_cost": 0.000004,
        "cache_creation_input_token_cost": 0.00006,
        "litellm_provider": "dashscope",
        "mode": "chat",
    },
    
    # Claude Series
    "claude-opus-4": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "cache_read_input_token_cost": 0.0000015,
        "cache_creation_input_token_cost": 0.00001875,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-sonnet-4": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_read_input_token_cost": 0.0000003,
        "cache_creation_input_token_cost": 0.00000375,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-5-sonnet": {
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_read_input_token_cost": 0.0000003,
        "cache_creation_input_token_cost": 0.00000375,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "claude-3-5-haiku": {
        "input_cost_per_token": 0.0000008,
        "output_cost_per_token": 0.000004,
        "cache_read_input_token_cost": 0.00000008,
        "cache_creation_input_token_cost": 0.000001,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    
    # GPT-4 Series
    "gpt-4o": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.00001,
        "cache_read_input_token_cost": 0.00000125,
        "cache_creation_input_token_cost": 0.000015,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4o-mini": {
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.0000006,
        "cache_read_input_token_cost": 0.000000075,
        "cache_creation_input_token_cost": 0.00000035,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-4-turbo": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "cache_read_input_token_cost": 0.00001,
        "cache_creation_input_token_cost": 0.00003,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    
    # DeepSeek Series
    "deepseek-chat": {
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.00000028,
        "cache_read_input_token_cost": 0.00000001,
        "cache_creation_input_token_cost": 0.00000014,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    "deepseek-coder": {
        "input_cost_per_token": 0.00000014,
        "output_cost_per_token": 0.00000049,
        "cache_read_input_token_cost": 0.000000014,
        "cache_creation_input_token_cost": 0.00000049,
        "litellm_provider": "deepseek",
        "mode": "chat",
    },
    
    # Gemini Series
    "gemini-1.5-pro": {
        "input_cost_per_token": 0.00000125,
        "output_cost_per_token": 0.000005,
        "cache_read_input_token_cost": 0.000000125,
        "cache_creation_input_token_cost": 0.0000035,
        "litellm_provider": "gemini",
        "mode": "chat",
    },
    "gemini-1.5-flash": {
        "input_cost_per_token": 0.000000035,
        "output_cost_per_token": 0.00000014,
        "cache_read_input_token_cost": 0.0000000175,
        "cache_creation_input_token_cost": 0.00000007,
        "litellm_provider": "gemini",
        "mode": "chat",
    },
}


def get_model_pricing(model_name: str) -> dict:
    """Get pricing for a model. Returns None if not found."""
    model_lower = model_name.lower()
    
    # Direct match
    if model_lower in MODEL_PRICING:
        return MODEL_PRICING[model_lower]
    
    # Partial match (e.g., "qwen-flash" matches "qwen-flash-8k")
    for key in MODEL_PRICING:
        if key in model_lower or model_lower in key:
            return MODEL_PRICING[key]
    
    # Try without version suffixes
    base_name = model_lower.split("-")[0]
    if base_name in MODEL_PRICING:
        return MODEL_PRICING[base_name]
    
    return None


def calculate_cost(
    model_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0
) -> dict:
    """
    Calculate cost for an API call.
    
    Returns dict with:
    - input_cost: cost for fresh input tokens
    - cache_read_cost: cost for cache read tokens  
    - cache_write_cost: cost for cache write tokens
    - output_cost: cost for completion tokens
    - total_cost: total cost
    """
    pricing = get_model_pricing(model_name)
    
    if pricing is None:
        return {
            "input_cost": 0.0,
            "cache_read_cost": 0.0,
            "cache_write_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "estimated": True,
            "warning": f"Pricing not found for {model_name}"
        }
    
    fresh_tokens = max(0, prompt_tokens - cache_read_tokens)
    
    input_cost = fresh_tokens * pricing["input_cost_per_token"]
    cache_read_cost = cache_read_tokens * pricing.get("cache_read_input_token_cost", pricing["input_cost_per_token"])
    cache_write_cost = cache_write_tokens * pricing.get("cache_creation_input_token_cost", pricing["input_cost_per_token"])
    output_cost = completion_tokens * pricing["output_cost_per_token"]
    
    total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "cache_read_cost": cache_read_cost,
        "cache_write_cost": cache_write_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "estimated": False,
        "pricing_model": model_name
    }


def format_cost_usd(cost: float) -> str:
    """Format cost in USD."""
    if cost < 0.0001:
        return f"${cost * 1000000:.2f}/M"
    elif cost < 0.01:
        return f"${cost * 1000:.4f}/K"
    else:
        return f"${cost:.4f}"


def format_cost_cny(cost: float) -> str:
    """Format cost in CNY."""
    return f"¥{cost * USD_TO_CNY:.4f}"
