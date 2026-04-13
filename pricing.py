"""
Model Pricing Data for Cost Estimation.

Pricing data sourced from LiteLLM model_prices_and_context_window.json.
All prices should be configured in CNY per token.

Note: For Chinese models like Qwen via DashScope, prices may vary.
      These are estimated values based on publicly available information.
"""

MODEL_ALIASES = {
    # DeepSeek public model aliases
    "deepseek-v3": "deepseek-chat",
    "deepseek-v3.1": "deepseek-chat",
    "deepseek-v3.2": "deepseek-chat",
    "deepseek-reasoner": "deepseek-chat",
    # Common provider-prefixed names
    "deepseek/deepseek-chat": "deepseek-chat",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
}

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
        "input_cost_per_token": 0.002 / 1000,
        "output_cost_per_token": 0.003 / 1000,
        "cache_read_input_token_cost": 0.002 / 5000,
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


# Source: Aliyun IQS billing docs (lowest tier / default tier assumptions). 
TOOL_PRICING = {
    "aliyun_iqs_search": {
        "unit": "calls",
        "price_per_1000_calls": 42.0,
        "billing_calls_key": "effective_calls",
        "note": "Default to lowest ladder tier (tier-1) for standard search.",
    },
    # Local/sandbox tools or third-party tools without this doc's billing scope.
    "python_interpreter": {
        "unit": "calls",
        "price_per_1000_calls": 0.0,
        "billing_calls_key": "calls",
    },
    "PythonInterpreter": {
        "unit": "calls",
        "price_per_1000_calls": 0.0,
        "billing_calls_key": "calls",
    },
    "google_search": {
        "unit": "calls",
        "price_per_1000_calls": 198.0,
        "billing_calls_key": "effective_calls",
    },
    "google_scholar": {
        "unit": "calls",
        "price_per_1000_calls": 198.0,
        "billing_calls_key": "effective_calls",
    },
    "baidu_search": {
        "unit": "calls",
        "price_per_1000_calls": 36.0,
        "billing_calls_key": "effective_calls",
    },
    "Visit": {
        "unit": "calls",
        "price_per_1000_calls": 0.0,
        "billing_calls_key": "calls",
    },
}


def get_model_pricing(model_name: str) -> dict:
    """Get pricing for a model. Returns None if not found."""
    model_lower = (model_name or "").strip().lower()

    # Normalize provider prefixes, e.g. openai/gpt-4o -> gpt-4o
    if "/" in model_lower:
        model_lower = model_lower.split("/")[-1]

    # Normalize endpoint-like names, e.g. xxx:model
    if ":" in model_lower:
        model_lower = model_lower.split(":")[-1]

    # Alias remapping first
    model_lower = MODEL_ALIASES.get(model_lower, model_lower)
    
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


def get_tool_pricing(tool_name: str) -> dict:
    """Get pricing configuration for a tool. Returns None if not found."""
    if not tool_name:
        return None
    return TOOL_PRICING.get(tool_name)


def calculate_tool_cost(
    tool_name: str,
    calls: int = 0,
    effective_calls: int = 0,
) -> dict:
    """Calculate tool cost using configured CNY pricing."""
    pricing = get_tool_pricing(tool_name)
    if pricing is None:
        return {
            "tool_name": tool_name,
            "billed_calls": 0,
            "unit_price_per_1000_calls": 0.0,
            "total_cost": 0.0,
            "estimated": True,
            "warning": f"Tool pricing not found for {tool_name}",
        }

    billing_key = pricing.get("billing_calls_key", "calls")
    billed_calls = max(0, int(effective_calls if billing_key == "effective_calls" else calls))
    price_per_1000 = float(pricing.get("price_per_1000_calls", 0.0))
    total_cost = billed_calls * (price_per_1000 / 1000.0)

    return {
        "tool_name": tool_name,
        "billed_calls": billed_calls,
        "unit_price_per_1000_calls": price_per_1000,
        "total_cost": total_cost,
        "estimated": False,
        "note": pricing.get("note", ""),
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
    return f"¥{cost:.4f}"
