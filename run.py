"""
IterResearch: Iterative Research Agent

Main entry point for running the iterative research agent.
The agent performs deep research tasks through multi-turn tool interactions.
"""
import os
import sys
import json
import copy
import time
import re
import random
import argparse
import traceback
import requests
import threading
from urllib.parse import urlparse
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from easydict import EasyDict

# Local imports
from config import (
    LLM_URL,
    MAX_TURN,
    MAX_FORMAT_RETRIES,
    MAX_WORKERS,
    MAX_OBSERVATION_TOKENS,
    MAX_WEBPAGE_TOKENS,
    TOKENIZER_PATH,
    SUMMARY_LLM_URL,
    SUMMARY_LLM_AUTH,
    ALIYUN_LLM_URL,
    ALIYUN_API_KEY,
    DEEPSEEK_LLM_URL,
    DEEPSEEK_API_KEY
)
from prompts import (
    initial_instruction_prompt,
    instruction_prompt,
    observation_prompt,
    last_instruction_prompt
)
from tools import Search, BaiduSearch, AliyunIQSSearch, Scholar, PythonInterpreter, Visit
from config import SUMMARY_LLM_AUTH, OPENAI_API_KEY
from metrics import get_metrics_collector, StepRecord


# =============================================================================
# Tool Definitions
# =============================================================================
GOOGLE_SEARCH_TOOL = {
    "name": "google_search",
    "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries."
            }
        },
        "required": ["query"]
    }
}

BAIDU_SEARCH_TOOL = {
    "name": "baidu_search",
    "description": "Perform Baidu web searches via Qianfan API then returns a string of the top search results. Accepts multiple queries.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries."
            }
        },
        "required": ["query"]
    }
}

ALIYUN_IQS_TOOL = {
    "name": "aliyun_iqs_search",
    "description": "Aliyun Information Query Service search, providing real-time open-domain search capabilities. Accepts multiple queries.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries."
            }
        },
        "required": ["query"]
    }
}

GOOGLE_SCHOLAR_TOOL = {
    "name": "google_scholar",
    "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            }
        },
        "required": ["query"]
    }
}

VISIT_TOOL = {
    "name": "Visit",
    "description": "Visit webpage(s) or paper(s) and return the summary of the content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) or paper(s) to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s) or paper(s)."
            },
            "parse_type": {
                "type": "string",
                "enum": ["html", "pdf"],
                "default": "html",
                "description": "Specify whether to visit a HTML webpage or a PDF paper."
            }
        },
        "required": ["url", "goal"]
    }
}

PYTHON_INTERPRETER_TOOL = {
    "name": "PythonInterpreter",
    "description": "Executes arbitrary Python code in a secure, sandboxed environment. This tool is designed for performing complex calculations, data manipulations, string processing, logical operations, and general programming tasks. Use print() for any output you want to see.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. All output should be explicitly printed using print() functions."
            }
        },
        "required": ["code"]
    }
}

TOOLS = [
    GOOGLE_SEARCH_TOOL,
    BAIDU_SEARCH_TOOL,
    GOOGLE_SCHOLAR_TOOL,
    VISIT_TOOL,
    PYTHON_INTERPRETER_TOOL
]

ALL_TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS + [ALIYUN_IQS_TOOL]}
TOOL_NAME_ALIASES = {
    "google": "google_search",
    "google_search": "google_search",
    "baidu": "baidu_search",
    "baidu_search": "baidu_search",
    "aliyun": "aliyun_iqs_search",
    "aliyun_iqs": "aliyun_iqs_search",
    "aliyun_iqs_search": "aliyun_iqs_search",
    "scholar": "google_scholar",
    "google_scholar": "google_scholar",
    "visit": "Visit",
    "python": "PythonInterpreter",
    "python_interpreter": "PythonInterpreter",
    "pythoninterpreter": "PythonInterpreter",
    "PythonInterpreter": "PythonInterpreter"
}


# =============================================================================
# Global Variables
# =============================================================================


def _parse_tools_arg(tools_arg: str) -> list:
    """Parse --tools CSV argument into canonical tool names."""
    if not tools_arg:
        return []

    raw_items = [item.strip() for item in tools_arg.split(',') if item.strip()]
    if not raw_items:
        return []

    resolved = []
    unknown = []
    for item in raw_items:
        key = item.replace('-', '_')
        canonical = TOOL_NAME_ALIASES.get(key) or TOOL_NAME_ALIASES.get(key.lower())
        if not canonical:
            unknown.append(item)
            continue
        if canonical not in resolved:
            resolved.append(canonical)

    if unknown:
        valid_names = sorted(set(TOOL_NAME_ALIASES.keys()))
        raise ValueError(
            f"Unknown tool(s): {', '.join(unknown)}. "
            f"Valid values include: {', '.join(valid_names)}"
        )

    return resolved


def get_tools_for_engine(search_engine: str) -> list:
    """Get filtered TOOLS list based on selected search engine or --tools override."""
    if ACTIVE_TOOL_NAMES:
        return [ALL_TOOLS_BY_NAME[name] for name in ACTIVE_TOOL_NAMES]

    base_tools = [VISIT_TOOL, PYTHON_INTERPRETER_TOOL]
    
    if search_engine == "google":
        return base_tools + [GOOGLE_SEARCH_TOOL, GOOGLE_SCHOLAR_TOOL]
    elif search_engine == "baidu":
        return base_tools + [BAIDU_SEARCH_TOOL]
    elif search_engine == "aliyun":
        return base_tools + [ALIYUN_IQS_TOOL]
    
    return base_tools + [GOOGLE_SEARCH_TOOL, BAIDU_SEARCH_TOOL, GOOGLE_SCHOLAR_TOOL, ALIYUN_IQS_TOOL]


def get_tool_str_for_engine(search_engine: str) -> str:
    """Get filtered TOOLS JSON string based on selected search engine."""
    filtered_tools = get_tools_for_engine(search_engine)
    return json.dumps(filtered_tools, indent=2)


# Initialize tools
python_executor = PythonInterpreter()
google_search_engine = Search()
baidu_search_engine = BaiduSearch()
aliyun_iqs_engine = AliyunIQSSearch()
search_engine = google_search_engine
scholar_engine = Scholar()
visit_tool = Visit()

SEARCH_ENGINE = "google"
ACTIVE_TOOL_NAMES = None
RESEARCH_MODEL = "qwen-flash"
LLM_URL_CONFIG = LLM_URL  # Main LLM URL for all purposes
SUMMARY_LLM_URL_CONFIG = SUMMARY_LLM_URL  # Kept for backward compatibility
SUMMARY_MODEL = "qwen-flash"
TOKENIZER_PATH_CONFIG = TOKENIZER_PATH
MAX_OBSERVATION_TOKENS_CONFIG = MAX_OBSERVATION_TOKENS
MAX_WEBPAGE_TOKENS_CONFIG = MAX_WEBPAGE_TOKENS
DISABLE_GOOGLE_SCHOLAR = False
EVALUATOR_ENABLED = True
EVALUATOR_LLM_URL_CONFIG = LLM_URL
EVALUATOR_MODEL = "qwen-flash"
MAX_COMPLETION_TOKENS_CONFIG = 4096
CURRENT_API_KEY = None  # API key for current provider

# Statistics
failed_call = 0
total_call = 0
CALL_COUNTER_LOCK = threading.Lock()

# Tokenizer for observation length control
_tokenizer = None

def configure_provider(provider: str) -> tuple:
    """
    Load LLM configuration based on provider.
    
    Args:
        provider: Provider name ('aliyun', 'deepseek', or 'default')
        
    Returns:
        Tuple of (llm_url, api_key) for the provider
    """
    global LLM_URL_CONFIG, EVALUATOR_LLM_URL_CONFIG, CURRENT_API_KEY
    
    provider = provider.lower() if provider else "default"
    
    if provider == "aliyun":
        if not ALIYUN_LLM_URL:
            raise ValueError("ALIYUN_LLM_URL not configured in environment")
        llm_url = ALIYUN_LLM_URL
        api_key = ALIYUN_API_KEY
        print(f"[CONFIG] Using Aliyun provider: {llm_url}")
    elif provider == "deepseek":
        if not DEEPSEEK_LLM_URL:
            raise ValueError("DEEPSEEK_LLM_URL not configured in environment")
        llm_url = DEEPSEEK_LLM_URL
        api_key = DEEPSEEK_API_KEY
        print(f"[CONFIG] Using DeepSeek provider: {llm_url}")
    else:  # default or other
        llm_url = LLM_URL
        api_key = OPENAI_API_KEY
        print(f"[CONFIG] Using default provider: {llm_url}")
    
    LLM_URL_CONFIG = llm_url
    EVALUATOR_LLM_URL_CONFIG = llm_url
    CURRENT_API_KEY = api_key or OPENAI_API_KEY  # Fallback to OPENAI_API_KEY if provider-specific key is empty
    
    return llm_url, CURRENT_API_KEY

def get_tokenizer():
    """Lazy load tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH_CONFIG)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using tokenizer."""
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text))
    return len(text) // 4


def estimate_prompt_tokens(
    content: str,
    question: str,
    tools_json: str,
    report: str = "",
    observation: str = ""
) -> dict:
    """
    Estimate token counts for each category in the prompt.
    
    Args:
        content: Complete prompt content
        question: User's question
        tools_json: Tool definitions JSON string
        report: Previous report content (optional)
        observation: Tool observation content (optional)
    
    Returns:
        dict with keys: system, user, tools, report, observation, total
    """
    system_content = content
    for placeholder in [question, tools_json, report, observation]:
        if placeholder:
            system_content = system_content.replace(placeholder, "")
    
    return {
        "system": estimate_tokens(system_content),
        "user": estimate_tokens(question),
        "tools": estimate_tokens(tools_json),
        "report": estimate_tokens(report) if report else 0,
        "observation": estimate_tokens(observation) if observation else 0,
    }


def estimate_tool_definition_tokens(tools_json: str) -> dict:
    """
    Estimate token counts for each tool's definition.
    
    Args:
        tools_json: JSON string containing tool definitions
    
    Returns:
        dict mapping tool name to token count
    """
    try:
        tools = json.loads(tools_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    
    tokenizer = get_tokenizer()
    result = {}
    
    for tool in tools:
        tool_str = json.dumps(tool, ensure_ascii=False)
        if tokenizer:
            tokens = len(tokenizer.encode(tool_str))
        else:
            tokens = len(tool_str) // 4
        name = tool.get("name", "unknown")
        result[name] = tokens
    
    return result


# =============================================================================
# Utility Functions
# =============================================================================
def format_token_bar(value: int, total: int, width: int = 40) -> str:
    """Generate aligned ASCII bar."""
    if total == 0:
        return "░" * width
    ratio = value / total
    filled = int(ratio * width)
    return "█" * filled + "░" * (width - filled)

def extract_tags(text: str, tag: str) -> str:
    """Extract content from XML-style tags."""
    pattern = r"<{TAG}>(.*?)</{TAG}>".format(TAG=tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''


def clean_reasoning(messages: list) -> list:
    """Remove reasoning fields from messages."""
    messages = copy.deepcopy(messages)
    for msg in messages:
        if msg['role'] == 'assistant':
            msg.pop('reasoning', None)
            msg.pop('reasoning_details', None)
    return messages


def random_date(start=datetime(2023, 1, 1), end=datetime(2025, 12, 31)) -> str:
    """Generate a random date string for the agent context."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    final_date = start + timedelta(days=random_days)
    return final_date.strftime("%Y-%m-%d")


def check_report_action(response) -> tuple:
    """
    Validate the response format.
    
    Returns:
        Tuple of (is_valid, error_message).
    """
    text = response.choices[0].message.content
    report = extract_tags(text, 'report')
    action = extract_tags(text, 'tool_call')
    answer = extract_tags(text, 'answer')
    tool_call = None
    print(f"Tool call extracted: {action}")
    if action:
        tool_call = parse_tool_call_payload(action)
        if not (isinstance(tool_call, dict) and 'arguments' in tool_call):
            return False, 'Tool parse error!'
    
    if not report:
        return False, 'Report not found!'
    
    if not action and not answer:
        return False, 'Neither answer nor action found!'
    
    if not answer and not isinstance(tool_call, dict):
        return False, 'No valid answer or tool call!'
    
    return True, 'success'


def _normalize_llm_content_for_check(raw_content: str) -> str:
    """Normalize JSON-mode output to the legacy tag format required by downstream logic."""
    if not isinstance(raw_content, str):
        raw_content = str(raw_content or "")

    content = raw_content.strip()
    if not content:
        return ""

    # Native tag format, keep as-is.
    if "<report>" in content and ("<tool_call>" in content or "<answer>" in content):
        return content

    parsed = _extract_json_object(content)
    if not parsed:
        return content

    # Preferred JSON mode envelope.
    wrapped_content = parsed.get("content")
    if isinstance(wrapped_content, str) and wrapped_content.strip():
        wrapped_content = wrapped_content.strip()
        if "<report>" in wrapped_content and ("<tool_call>" in wrapped_content or "<answer>" in wrapped_content):
            return wrapped_content

    report = parsed.get("report", "")
    tool_call = parsed.get("tool_call", "")
    answer = parsed.get("answer", "")

    if report and not isinstance(report, str):
        report = str(report)
    if answer and not isinstance(answer, str):
        answer = str(answer)
    if isinstance(tool_call, dict):
        tool_call = json.dumps(tool_call, ensure_ascii=False)
    elif tool_call and not isinstance(tool_call, str):
        tool_call = str(tool_call)

    sections = []
    if report:
        sections.append(f"<report>\n{report.strip()}\n</report>")
    if tool_call:
        sections.append(f"<tool_call>\n{tool_call.strip()}\n</tool_call>")
    if answer:
        sections.append(f"<answer>\n{answer.strip()}\n</answer>")

    return "\n\n".join(sections) if sections else content


def parse_tool_call_payload(action_text: str):
    """Parse tool_call payload with tolerant fallbacks for escaped JSON variants."""
    if not action_text or not isinstance(action_text, str):
        return None

    candidate = action_text.strip()
    if not candidate:
        return None

    # 1) Direct JSON object.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2) JSON string that contains an object text.
    try:
        parsed = json.loads(f'"{candidate}"')
        if isinstance(parsed, str):
            nested = json.loads(parsed)
            if isinstance(nested, dict):
                return nested
    except Exception:
        pass

    # 3) Relaxed cleanup for common double-escaped model outputs.
    repaired = candidate
    repaired = repaired.replace('\\"', '"')
    repaired = repaired.replace('\\n', '\n').replace('\\t', '\t')
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 4) Last resort: extract first JSON object snippet.
    parsed = _extract_json_object(repaired)
    if isinstance(parsed, dict) and parsed:
        return parsed

    return None


def _extract_cache_read_tokens(usage) -> int:
    """Extract cached prompt tokens from Aliyun/OpenAI-compatible usage payloads.

    The official Aliyun doc exposes cache hit tokens as:
    usage.prompt_tokens_details.cached_tokens
    """
    if not usage:
        return 0

    def _search(value):
        if isinstance(value, dict):
            prompt_details = value.get("prompt_tokens_details")
            if isinstance(prompt_details, dict):
                cached_tokens = prompt_details.get("cached_tokens")
                if isinstance(cached_tokens, (int, float)):
                    return int(cached_tokens)

            for key in ("cached_tokens", "cache_read_tokens", "cached_prompt_tokens"):
                token_value = value.get(key)
                if isinstance(token_value, (int, float)):
                    return int(token_value)

            for nested_key in ("prompt_token_details", "prompt_tokens_detail", "details"):
                nested_value = value.get(nested_key)
                if nested_value is not None:
                    nested_tokens = _search(nested_value)
                    if nested_tokens:
                        return nested_tokens

        else:
            prompt_details = getattr(value, "prompt_tokens_details", None)
            if prompt_details is not None:
                cached_tokens = getattr(prompt_details, "cached_tokens", None)
                if isinstance(cached_tokens, (int, float)):
                    return int(cached_tokens)

            for key in ("cached_tokens", "cache_read_tokens", "cached_prompt_tokens"):
                token_value = getattr(value, key, None)
                if isinstance(token_value, (int, float)):
                    return int(token_value)

            for nested_key in ("prompt_token_details", "prompt_tokens_detail", "details"):
                nested_value = getattr(value, nested_key, None)
                if nested_value is not None:
                    nested_tokens = _search(nested_value)
                    if nested_tokens:
                        return nested_tokens

        return 0

    return _search(usage)


def _extract_usage_number(usage, *path_candidates) -> int:
    """Extract numeric usage field from dict/object payload by candidate paths."""
    if usage is None:
        return 0

    def _get(value, key):
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    for path in path_candidates:
        current = usage
        found = True
        for key in path:
            current = _get(current, key)
            if current is None:
                found = False
                break
        if found and isinstance(current, (int, float)):
            return int(current)

    return 0


def _extract_cache_write_tokens(usage) -> int:
    """Extract cache-miss/write tokens across DeepSeek/OpenAI-compatible payloads."""
    if not usage:
        return 0

    # DeepSeek official fields
    deepseek_miss = _extract_usage_number(usage, ("prompt_cache_miss_tokens",))
    if deepseek_miss > 0:
        return deepseek_miss

    # Fallback aliases for compatible gateways
    return _extract_usage_number(
        usage,
        ("prompt_tokens_details", "cache_miss_tokens"),
        ("prompt_token_details", "cache_miss_tokens"),
        ("cache_write_tokens",),
        ("cached_prompt_miss_tokens",)
    )


def _parse_usage_tokens(usage) -> dict:
    """Parse prompt/completion/cache tokens from heterogeneous usage payloads."""
    prompt_tokens = _extract_usage_number(
        usage,
        ("prompt_tokens",),
        ("input_tokens",),
        ("usage", "prompt_tokens")
    )
    completion_tokens = _extract_usage_number(
        usage,
        ("completion_tokens",),
        ("output_tokens",),
        ("generated_tokens",),
        ("usage", "completion_tokens")
    )

    cache_read_tokens = _extract_usage_number(
        usage,
        ("prompt_cache_hit_tokens",),  # DeepSeek
        ("prompt_tokens_details", "cached_tokens")  # DashScope/OpenAI-compatible
    )
    if cache_read_tokens == 0:
        cache_read_tokens = _extract_cache_read_tokens(usage)

    cache_write_tokens = _extract_cache_write_tokens(usage)

    # DeepSeek returns hit/miss explicitly; reconstruct prompt_tokens if absent.
    if prompt_tokens == 0 and (cache_read_tokens > 0 or cache_write_tokens > 0):
        prompt_tokens = cache_read_tokens + cache_write_tokens

    # Fallback for some providers that return details-only output token count.
    if completion_tokens == 0:
        completion_tokens = _extract_usage_number(
            usage,
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens")
        )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens
    }


def _detect_provider_from_url(llm_url: str) -> str:
    """Detect provider type by configured base URL/endpoint URL."""
    if not llm_url:
        return "openai-compatible"

    parsed = urlparse(llm_url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    if "deepseek" in host:
        return "deepseek"
    if "dashscope" in host or ("aliyuncs.com" in host and "compatible-mode" in path):
        return "dashscope"
    return "openai-compatible"


def _normalize_llm_endpoint(llm_url: str) -> str:
    """Normalize user-provided endpoint/base_url to chat completions endpoint."""
    if not llm_url:
        return llm_url

    parsed = urlparse(llm_url)
    path = (parsed.path or "").rstrip("/")

    if path.endswith("/chat/completions"):
        return llm_url

    if path.endswith("/v1"):
        new_path = f"{path}/chat/completions"
    elif path.endswith("/compatible-mode"):
        new_path = f"{path}/v1/chat/completions"
    elif path.endswith("/compatible-mode/v1"):
        new_path = f"{path}/chat/completions"
    elif path in ("", "/"):
        new_path = "/chat/completions"
    else:
        new_path = f"{path}/chat/completions"

    return parsed._replace(path=new_path).geturl()


def _adapt_model_for_provider(model_name: str, provider: str) -> str:
    """Adapt default model name for provider-specific endpoints when needed."""
    if provider != "deepseek":
        return model_name

    if not model_name:
        return "deepseek-chat"

    model_lower = model_name.lower()
    if model_lower.startswith("deepseek-"):
        return model_name

    # Keep compatibility with existing defaults in this repo (qwen-*),
    # while DeepSeek API only accepts deepseek-* model ids.
    if model_lower in {"qwen-flash", "qwen-plus", "qwen-max", "qwen-long", "qwen-coder-plus"}:
        print(f"[LLM] DeepSeek endpoint detected, remapping model '{model_name}' -> 'deepseek-chat'")
        return "deepseek-chat"

    return model_name


def _get_effective_tool_calls(tool_name: str, arguments: dict) -> int:
    """Return the weighted call count for tools that accept batched queries.

    Search tools may receive a question/query list. In that case, one tool
    invocation can trigger multiple API requests, so we count the effective
    calls as the length of the query list.
    """
    if tool_name in {'google_search', 'baidu_search', 'aliyun_iqs_search', 'google_scholar'}:
        query = arguments.get('query', []) if isinstance(arguments, dict) else []
        if isinstance(query, (list, tuple)):
            return len(query)
        if isinstance(query, str) and query.strip():
            return 1
        return 0

    return 1


def normalize_reference_answer(answer_value) -> str:
    """Normalize answer/answers field to plain text for evaluation."""
    if answer_value is None:
        return ""
    if isinstance(answer_value, str):
        return answer_value.strip()
    if isinstance(answer_value, (list, tuple)):
        return "\n".join(str(x).strip() for x in answer_value if str(x).strip())
    if isinstance(answer_value, dict):
        return json.dumps(answer_value, ensure_ascii=False)
    return str(answer_value).strip()


def extract_final_answer_text(records: list) -> str:
    """Extract final answer text from assistant records."""
    for message in reversed(records):
        if message.get('role') != 'assistant':
            continue
        content = message.get('content', '') or ''
        tagged_answer = extract_tags(content, 'answer')
        if tagged_answer:
            return tagged_answer.strip()
    return ""


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a text blob."""
    if not text:
        return {}

    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def evaluate_answer_with_llm(
    question: str,
    predicted_answer: str,
    reference_answer: str,
    llm_url: str,
    model: str
) -> dict:
    """Evaluate predicted answer against reference answer using an LLM judge."""
    if not predicted_answer.strip():
        return {
            "evaluated": False,
            "is_correct": False,
            "score": 0.0,
            "reason": "empty_predicted_answer",
            "judge_model": model,
            "judge_raw": ""
        }

    if not reference_answer.strip():
        return {
            "evaluated": False,
            "is_correct": None,
            "score": None,
            "reason": "missing_reference_answer",
            "judge_model": model,
            "judge_raw": ""
        }

    provider = _detect_provider_from_url(llm_url)
    model_name = _adapt_model_for_provider(model, provider)
    endpoint = _normalize_llm_endpoint(llm_url)

    headers = {'Content-Type': 'application/json'}
    if OPENAI_API_KEY:
        headers['Authorization'] = f'Bearer {OPENAI_API_KEY}'

    judge_instruction = (
        "You are a strict evaluator. Compare the predicted answer with the reference answer. "
        "Focus on factual correctness, key entities, numbers, and required conclusions. "
        "Respond with JSON only using this schema: "
        "{\"is_correct\": boolean, \"score\": number, \"reason\": string}. "
        "score must be between 0 and 1."
    )
    judge_input = (
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Predicted Answer:\n{predicted_answer}\n"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": judge_instruction},
            {"role": "user", "content": judge_input}
        ],
        "temperature": 0
    }
    # if provider == "deepseek":
        # payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return {
                "evaluated": False,
                "is_correct": None,
                "score": None,
                "reason": f"judge_http_{resp.status_code}",
                "judge_model": model_name,
                "judge_raw": resp.text[:1000]
            }

        response_json = resp.json()
        content = ""
        try:
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            content = ""

        parsed = _extract_json_object(content)
        if not parsed:
            return {
                "evaluated": False,
                "is_correct": None,
                "score": None,
                "reason": "judge_invalid_json",
                "judge_model": model_name,
                "judge_raw": content[:1000]
            }

        raw_score = parsed.get("score", 0)
        try:
            score = float(raw_score)
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))

        is_correct = parsed.get("is_correct")
        if not isinstance(is_correct, bool):
            is_correct = score >= 0.5

        reason = str(parsed.get("reason", "")).strip()[:1000]

        return {
            "evaluated": True,
            "is_correct": is_correct,
            "score": round(score, 4),
            "reason": reason,
            "judge_model": model_name,
            "judge_raw": content[:2000]
        }
    except Exception as e:
        return {
            "evaluated": False,
            "is_correct": None,
            "score": None,
            "reason": f"judge_exception:{type(e).__name__}",
            "judge_model": model_name,
            "judge_raw": str(e)[:1000]
        }


# =============================================================================
# LLM Interface
# =============================================================================
def call_llm(
    messages: list,
    check_format: bool = False,
    max_retries: int = MAX_FORMAT_RETRIES,
    llm_url: str = None,
    model: str = None,
    turn: int = -1
) -> EasyDict:
    """
    Call the LLM with the given messages.
    
    Args:
        messages: List of chat messages.
        check_format: Whether to validate response format.
        max_retries: Maximum retries for format validation.
        llm_url: URL of the LLM endpoint. If None, uses LLM_URL_CONFIG (current provider).
        model: Model name to use. Falls back to RESEARCH_MODEL global.
        turn: Current iteration turn number for metrics tracking.
        
    Returns:
        LLM response as EasyDict.
    """
    global total_call, failed_call, RESEARCH_MODEL
    
    # Use configured provider URL if not explicitly specified
    if llm_url is None:
        llm_url = LLM_URL_CONFIG
    
    headers = {'Content-Type': 'application/json'}
    if CURRENT_API_KEY:
        headers['Authorization'] = f'Bearer {CURRENT_API_KEY}'
    elif OPENAI_API_KEY:
        headers['Authorization'] = f'Bearer {OPENAI_API_KEY}'
    
    response = None
    call_start_time = time.time()
    
    model_name = model or RESEARCH_MODEL
    provider = _detect_provider_from_url(llm_url)
    model_name = _adapt_model_for_provider(model_name, provider)
    endpoint = _normalize_llm_endpoint(llm_url)
    
    for attempt in range(max_retries):
        try:
            request_messages = messages
            if provider == "deepseek" and check_format:
                json_mode_guard = (
                    "You must output a valid JSON object only. "
                    "Put your full response that follows user-required tags (<report>/<tool_call>/<answer>) "
                    "inside key 'content'."
                )
                request_messages = [{"role": "system", "content": json_mode_guard}] + messages

            payload = {
                "model": model_name,
                "messages": request_messages,
                "temperature": 0.4,
                "top_p": 0.80,
                "presence_penalty": 1.5
            }
            if MAX_COMPLETION_TOKENS_CONFIG and MAX_COMPLETION_TOKENS_CONFIG > 0:
                payload["max_tokens"] = MAX_COMPLETION_TOKENS_CONFIG
            if provider == "deepseek" and check_format:
                payload["response_format"] = {"type": "json_object"}
            # TODO : Add Deepseek Entrance for LLM calls and metrics collection
            llm_call_start = time.time()
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=300)
            llm_call_latency_ms = (time.time() - llm_call_start) * 1000
            
            if resp.status_code != 200:
                print(f"LLM Error: {resp.status_code}: {resp.text}")
                metrics = get_metrics_collector()
                metrics.record_llm_failure(
                    turn=turn,
                    model=model_name,
                    error_type=f"HTTP_{resp.status_code}",
                    error_msg=resp.text
                )
                metrics.record_llm_call(
                    model=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=llm_call_latency_ms,
                    turn=turn,
                    success=False
                )
                continue
            
            response = EasyDict(resp.json())
            assert response.choices[0].message, "No message in response"

            if check_format:
                normalized_content = _normalize_llm_content_for_check(response.choices[0].message.content)
                response.choices[0].message.content = normalized_content
            
            with CALL_COUNTER_LOCK:
                total_call += 1
            
            metrics = get_metrics_collector()
            usage_tokens = _parse_usage_tokens(response.usage if hasattr(response, 'usage') else None)
            prompt_tokens = usage_tokens.get('prompt_tokens', 0)
            completion_tokens = usage_tokens.get('completion_tokens', 0)
            cache_read_tokens = usage_tokens.get('cache_read_tokens', 0)
            cache_write_tokens = usage_tokens.get('cache_write_tokens', 0)

            effective_model = response.get('model', model_name) if isinstance(response, dict) else getattr(response, 'model', model_name)
            
            llm_success = True
            if check_format:
                is_valid, reason = check_report_action(response)
                if not is_valid:
                    with CALL_COUNTER_LOCK:
                        failed_call += 1
                    llm_success = False
                    print(f"Format check failed: {reason}")
                    print(f"Response: {response}")
            
            metrics.record_llm_call(
                model=effective_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=llm_call_latency_ms,
                turn=turn,
                success=llm_success,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens
            )
            
            if not llm_success:
                raise Exception(f"FormatCheckFailed: {reason}")
            
            return response
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            metrics = get_metrics_collector()
            metrics.record_llm_failure(
                turn=turn,
                model=model_name,
                error_type=type(e).__name__,
                error_msg=str(e)
            )
            time.sleep(2)
    
    return response


# =============================================================================
# Tool Execution
# =============================================================================
def _print_search_error(tool_name: str, error_type: str, error_msg: str):
    """Print prominent error message for search failures."""
    print(f"\n{'='*60}")
    print(f"🔴 [{tool_name.upper()}] SEARCH FAILED")
    print(f"{'='*60}")
    print(f"Error Type: {error_type}")
    print(f"Details: {error_msg}")
    print(f"{'='*60}\n")


def _is_search_failure(result: str) -> bool:
    """Check if search result indicates a failure."""
    if not isinstance(result, str):
        return False
    failure_indicators = [
        "Search failed",
        "No results found",
        "Error:",
        "API_KEY",
        "Invalid request",
        "[Search] Error",
        "[BaiduSearch] Error",
        "[AliyunIQS] Error",
        "[Google Scholar] Error",
        "Google Scholar search failed",
        "TimeoutError"
    ]
    return any(result.startswith(indicator) or indicator in result for indicator in failure_indicators)


def _is_tool_failure(result: str, tool_name: str = None) -> bool:
    """Check if tool result indicates a failure."""
    if not isinstance(result, str):
        return False
    if tool_name == 'PythonInterpreter':
        return result.startswith("[Python Interpreter Error]") or "TimeoutError" in result
    if tool_name == 'Visit':
        return result.startswith("[visit] Failed")
    return _is_search_failure(result)


def execute_tool(tool_name: str, arguments: dict) -> tuple:
    """
    Execute a tool with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute.
        arguments: Arguments for the tool.
        
    Returns:
        Tuple of (result_string, summary_messages_list).
    """
    tool_start_time = time.time()
    metrics = get_metrics_collector()
    effective_calls = _get_effective_tool_calls(tool_name, arguments)

    enabled_tools = {tool["name"] for tool in get_tools_for_engine(SEARCH_ENGINE)}
    if tool_name not in enabled_tools:
        msg = f"Tool '{tool_name}' is not enabled in current configuration. Enabled tools: {sorted(enabled_tools)}"
        metrics.record_tool_call(
            tool_name=tool_name,
            args=arguments,
            latency_ms=(time.time() - tool_start_time) * 1000,
            success=False,
            error="ToolNotEnabled",
            effective_calls=effective_calls
        )
        return msg, []
    
    try:
        if tool_name == 'google_search':
            result = google_search_engine.call(arguments)
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_search_failure(result):
                _print_search_error(tool_name, "Search Error", result)
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="SearchFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, []
        
        elif tool_name == 'baidu_search':
            result = baidu_search_engine.call(arguments)
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_search_failure(result):
                _print_search_error(tool_name, "Search Error", result)
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="SearchFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, []
        
        elif tool_name == 'aliyun_iqs_search':
            result = aliyun_iqs_engine.call(arguments)
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_search_failure(result):
                _print_search_error(tool_name, "Search Error", result)
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="SearchFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, []
        
        elif tool_name == 'google_scholar':
            query = arguments.get('query', [])
            if DISABLE_GOOGLE_SCHOLAR:
                result = search_engine.call(arguments)
            else:
                scholar_result = scholar_engine.call({"query": query})
                search_result = google_search_engine.call(arguments)
                result = f"{scholar_result}\n\n{search_result}"
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_search_failure(result):
                _print_search_error(tool_name, "Search Error", result)
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="SearchFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, []
        
        elif tool_name == 'PythonInterpreter':
            result = python_executor.call({"code": arguments.get('code', '')})
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_tool_failure(result, tool_name):
                print(f"\n{'='*60}")
                print(f"🔴 [{tool_name.upper()}] EXECUTION FAILED")
                print(f"{'='*60}")
                print(f"Details: {result}")
                print(f"{'='*60}\n")
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="ExecutionFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, []
        
        elif tool_name == 'Visit':
            result, summary_messages = visit_tool.call(arguments)
            latency_ms = (time.time() - tool_start_time) * 1000
            if _is_tool_failure(result, tool_name):
                print(f"\n{'='*60}")
                print(f"🔴 [{tool_name.upper()}] VISIT FAILED")
                print(f"{'='*60}")
                print(f"Details: {result}")
                print(f"{'='*60}\n")
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=False,
                    error="VisitFailed",
                    effective_calls=effective_calls
                )
            else:
                metrics.record_tool_call(
                    tool_name=tool_name,
                    args=arguments,
                    latency_ms=latency_ms,
                    success=True,
                    effective_calls=effective_calls
                )
            return result, summary_messages
        
        else:
            result = f"Unknown tool: {tool_name}"
            metrics.record_tool_call(
                tool_name=tool_name,
                args=arguments,
                latency_ms=(time.time() - tool_start_time) * 1000,
                success=True,
                effective_calls=effective_calls
            )
            return result, []
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        tool_latency_ms = (time.time() - tool_start_time) * 1000
        
        if tool_name in ('google_search', 'baidu_search', 'aliyun_iqs_search', 'google_scholar'):
            _print_search_error(tool_name, error_type, error_msg)
        
        metrics.record_tool_call(
            tool_name=tool_name,
            args=arguments,
            latency_ms=tool_latency_ms,
            success=False,
            error=error_type,
            effective_calls=effective_calls
        )
        print(f"[{tool_name}] Failure: {error_msg}")
        return f"Tool execution error: {error_msg}", []


# =============================================================================
# Context Formatting
# =============================================================================
def format_context(
    messages: list,
    question: str,
    date: str,
    is_last: bool = False,
    search_engine: str = "google"
) -> str:
    """
    Format the context for the next turn.
    
    Args:
        messages: List containing [user_msg, assistant_msg, tool_msg].
        question: The original question.
        date: Current date string.
        is_last: Whether this is the last turn.
        search_engine: The search engine to use ("google" or "baidu").
        
    Returns:
        Formatted prompt string.
    """
    user, bot, tool = messages[0], messages[1], messages[2]
    
    report = extract_tags(bot['content'], 'report')
    action = extract_tags(bot['content'], 'tool_call')
    tool_response = tool['content']
    
    observation = observation_prompt.format(tool_response=tool_response)
    
    # Truncate observation if too long
    tokenizer = get_tokenizer()
    if tokenizer and len(observation) > 32000:
        tokens = tokenizer.encode(observation)
        if len(tokens) > MAX_OBSERVATION_TOKENS_CONFIG:
            observation = tokenizer.decode(
                tokens[:MAX_OBSERVATION_TOKENS_CONFIG],
                skip_special_tokens=True
            )
            print(f'Observation truncated to {MAX_OBSERVATION_TOKENS_CONFIG} tokens')
    
    tool_str = get_tool_str_for_engine(search_engine)
    
    if is_last:
        return last_instruction_prompt.replace("{question}", question)\
            .replace("{report}", report)\
            .replace("{action}", action)\
            .replace("{observation}", observation)\
            .replace("{tools}", tool_str)\
            .replace('{date_to_use}', date)
    else:
        return instruction_prompt.replace("{question}", question)\
            .replace("{report}", report)\
            .replace("{action}", action)\
            .replace("{observation}", observation)\
            .replace("{tools}", tool_str)\
            .replace('{date_to_use}', date)


# =============================================================================
# Main Agent Loop
# =============================================================================
def agentic_loop(
    data: dict,
    max_format_retries: int = MAX_FORMAT_RETRIES,
    max_turn: int = MAX_TURN,
    llm_url: str = LLM_URL,
    model: str = None
) -> tuple:
    """
    Run the agent loop for a single question.
    
    Args:
        data: Dict containing 'question' and 'answer'/'answers'.
        max_format_retries: Max retries for format validation.
        max_turn: Maximum number of turns.
        llm_url: LLM endpoint URL.
        model: Model name to use for research.
        
    Returns:
        Tuple of (result_dict, full_result_dict).
    """
    task = data['question']
    answer = data.get('answers') or data.get('answer', '')
    
    date = random_date()
    is_last_turn = False
    conversations = []
    
    metrics = get_metrics_collector()
    metrics.start_question(task)
    
    # Initial message
    tool_str_for_engine = get_tool_str_for_engine(SEARCH_ENGINE)
    messages = [{
        "role": "user",
        "content": initial_instruction_prompt.replace('{question}', task)\
            .replace("{tools}", tool_str_for_engine)\
            .replace('{date_to_use}', date)
    }]
    print(messages[0]['content'])
    
    records = copy.deepcopy(messages)
    summary_records = []
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    
    last_tool_call = None

    for turn in range(max_turn):
        print(f"Turn {turn + 1}/{max_turn}")
        
        turn_start_time = time.time()
        metrics.start_iteration(turn=turn, query=task, action="llm_call")
        
        # Track token breakdown and tool definitions
        tool_str = get_tool_str_for_engine(SEARCH_ENGINE)
        if turn == 0:
            # Turn 0: initial prompt
            breakdown = estimate_prompt_tokens(
                messages[0]['content'],
                task,
                tool_str
            )
            # Record tool definitions for Turn 0 only
            tool_def_tokens = estimate_tool_definition_tokens(tool_str)
            for name, tokens in tool_def_tokens.items():
                metrics.record_tool_definition(name, tokens)
        else:
            # Turn N: instruction prompt with report and observation
            content = messages[0]['content']
            report = extract_tags(content, 'report')
            observation = extract_tags(content, 'tool_response')
            breakdown = estimate_prompt_tokens(
                content,
                task,
                tool_str,
                report=report,
                observation=observation
            )
        
        metrics.record_prompt_breakdown(
            system_tokens=breakdown['system'],
            user_tokens=breakdown['user'],
            tools_definition_tokens=breakdown['tools'],
            report_tokens=breakdown['report'],
            observation_tokens=breakdown['observation'],
            turn=turn
        )
        
        # Get LLM response
        response = call_llm(
            messages,
            check_format=True,
            max_retries=max_format_retries,
            llm_url=llm_url,
            model=model,
            turn=turn
        )
        
        if not response:
            print("Failed to get valid response")
            metrics.end_iteration()
            break
        
        cur_message = dict(response.choices[0].message)
        
        # Track token usage
        if hasattr(response, 'usage') and response.usage:
            usage['prompt_tokens'] += response.usage.get('prompt_tokens', 0)
            usage['completion_tokens'] += response.usage.get('completion_tokens', 0)
        
        records.append(cur_message)
        messages.append(cur_message)
        
        # Check for tool call
        tool_call_str = extract_tags(cur_message['content'], 'tool_call')
        tool_call = None
        
        if tool_call_str:
            tool_call = parse_tool_call_payload(tool_call_str)
        
        # If no tool call, agent has finished
        if not tool_call:
            metrics.update_iteration_action(action="final_answer")
            metrics.end_iteration()
            conversations.append(messages)
            break

        last_tool_call = tool_call
        
        # Execute tool
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('arguments', {})
        
        metrics.update_iteration_action(
            action=f"tool_call:{tool_name}",
            tool_name=tool_name,
            tool_args=tool_args
        )
        
        print(f"Calling tool: {tool_name}")
        observation, summary_messages = execute_tool(tool_name, tool_args)
        summary_records.extend(summary_messages)
        
        metrics.update_iteration_tool_result(
            result=observation,
            result_length=len(observation) if observation else 0
        )
        
        # Create tool message
        tool_message = {
            "role": "tool",
            "name": tool_name,
            "content": observation
        }
        
        messages.append(tool_message)
        records.append(tool_message)
        
        metrics.end_iteration()
        
        # Check if this is the second-to-last turn
        if turn == max_turn - 2:
            is_last_turn = True
        
        conversations.append(messages)
        
        # Format context for next turn
        try:
            cur_turn = format_context(messages, task, date, is_last=is_last_turn, search_engine=SEARCH_ENGINE)
        except Exception as e:
            print(f"Context formatting error: {e}")
            break
        
        messages = [{"role": "user", "content": cur_turn}]
    
    final_answer_text = extract_final_answer_text(records)
    reference_answer = normalize_reference_answer(answer)

    evaluation_result = {
        "evaluated": False,
        "is_correct": None,
        "score": None,
        "reason": "evaluator_disabled"
    }
    if EVALUATOR_ENABLED:
        evaluation_result = evaluate_answer_with_llm(
            question=task,
            predicted_answer=final_answer_text,
            reference_answer=reference_answer,
            llm_url=EVALUATOR_LLM_URL_CONFIG,
            model=EVALUATOR_MODEL
        )

    # End question and get metrics
    question_metrics = metrics.end_question(
        final_answer_found=not last_tool_call,
        evaluation=evaluation_result
    )
    
    result = {
        'question': task,
        'answer': answer,
        'records': records,
        'usage': usage,
        'conversations': conversations,
        'evaluation': evaluation_result,
        'metrics': question_metrics
    }
    
    full_result = {
        'question': task,
        'answer': answer,
        'records': records,
        'summary_records': summary_records,
        'usage': usage,
        'conversations': conversations,
        'evaluation': evaluation_result,
        'metrics': question_metrics
    }
    
    return result, full_result


# =============================================================================
# Main Function
# =============================================================================
def main(args):
    """Main entry point."""
    global SEARCH_ENGINE, ACTIVE_TOOL_NAMES, RESEARCH_MODEL, SUMMARY_LLM_URL_CONFIG, SUMMARY_MODEL
    global TOKENIZER_PATH_CONFIG, MAX_OBSERVATION_TOKENS_CONFIG, MAX_WEBPAGE_TOKENS_CONFIG
    global visit_tool, DISABLE_GOOGLE_SCHOLAR
    global EVALUATOR_ENABLED, EVALUATOR_LLM_URL_CONFIG, EVALUATOR_MODEL
    global MAX_COMPLETION_TOKENS_CONFIG, LLM_URL_CONFIG, CURRENT_API_KEY
    
    # Resolve runtime LLM endpoint: provider takes precedence; otherwise use --llm_url.
    if hasattr(args, 'provider') and args.provider:
        configure_provider(args.provider)
    else:
        LLM_URL_CONFIG = args.llm_url
        EVALUATOR_LLM_URL_CONFIG = args.llm_url
        CURRENT_API_KEY = OPENAI_API_KEY
    
    SEARCH_ENGINE = args.search_engine
    try:
        ACTIVE_TOOL_NAMES = _parse_tools_arg(args.tools)
    except ValueError as e:
        print(f"Invalid --tools value: {e}")
        sys.exit(2)

    DISABLE_GOOGLE_SCHOLAR = args.disable_google_scholar
    RESEARCH_MODEL = args.research_model
    SUMMARY_LLM_URL_CONFIG = LLM_URL_CONFIG  # Use unified LLM URL from provider
    SUMMARY_MODEL = args.summary_model
    TOKENIZER_PATH_CONFIG = args.tokenizer_path
    MAX_OBSERVATION_TOKENS_CONFIG = args.max_observation_tokens
    MAX_WEBPAGE_TOKENS_CONFIG = args.max_webpage_tokens
    MAX_COMPLETION_TOKENS_CONFIG = args.max_completion_tokens
    EVALUATOR_ENABLED = not args.disable_evaluator
    EVALUATOR_LLM_URL_CONFIG = LLM_URL_CONFIG  # Use unified LLM URL from provider
    EVALUATOR_MODEL = args.evaluator_model
    
    visit_tool = Visit(
        summary_llm_url=LLM_URL_CONFIG,  # Use unified LLM URL from provider
        summary_llm_auth=CURRENT_API_KEY or SUMMARY_LLM_AUTH,
        summary_model=args.summary_model,
        max_webpage_tokens=args.max_webpage_tokens,
        tokenizer_path=args.tokenizer_path
    )
    
    print(f"Using search engine: {SEARCH_ENGINE}")
    if ACTIVE_TOOL_NAMES:
        print(f"Using tool override (--tools): {ACTIVE_TOOL_NAMES}")
    else:
        print("Using default tool list from --search_engine")
    print(f"Using research model: {RESEARCH_MODEL}")
    print(f"Using summary LLM URL: {SUMMARY_LLM_URL_CONFIG}")
    print(f"Using summary model: {SUMMARY_MODEL}")
    print(f"Using tokenizer path: {TOKENIZER_PATH_CONFIG}")
    print(f"Max observation tokens: {MAX_OBSERVATION_TOKENS_CONFIG}")
    print(f"Max webpage tokens: {MAX_WEBPAGE_TOKENS_CONFIG}")
    print(f"Max completion tokens: {MAX_COMPLETION_TOKENS_CONFIG}")
    print(f"Google Scholar: {'disabled' if DISABLE_GOOGLE_SCHOLAR else 'enabled'}")
    print(f"Evaluator: {'enabled' if EVALUATOR_ENABLED else 'disabled'}")
    if EVALUATOR_ENABLED:
        print(f"Evaluator LLM URL: {EVALUATOR_LLM_URL_CONFIG}")
        print(f"Evaluator model: {EVALUATOR_MODEL}")
    
    # Load input data
    all_data = []
    with open(args.input_fp, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    print(f"Loaded {len(all_data)} questions")
    
    # Setup output path
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_path, exist_ok=True)
    
    output_file = os.path.join(
        args.output_path,
        f"{os.path.basename(args.input_fp)}_{args.prefix}_{current_time}.jsonl"
    )
    full_output_file = os.path.join(
        args.output_path,
        f"{os.path.basename(args.input_fp)}_{args.prefix}_{current_time}_full.jsonl"
    )
    
    # Process questions
    all_results = []
    
    metrics = get_metrics_collector()
    metrics.start_global_timer()
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                agentic_loop,
                data=data,
                max_format_retries=args.max_format_retries,
                max_turn=args.max_turn,
                llm_url=LLM_URL_CONFIG,
                model=args.research_model
            )
            for data in all_data
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result, full_result = future.result()
                all_results.append(result)
                
                # Save incrementally
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                with open(full_output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(full_result, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing question: {e}")
    
    metrics.end_global_timer()
    
    print(f"\nResults saved to: {output_file}")
    print(f"Full results saved to: {full_output_file}")
    print(f"Total: {len(all_results)} questions processed")
    print(f"LLM call success rate: {((total_call - failed_call) / total_call * 100):.2f}%" if total_call > 0 else "N/A")
    
    summary = metrics.get_summary()
    if summary:
        json.dump(summary, open(os.path.join(args.output_path, f"{args.prefix}_summary_{current_time}.json"), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        llm_stats = summary.get('llm', {})
        cache_stats = summary.get('cache', {})
        
        total_input = llm_stats.get('total_prompt_tokens', 0)
        cache_read = cache_stats.get('cache_read_tokens', 0)
        cache_write = cache_stats.get('cache_write_tokens', 0)
        fresh_input = cache_stats.get('fresh_input_tokens', 0)
        completion_tokens = llm_stats.get('total_completion_tokens', 0)
        
        cache_hit_rate = cache_stats.get('cache_hit_rate', 0)
        
        bar_width = 40
        
        print("\n" + "=" * 70)
        print("METRICS SUMMARY")
        print("=" * 70)
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Total Turns: {summary.get('total_turns', 0)}")
        print(f"Avg Turns/Question: {summary.get('avg_turns_per_question', 0)}")
        print(f"Answer Success Rate: {summary.get('answer_success_rate', 'N/A')}")
        eval_stats = summary.get('evaluation', {})
        if eval_stats:
            print(f"Evaluation Coverage: {eval_stats.get('evaluated_questions', 0)}/{summary.get('total_questions', 0)}")
            print(f"Evaluation Accuracy: {eval_stats.get('accuracy_str', 'N/A')}")
            print(f"Evaluation Avg Score: {eval_stats.get('avg_score', 0):.4f}")
        
        print(f"\n{'─' * 70}")
        print("LLM STATISTICS")
        print(f"{'─' * 70}")
        print(f"Total LLM Calls: {llm_stats.get('total_calls', 0)}")
        print(f"Total Prompt Tokens: {llm_stats.get('total_prompt_tokens', 0):,}")
        print(f"Total Completion Tokens: {llm_stats.get('total_completion_tokens', 0):,}")
        print(f"Total LLM Latency: {llm_stats.get('total_latency_ms', 0):.2f}ms")
        
        if llm_stats.get('by_model'):
            print(f"\nBy Model:")
            for model, model_stats in llm_stats.get('by_model', {}).items():
                model_cache_read = model_stats.get('cache_read_tokens', 0)
                model_fresh = model_stats.get('prompt_tokens', 0) - model_cache_read
                print(f"  {model}:")
                print(f"    Calls: {model_stats['calls']}, Prompt: {model_stats['prompt_tokens']:,}, Completion: {model_stats['completion_tokens']:,}")
                if model_cache_read > 0:
                    print(f"    Cache: {model_cache_read:,} read, {model_fresh:,} fresh")
        
        if cache_read > 0 or cache_write > 0 or fresh_input > 0:
            print(f"\n{'─' * 70}")
            print("CACHE EFFICIENCY")
            print(f"{'─' * 70}")
            
            if total_input > 0:
                cache_bar_len = int((cache_read / total_input) * bar_width) if total_input > 0 else 0
                fresh_bar_len = bar_width - cache_bar_len
                cache_pct = (cache_read / total_input) * 100 if total_input > 0 else 0
                fresh_pct = (fresh_input / total_input) * 100 if total_input > 0 else 0
                
                cache_bar = "█" * cache_bar_len + "░" * fresh_bar_len
                print(f"Token Distribution:")
                print(f"  Cache Read: {cache_read:>12,} tokens  {cache_bar}  {cache_pct:5.1f}%")
                fresh_bar = "░" * cache_bar_len + "█" * fresh_bar_len
                print(f"  Fresh Input: {fresh_input:>11,} tokens  {fresh_bar}  {fresh_pct:5.1f}%")
                print(f"{'─' * 70}")
                print(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
        
        prompt_breakdown = summary.get('prompt_breakdown', {})
        if prompt_breakdown.get('total', 0) > 0:
            print(f"\n{'─' * 70}")
            print("TOKEN BREAKDOWN BY CATEGORY")
            print(f"{'─' * 70}")
            print("Estimated using tokenizer analysis of message content:")
            print()
            print("Input Categories:")
            
            total_breakdown = prompt_breakdown.get('total', 0)
            categories = [
                ("SYSTEM", prompt_breakdown.get('system', 0)),
                ("USER", prompt_breakdown.get('user', 0)),
                ("TOOLS", prompt_breakdown.get('tools_definition', 0)),
            ]
            
            if prompt_breakdown.get('report', 0) > 0:
                categories.append(("REPORT", prompt_breakdown.get('report', 0)))
            if prompt_breakdown.get('observation', 0) > 0:
                categories.append(("OBSERVATION", prompt_breakdown.get('observation', 0)))
            
            for name, tokens in categories:
                pct = (tokens / total_breakdown) * 100 if total_breakdown > 0 else 0
                bar = format_token_bar(tokens, total_breakdown, bar_width)
                print(f"  {name:<12} {bar}  {pct:5.1f}% ({tokens:,})")
            
            print()
            print(f"  Subtotal: {total_breakdown:,} estimated input tokens")
        
        tool_defs_cost = summary.get('tool_definitions_cost', {})
        if tool_defs_cost:
            print(f"\n{'─' * 70}")
            print("TOOL DEFINITIONS COST")
            print(f"{'─' * 70}")
            
            total_tool_def_tokens = sum(tool_defs_cost.values())
            
            for tool_name in sorted(tool_defs_cost.keys(), key=lambda x: tool_defs_cost[x], reverse=True):
                tokens = tool_defs_cost[tool_name]
                pct = (tokens / total_tool_def_tokens) * 100 if total_tool_def_tokens > 0 else 0
                bar = format_token_bar(tokens, total_tool_def_tokens, bar_width)
                print(f"  {tool_name:<20} {bar}  {tokens:>10,} tokens")
            
            print(f"{'─' * 70}")
            print(f"  Total: {total_tool_def_tokens:,} tokens")
        
        print(f"\n{'─' * 70}")
        print("SESSION TOTALS")
        print(f"{'─' * 70}")
        print(f"Total API Calls: {llm_stats.get('total_calls', 0)}")
        print(f"  Input tokens (fresh):     {fresh_input:>12,}")
        print(f"  Cache read:               {cache_read:>12,}")
        print(f"  Cache write:              {cache_write:>12,}")
        print(f"  Output tokens:            {completion_tokens:>12,}")
        print(f"{'─' * 70}")
        print(f"Session Total:             {total_input + completion_tokens:>12,} tokens")
        
        try:
            from pricing import calculate_cost, format_cost_usd, MODEL_PRICING
            primary_model = summary.get('primary_model', 'qwen-flash')
            cost_result = calculate_cost(
                primary_model,
                prompt_tokens=fresh_input,
                completion_tokens=completion_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write
            )
            
            if cost_result["total_cost"] > 0:
                print(f"\n{'─' * 70}")
                print("ESTIMATED SESSION COST")
                print(f"{'─' * 70}")
                
                if cost_result.get("warning"):
                    print(f"  Note: {cost_result['warning']}")
                    print(f"  Using estimated pricing...")
                    print()
                
                pricing_info = MODEL_PRICING.get(primary_model.lower(), {})
                input_rate = pricing_info.get("input_cost_per_token", 0) * 1_000_000
                cache_read_rate = pricing_info.get("cache_read_input_token_cost", 0) * 1_000_000
                cache_write_rate = pricing_info.get("cache_creation_input_token_cost", 0) * 1_000_000
                output_rate = pricing_info.get("output_cost_per_token", 0) * 1_000_000
                
                print(f"  Input tokens:  {fresh_input:>10,} × ${input_rate:.4f}/M = ${cost_result['input_cost']:.4f}")
                print(f"  Cache read:    {cache_read:>10,} × ${cache_read_rate:.4f}/M = ${cost_result['cache_read_cost']:.4f}")
                print(f"  Cache write:   {cache_write:>10,} × ${cache_write_rate:.4f}/M = ${cost_result['cache_write_cost']:.4f}")
                print(f"  Output tokens: {completion_tokens:>9,} × ${output_rate:.4f}/M = ${cost_result['output_cost']:.4f}")
                print(f"{'─' * 70}")
                print(f"ESTIMATED TOTAL: ${cost_result['total_cost']:.4f}")
                
                if cache_read > 0 and fresh_input > 0:
                    without_cache = (fresh_input + cache_read) * pricing_info.get("input_cost_per_token", 0) + completion_tokens * pricing_info.get("output_cost_per_token", 0)
                    savings = without_cache - cost_result['total_cost']
                    savings_pct = (savings / without_cache * 100) if without_cache > 0 else 0
                    print(f"\nCost Savings: ${savings:.4f} ({savings_pct:.1f}% reduction with caching)")
        except ImportError:
            pass
        
        print(f"\n{'─' * 70}")
        print("ITERATION DISTRIBUTION")
        print(f"{'─' * 70}")
        iter_dist = summary.get('iteration_distribution', {})
        print(f"  Configured Max Turns: {args.max_turn}")
        print(f"  Min: {iter_dist.get('min', 0)}, Max: {iter_dist.get('max', 0)}, Avg: {iter_dist.get('avg', 0)}")
        
        print(f"\n{'─' * 70}")
        print("TOOL STATISTICS")
        print(f"{'─' * 70}")
        total_effective_search_calls = 0
        total_raw_search_calls = 0
        for tool_name, tool_stats in summary.get('tools', {}).items():
            print(f"  {tool_name}:")
            print(f"    Calls: {tool_stats.get('calls', 0)}, Success: {tool_stats.get('success_rate', 'N/A')}")
            effective_calls = tool_stats.get('effective_calls', 0)
            if effective_calls and effective_calls != tool_stats.get('calls', 0):
                print(f"    Effective Calls: {effective_calls}")
            print(f"    Avg Latency: {tool_stats.get('avg_latency_ms', 0):.2f}ms, Total: {tool_stats.get('total_latency_ms', 0):.2f}ms")
            if tool_stats.get('total_input_tokens', 0) > 0 or tool_stats.get('total_output_tokens', 0) > 0:
                print(f"    Tokens: {tool_stats.get('total_input_tokens', 0)} input + {tool_stats.get('total_output_tokens', 0)} output")
            if tool_name in ('google_search', 'baidu_search', 'aliyun_iqs_search', 'google_scholar'):
                total_raw_search_calls += tool_stats.get('calls', 0)
                total_effective_search_calls += effective_calls

        if total_raw_search_calls > 0:
            print(f"\n{'─' * 70}")
            print("SEARCH CALL STATISTICS")
            print(f"{'─' * 70}")
            print(f"  Raw Search Tool Invocations: {total_raw_search_calls}")
            print(f"  Effective Search API Calls:   {total_effective_search_calls}")
        
        print(f"\n{'─' * 70}")
        print(f"Global Time: {summary.get('global_time_ms', 0):.2f}ms")
        print("=" * 70)
        
        if args.save_metrics:
            try:
                metrics_file = args.save_metrics
                os.makedirs(os.path.dirname(metrics_file) or '.', exist_ok=True)
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                print(f"\nMetrics saved to: {metrics_file}")
            except Exception as e:
                print(f"\nFailed to save metrics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IterResearch: Iterative Research Agent")
    
    parser.add_argument(
        "--input_fp",
        type=str,
        required=True,
        help="Path to input JSONL file containing questions"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="iterresearch",
        help="Prefix for output file names"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--max_format_retries",
        type=int,
        default=MAX_FORMAT_RETRIES,
        help="Maximum retries for format validation"
    )
    parser.add_argument(
        "--max_turn",
        type=int,
        default=MAX_TURN,
        help="Maximum number of turns per question"
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=LLM_URL,
        help="URL of the LLM endpoint"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["aliyun", "deepseek", "default"],
        help="LLM provider to use: 'aliyun', 'deepseek', or 'default'. Overrides --llm_url when set."
    )
    parser.add_argument(
        "--research_model",
        type=str,
        default="qwen-flash",
        help="Name of the research model to use (e.g., qwen-flash, qwen-plus)"
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default="qwen-flash",
        help="Name of the model to use for summarization"
    )
    parser.add_argument(
        "--search_engine",
        type=str,
        default=None,
        choices=["google", "baidu","aliyun"],
        help="Search engine to use: 'google' for Google SerpAPI, 'baidu' for Baidu Qianfan"
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help=(
            "Comma-separated tool list override. "
            "Examples: google,baidu,aliyun,python_interpreter,visit,google_scholar. "
            "When set, this overrides --search_engine tool defaults."
        )
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=TOKENIZER_PATH,
        help="Path to tokenizer model for token counting"
    )
    parser.add_argument(
        "--max_observation_tokens",
        type=int,
        default=MAX_OBSERVATION_TOKENS,
        help="Maximum tokens for observation content"
    )
    parser.add_argument(
        "--max_webpage_tokens",
        type=int,
        default=MAX_WEBPAGE_TOKENS,
        help="Maximum tokens for webpage content"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens for main LLM responses"
    )
    parser.add_argument(
        "--disable_google_scholar",
        action="store_true",
        help="Disable Google Scholar tool, fallback to regular search"
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default=None,
        help="Path to save metrics summary as JSON file (e.g., ./output/metrics.json)"
    )
    parser.add_argument(
        "--disable_evaluator",
        action="store_true",
        help="Disable LLM-based final answer evaluator"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="qwen-flash",
        help="Name of the model to use for final answer evaluation"
    )
    
    args = parser.parse_args()
    main(args)
