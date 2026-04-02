"""
Configuration file for IterResearch.

All configurations can be overridden by environment variables.
"""
import os

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LLM Configuration
# =============================================================================
# Main LLM endpoint for agent reasoning
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:10086/v1/chat/completions")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Summary LLM endpoint for webpage content extraction
SUMMARY_LLM_URL = os.getenv("SUMMARY_LLM_URL", "http://127.0.0.1:10086/v1/chat/completions")
SUMMARY_LLM_AUTH = os.getenv("SUMMARY_LLM_AUTH", "")

# =============================================================================
# Search API Configuration
# =============================================================================
# Google Search API (via SerpAPI or similar service)
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://serpapi.com/search")

# Google Scholar API
SCHOLAR_API_KEY = os.getenv("SCHOLAR_API_KEY", "")
SCHOLAR_API_URL = os.getenv("SCHOLAR_API_URL", "https://serpapi.com/search")

# Baidu Qianfan Search API
BAIDU_API_KEY = os.getenv("BAIDU_API", os.getenv("BAIDU_API_KEY", ""))
BAIDU_API_URL = os.getenv("BAIDU_URL", "https://qianfan.baidubce.com/v2/ai_search/web_search")

# Aliyun IQS API
IQS_API_KEY = os.getenv("IQS_API_KEY", "")
IQS_BASE_URL = os.getenv("IQS_BASE_URL", "https://cloud-iqs.aliyuncs.com")

# =============================================================================
# Webpage Reading Configuration
# =============================================================================
# Jina Reader API for webpage content extraction
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# ScraperAPI for proxy-based web scraping
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")

# =============================================================================
# Python Sandbox Configuration
# =============================================================================
# Sandbox endpoint for code execution (e.g., sandbox-fusion)
SANDBOX_ENDPOINTS = os.getenv("SANDBOX_ENDPOINTS", "http://127.0.0.1:8080").split(",")


# =============================================================================
# Tokenizer Configuration
# =============================================================================
# Path to tokenizer model for token counting
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")

# =============================================================================
# Agent Configuration
# =============================================================================
# Maximum number of turns for the agent loop
MAX_TURN = int(os.getenv("MAX_TURN", "25"))

# Maximum retries for format checking
MAX_FORMAT_RETRIES = int(os.getenv("MAX_FORMAT_RETRIES", "5"))

# Maximum workers for parallel processing
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))

# Maximum observation length (in tokens)
MAX_OBSERVATION_TOKENS = int(os.getenv("MAX_OBSERVATION_TOKENS", "32000"))

# Maximum webpage content length (in tokens)
MAX_WEBPAGE_TOKENS = int(os.getenv("MAX_WEBPAGE_TOKENS", "48000"))
