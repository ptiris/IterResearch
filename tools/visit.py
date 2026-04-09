"""
Web Page Visit Tool for IterResearch.

Provides webpage content extraction and summarization functionality.
"""
import os
import re
import json
import time
import copy
import uuid
import requests
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from config import (
        JINA_API_KEY, 
        SCRAPER_API_KEY, 
        SUMMARY_LLM_URL, 
        SUMMARY_LLM_AUTH,
        MAX_WEBPAGE_TOKENS,
        TOKENIZER_PATH
    )
except ImportError:
    import os
    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")
    SUMMARY_LLM_URL = os.getenv("SUMMARY_LLM_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    SUMMARY_LLM_AUTH = os.getenv("SUMMARY_LLM_AUTH", "")
    MAX_WEBPAGE_TOKENS = int(os.getenv("MAX_WEBPAGE_TOKENS", "48000"))
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")

try:
    from .pdf_parser import download_pdf, parse_pdf
except ImportError:
    from pdf_parser import download_pdf, parse_pdf


# Prompt for extracting useful information from webpage content
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""


def _load_tokenizer(tokenizer_path: str = None):
    """Load tokenizer for token counting."""
    try:
        from transformers import AutoTokenizer
        path = tokenizer_path or TOKENIZER_PATH
        return AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return None


def _parse_json_output(raw: str) -> Optional[dict]:
    """Parse JSON output from LLM response."""
    if not raw:
        return None
        
    # Try to extract from markdown code block
    triple_match = re.search(r'```json\s*\n(.*?)\n```', raw, re.DOTALL)
    if triple_match:
        json_str = triple_match.group(1)
        try:
            import json5
            return json5.loads(json_str)
        except:
            try:
                return json.loads(json_str)
            except:
                return None
    
    # Try direct parsing
    try:
        import json5
        return json5.loads(raw)
    except:
        try:
            return json.loads(raw)
        except:
            return None


def _detect_provider_from_url(llm_url: str) -> str:
    """Detect provider by endpoint URL."""
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
    """Normalize URL to chat completions endpoint."""
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
    """Adapt model name for provider specific constraints."""
    if provider != "deepseek":
        return model_name

    model = (model_name or "").strip().lower()
    if model.startswith("deepseek-"):
        return model_name

    if model in {"qwen-flash", "qwen-plus", "qwen-max", "qwen-long", "qwen-coder-plus"}:
        return "deepseek-chat"

    # Safe fallback for unknown/non-deepseek model ids on DeepSeek endpoint.
    return "deepseek-chat"


class Visit:
    """
    Web page visiting and content extraction tool.
    
    Features:
    - Visit HTML webpages or PDF documents
    - Extract content using Jina Reader API
    - Summarize content using LLM
    - Support for parallel processing of multiple URLs
    """
    
    name = 'visit'
    description = 'Visit webpage(s) or paper(s) and return the summary of the content.'
    parameters = {
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
                "description": "The goal of the visit - what information to extract."
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

    def __init__(
        self,
        jina_api_key: Optional[str] = None,
        scraper_api_key: Optional[str] = None,
        summary_llm_url: Optional[str] = None,
        summary_llm_auth: Optional[str] = None,
        summary_model: Optional[str] = None,
        max_webpage_tokens: int = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize the Visit tool.
        
        Args:
            jina_api_key: API key for Jina Reader.
            scraper_api_key: API key for ScraperAPI.
            summary_llm_url: URL for the summary LLM.
            summary_llm_auth: Authorization header for summary LLM.
            summary_model: Model name for the summary LLM.
            max_webpage_tokens: Maximum tokens for webpage content.
            tokenizer_path: Path to tokenizer model for token counting.
        """
        self.jina_api_key = jina_api_key or JINA_API_KEY
        self.scraper_api_key = scraper_api_key or SCRAPER_API_KEY
        self.summary_llm_url = summary_llm_url or SUMMARY_LLM_URL
        self.summary_llm_auth = summary_llm_auth or SUMMARY_LLM_AUTH
        self.summary_model = summary_model or "qwen-flash"
        self.max_webpage_tokens = max_webpage_tokens or MAX_WEBPAGE_TOKENS
        self.tokenizer = _load_tokenizer(tokenizer_path or TOKENIZER_PATH)
        
        # Stats tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_fetch_time_ms = 0.0
        self.total_summary_llm_time_ms = 0.0
        self.total_summary_input_tokens = 0
        self.total_summary_output_tokens = 0
    
    def get_stats(self) -> dict:
        """Get visit tool statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": f"{(self.successful_calls / self.total_calls * 100):.2f}%" if self.total_calls > 0 else "0%",
            "total_fetch_time_ms": round(self.total_fetch_time_ms, 2),
            "avg_fetch_time_ms": round(self.total_fetch_time_ms / self.total_calls, 2) if self.total_calls > 0 else 0,
            "total_summary_llm_time_ms": round(self.total_summary_llm_time_ms, 2),
            "total_summary_input_tokens": self.total_summary_input_tokens,
            "total_summary_output_tokens": self.total_summary_output_tokens,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_fetch_time_ms = 0.0
        self.total_summary_llm_time_ms = 0.0
        self.total_summary_input_tokens = 0
        self.total_summary_output_tokens = 0

    def call(self, params: Union[str, dict], **kwargs) -> Tuple[str, list]:
        """
        Visit webpage(s) and extract relevant information.
        
        Args:
            params: Parameters containing url, goal, and optional parse_type.
            
        Returns:
            Tuple of (extracted information, summary messages list).
        """
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                return """[Visit] Invalid request format. Required format:
{
    "url": ["url1", "url2"],
    "goal": "what to extract",
    "parse_type": "html" or "pdf"
}""", []
        
        url = params.get("url")
        goal = params.get("goal")
        parse_type = params.get("parse_type", "html")
        
        if not url or not goal:
            return "[Visit] Missing required parameters: 'url' and 'goal' are required.", []
        
        summary_message_list = []
        
        # Handle single URL
        if isinstance(url, str):
            result, summary_msg = self._visit_single(url, goal, parse_type)
            if summary_msg:
                summary_message_list.append(summary_msg)
            return result, summary_message_list
        
        # Handle multiple URLs in parallel
        responses = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._visit_single, u, goal, parse_type): u for u in url}
            for future in as_completed(futures):
                try:
                    result, summary_msg = future.result()
                    responses.append(result)
                    if summary_msg:
                        summary_message_list.append(summary_msg)
                except Exception as e:
                    responses.append(f"Error fetching {futures[future]}: {str(e)}")
        
        return "\n=======\n".join(responses), summary_message_list

    def _visit_single(self, url: str, goal: str, parse_type: str = 'html') -> Tuple[str, Optional[dict]]:
        """
        Visit a single URL and extract information.
        
        Args:
            url: URL to visit.
            goal: What information to extract.
            parse_type: 'html' or 'pdf'.
            
        Returns:
            Tuple of (extracted information, summary message dict).
        """
        max_attempts = 3
        self.total_calls += 1
        
        for attempt in range(max_attempts):
            fetch_start = time.time()
            try:
                # Get content based on type detection
                content_type, content = self._fetch_content(url)
                
                if content_type == 'pdf':
                    content = parse_pdf(content, use_cloud=False)
                elif not content or content.startswith("[visit]"):
                    content = self._jina_fetch(url)
                
                self.total_fetch_time_ms += (time.time() - fetch_start) * 1000
                
                if not content or content.startswith("[visit]"):
                    continue
                
                # Truncate if too long
                if self.tokenizer:
                    tokens = self.tokenizer.encode(content)
                    if len(tokens) > self.max_webpage_tokens:
                        content = self.tokenizer.decode(
                            tokens[:self.max_webpage_tokens],
                            skip_special_tokens=True
                        )
                
                # Extract useful information using LLM
                messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )}]
                
                raw_response, input_tokens, output_tokens, llm_time_ms = self._call_summary_llm(messages)
                
                self.total_summary_input_tokens += input_tokens
                self.total_summary_output_tokens += output_tokens
                self.total_summary_llm_time_ms += llm_time_ms
                
                if not raw_response:
                    useful_info = f"The useful information in '{url}' for user goal '{goal}':\n\n"
                    useful_info += "Evidence: The webpage content could not be processed.\n"
                    useful_info += "Summary: No information available.\n"
                    self.successful_calls += 1
                    return useful_info, None
                
                # Parse the response
                parsed = _parse_json_output(raw_response)
                
                useful_info = f"The useful information in '{url}' for user goal '{goal}':\n\n"
                
                if parsed and 'evidence' in parsed and 'summary' in parsed:
                    useful_info += f"Evidence in page:\n{parsed['evidence']}\n\n"
                    useful_info += f"Summary:\n{parsed['summary']}\n"
                else:
                    useful_info += raw_response
                
                summary_message = copy.deepcopy(messages)
                summary_message.append({"role": "assistant", "content": raw_response})
                
                self.successful_calls += 1
                return useful_info, summary_message
                
            except Exception as e:
                self.total_fetch_time_ms += (time.time() - fetch_start) * 1000
                if attempt == max_attempts - 1:
                    self.failed_calls += 1
                    return f"[visit] Failed to read page (url: {url}, goal: {goal}): {str(e)}", None
        
        self.failed_calls += 1
        return f"[visit] Failed to read page (url: {url}, goal: {goal})", None

    def _fetch_content(self, url: str) -> Tuple[str, str]:
        """
        Fetch content from URL, detecting if it's PDF or HTML.
        
        Returns:
            Tuple of (content_type, content_or_path).
        """
        if not self.scraper_api_key:
            return 'html', ''
        
        output_dir = "/tmp/pdf"
        os.makedirs(output_dir, exist_ok=True)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"document_{current_time}_{str(uuid.uuid4())}.pdf"
        file_path = os.path.join(output_dir, filename)
        
        scraper_params = {
            'api_key': self.scraper_api_key,
            'url': url,
            'device_type': 'desktop',
            'country_code': 'us',
            'output_format': 'markdown'
        }
        
        for attempt in range(3):
            try:
                response = requests.get(
                    'https://api.scraperapi.com/',
                    params=scraper_params,
                    timeout=60
                )
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                
                if 'pdf' in content_type:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    return 'pdf', file_path
                
                if 'text' in content_type:
                    return 'html', response.text
                
                return 'other', ''
                
            except Exception as e:
                if attempt == 2:
                    return 'other', ''
                time.sleep(1)
        
        return 'other', ''

    def _jina_fetch(self, url: str) -> str:
        """
        Fetch webpage content using Jina Reader API.
        
        Args:
            url: URL to fetch.
            
        Returns:
            Webpage content as text.
        """
        if not self.jina_api_key:
            # Try direct fetch without Jina
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
            except:
                return "[visit] Failed to read page."
        
        headers = {"Authorization": f"Bearer {self.jina_api_key}"}
        
        for attempt in range(3):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.text
            except:
                pass
            time.sleep(2)
        
        return "[visit] Failed to read page."

    def _call_summary_llm(self, messages: list) -> Tuple[Optional[str], int, int, float]:
        """
        Call the summary LLM to extract information.
        
        Args:
            messages: Chat messages for the LLM.
            
        Returns:
            Tuple of (LLM response text or None, input tokens, output tokens, latency_ms).
        """
        if not self.summary_llm_url:
            return None, 0, 0, 0.0

        provider = _detect_provider_from_url(self.summary_llm_url)
        endpoint = _normalize_llm_endpoint(self.summary_llm_url)
        model_name = _adapt_model_for_provider(self.summary_model, provider)

        print("=== Calling summary LLM ===")
        print(endpoint)
        headers = {'Content-Type': 'application/json'}
        if self.summary_llm_auth:
            auth = self.summary_llm_auth.strip()
            if auth.lower().startswith('bearer '):
                headers['Authorization'] = auth
            else:
                headers['Authorization'] = f"Bearer {auth}"
        
        # DeepSeek often rejects overly large max_tokens; keep a safe cap.
        max_tokens = 24000
        if provider == "deepseek":
            max_tokens = 8000

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": max_tokens,
        }
        if provider == "deepseek":
            payload["response_format"] = {"type": "json_object"}
        
        start_time = time.time()
        
        for attempt in range(5):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = 0
                output_tokens = 0
                
                if 'usage' in data:
                    input_tokens = data['usage'].get('prompt_tokens', 0)
                    output_tokens = data['usage'].get('completion_tokens', 0)
                
                return data['choices'][0]['message']['content'], input_tokens, output_tokens, latency_ms
            except Exception as e:
                if attempt == 4:
                    print(f"Summary LLM call failed: {e}")
                time.sleep(2)
        
        latency_ms = (time.time() - start_time) * 1000
        return None, 0, 0, latency_ms


if __name__ == "__main__":
    # Example usage
    tool = Visit()
    
    result, _ = tool.call({
        "url": ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        "goal": "What are the main features and history of Python programming language?"
    })
    
    print(result)
