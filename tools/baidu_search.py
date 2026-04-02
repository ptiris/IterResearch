"""
Baidu Qianfan Search Tool for IterResearch.

Provides web search functionality through Baidu Qianfan AI Search API.
"""
import json
import time
import requests
from typing import List, Union, Optional

try:
    from config import BAIDU_API_KEY, BAIDU_API_URL
except ImportError:
    import os
    BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")
    BAIDU_API_URL = os.getenv("BAIDU_URL", "https://qianfan.baidubce.com/v2/ai_search/web_search")


class BaiduSearch:
    """
    Web search tool that performs Baidu searches via Qianfan API.
    
    Supports:
    - Single query search
    - Batch queries (multiple queries in one call)
    - Configurable search API backend
    """
    
    name = "baidu_search"
    description = "Performs batched web searches using Baidu Qianfan API: supply an array 'query'; the tool retrieves results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the Baidu search tool.
        
        Args:
            api_key: API key for Baidu Qianfan service. Falls back to config if not provided.
            api_url: API URL for the search service. Falls back to config if not provided.
        """
        self.api_key = api_key or BAIDU_API_KEY
        self.api_url = api_url or BAIDU_API_URL
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{(self.successful_requests / self.total_requests * 100):.2f}%" if self.total_requests > 0 else "0%",
            "avg_latency_ms": round(self.total_latency_ms / self.total_requests, 2) if self.total_requests > 0 else 0,
            "total_latency_ms": round(self.total_latency_ms, 2)
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _search_single(self, query: str) -> str:
        """
        Perform a single search query.
        
        Args:
            query: The search query string.
            
        Returns:
            Formatted search results as a string.
        """
        start_time = time.time()
        self.total_requests += 1
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "resource_type_filter": [{"type": "web", "top_k": 10}]
        }
        
        max_retries = 5
        empty_result_retries = 0
        max_empty_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                response.raise_for_status()
                results = response.json()
                
                references = results.get("references", [])
                
                if len(references) == 0:
                    empty_result_retries += 1
                    if empty_result_retries >= max_empty_retries:
                        self.total_latency_ms += (time.time() - start_time) * 1000
                        return f"No results found for '{query}'. Try with a more general query."
                    time.sleep(1)
                    continue
                
                web_snippets = []
                for idx, ref in enumerate(references, 1):
                    title = ref.get("title", "No title")
                    url = ref.get("url", "")
                    snippet = ref.get("snippet", "")
                    date = ref.get("date", "")
                    website = ref.get("website", "")
                    
                    snippet_parts = []
                    if date:
                        snippet_parts.append(f"Date: {date}")
                    if website:
                        snippet_parts.append(f"Source: {website}")
                    if snippet:
                        snippet_parts.append(snippet)
                    
                    snippet_str = " - ".join(snippet_parts) if snippet_parts else ""
                    formatted = f"{idx}. [{title}]({url})"
                    if snippet_str:
                        formatted += f"\n   {snippet_str}"
                    
                    web_snippets.append(formatted)
                
                self.successful_requests += 1
                self.total_latency_ms += (time.time() - start_time) * 1000
                content = f"A Baidu search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.total_latency_ms += (time.time() - start_time) * 1000
                    self.failed_requests += 1
                    return f"Search failed for '{query}': {str(e)}"
                time.sleep(2)
        
        self.total_latency_ms += (time.time() - start_time) * 1000
        return f"No results found for '{query}'. Try with a more general query."
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute search with given parameters.
        
        Args:
            params: Either a JSON string or dict with 'query' field.
                   Query can be a single string or list of strings.
                   
        Returns:
            Formatted search results.
        """
        if not self.api_key:
            return "[BaiduSearch] Error: BAIDU_API_KEY is not configured. Please set it in environment variables."
        
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "[BaiduSearch] Invalid request format: Input must be a JSON object containing 'query' field"
        
        query = params.get("query")
        if not query:
            return "[BaiduSearch] Invalid request format: Input must contain 'query' field"
        
        if isinstance(query, str):
            return self._search_single(query)
        elif isinstance(query, list):
            responses = [self._search_single(q) for q in query]
            return "\n=======\n".join(responses)
        else:
            return "[BaiduSearch] Invalid query format: must be string or array of strings"


if __name__ == "__main__":
    import os
    if not os.getenv("BAIDU_API_KEY"):
        print("BAIDU_API_KEY not set, skipping test")
    else:
        search_tool = BaiduSearch()
        
        print("--- Testing with a single query ---")
        params = {"query": ["Python programming tutorial"]}
        result = search_tool.call(params)
        print(result)
        print(f"\nStats: {search_tool.get_stats()}")
