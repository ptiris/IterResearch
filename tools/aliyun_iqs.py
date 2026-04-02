"""
Aliyun IQS Search Tool for IterResearch.

Provides web search functionality through Aliyun Information Query Service.
"""
import json
import time
import requests
from typing import List, Union, Optional
from datetime import datetime

try:
    from config import IQS_API_KEY, IQS_BASE_URL
except ImportError:
    import os
    IQS_API_KEY = os.getenv("IQS_API_KEY", "")
    IQS_BASE_URL = os.getenv("IQS_BASE_URL", "https://cloud-iqs.aliyuncs.com")


class AliyunIQSSearch:
    """
    Web search tool that performs searches via Aliyun IQS API.
    
    Supports:
    - Single query search
    - Batch queries (multiple queries in one call)
    - Configurable API endpoint
    """
    
    name = "aliyun_iqs_search"
    description = "阿里云信息查询服务搜索，提供开放域的实时搜索能力。接受查询词数组，一次返回多个查询的搜索结果。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "搜索查询词数组"
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the Aliyun IQS search tool.
        
        Args:
            api_key: API key for Aliyun IQS service. Falls back to config if not provided.
            api_url: API URL for the search service. Falls back to config if not provided.
        """
        self.api_key = api_key or IQS_API_KEY
        self.api_url = api_url or IQS_BASE_URL
        
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
        
        url = f"{self.api_url}/search/unified"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "query": query
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                self.total_latency_ms += (time.time() - start_time) * 1000
                self.failed_requests += 1
                return f"Search failed for '{query}': HTTP {response.status_code}: {response.text}"
            
            search_resp = response.json()
            page_items = search_resp.get("pageItems", [])
            
            if not page_items:
                self.total_latency_ms += (time.time() - start_time) * 1000
                return f"No results found for '{query}'. Try with a more general query."
            
            page_results = []
            for page_item in page_items:
                title = page_item.get('title', '')
                link = page_item.get('link', '')
                snippet = page_item.get('snippet', '')
                main_text = page_item.get('mainText', '')
                published_date = page_item.get('publishedDate', '')
                hostname = page_item.get('hostname', '')
                
                result = (
                    f"Title: {title}\n"
                    f"URL: {link}\n"
                )
                if published_date:
                    result += f"Published Date: {published_date}\n"
                if hostname:
                    result += f"Source: {hostname}\n"
                if snippet:
                    result += f"Snippet: {snippet}\n"
                if main_text:
                    result += f"Text: {main_text}\n"
                
                page_results.append(result)
            
            self.successful_requests += 1
            self.total_latency_ms += (time.time() - start_time) * 1000
            content = f"An Aliyun IQS search for '{query}' found {len(page_results)} results:\n\n" + "---\n".join(page_results)
            return content
            
        except requests.exceptions.Timeout:
            self.total_latency_ms += (time.time() - start_time) * 1000
            self.failed_requests += 1
            return f"Search failed for '{query}': Request timeout"
        except requests.exceptions.RequestException as e:
            self.total_latency_ms += (time.time() - start_time) * 1000
            self.failed_requests += 1
            return f"Search failed for '{query}': {str(e)}"
        except Exception as e:
            self.total_latency_ms += (time.time() - start_time) * 1000
            self.failed_requests += 1
            return f"Search failed for '{query}': {str(e)}"
    
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
            return "[AliyunIQS] Error: IQS_API_KEY is not configured. Please set it in environment variables."
        
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "[AliyunIQS] Invalid request format: Input must be a JSON object containing 'query' field"
        
        query = params.get("query")
        if not query:
            return "[AliyunIQS] Invalid request format: Input must contain 'query' field"
        
        if isinstance(query, str):
            return self._search_single(query)
        elif isinstance(query, list):
            responses = [self._search_single(q) for q in query]
            return "\n=======\n".join(responses)
        else:
            return "[AliyunIQS] Invalid query format: must be string or array of strings"


if __name__ == "__main__":
    import os
    if not os.getenv("IQS_API_KEY"):
        print("IQS_API_KEY not set, skipping test")
    else:
        search_tool = AliyunIQSSearch()
        
        print("--- Testing with a single query ---")
        params = {"query": ["Python programming tutorial"]}
        result = search_tool.call(params)
        print(result)
        print(f"\nStats: {search_tool.get_stats()}")