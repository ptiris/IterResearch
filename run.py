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
    SUMMARY_LLM_AUTH
)
from prompts import (
    initial_instruction_prompt,
    instruction_prompt,
    observation_prompt,
    last_instruction_prompt
)
from tools import Search, BaiduSearch, Scholar, PythonInterpreter, Visit
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


# =============================================================================
# Global Variables
# =============================================================================


def get_tools_for_engine(search_engine: str) -> list:
    """Get filtered TOOLS list based on selected search engine."""
    base_tools = [VISIT_TOOL, PYTHON_INTERPRETER_TOOL]
    
    if search_engine == "google":
        return base_tools + [GOOGLE_SEARCH_TOOL, GOOGLE_SCHOLAR_TOOL]
    elif search_engine == "baidu":
        return base_tools + [BAIDU_SEARCH_TOOL]
    
    return base_tools + [GOOGLE_SEARCH_TOOL, BAIDU_SEARCH_TOOL, GOOGLE_SCHOLAR_TOOL]


def get_tool_str_for_engine(search_engine: str) -> str:
    """Get filtered TOOLS JSON string based on selected search engine."""
    filtered_tools = get_tools_for_engine(search_engine)
    return json.dumps(filtered_tools, indent=2)


# Initialize tools
python_executor = PythonInterpreter()
google_search_engine = Search()
baidu_search_engine = BaiduSearch()
search_engine = google_search_engine
scholar_engine = Scholar()
visit_tool = Visit()

SEARCH_ENGINE = "google"
RESEARCH_MODEL = "qwen-flash"
SUMMARY_LLM_URL_CONFIG = SUMMARY_LLM_URL
SUMMARY_MODEL = "qwen-flash"
TOKENIZER_PATH_CONFIG = TOKENIZER_PATH
MAX_OBSERVATION_TOKENS_CONFIG = MAX_OBSERVATION_TOKENS
MAX_WEBPAGE_TOKENS_CONFIG = MAX_WEBPAGE_TOKENS
DISABLE_GOOGLE_SCHOLAR = False

# Statistics
failed_call = 0
total_call = 0

# Tokenizer for observation length control
_tokenizer = None

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


# =============================================================================
# Utility Functions
# =============================================================================
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
    print(f"Tool call extracted: {action}")
    if action:
        try:
            tool_call = json.loads(action)
            assert isinstance(tool_call, dict)
            assert 'arguments' in tool_call
        except:
            return False, 'Tool parse error!'
    
    if not report:
        return False, 'Report not found!'
    
    if not action and not answer:
        return False, 'Neither answer nor action found!'
    
    if not answer and not isinstance(tool_call, dict):
        return False, 'No valid answer or tool call!'
    
    return True, 'success'


# =============================================================================
# LLM Interface
# =============================================================================
def call_llm(
    messages: list,
    check_format: bool = False,
    max_retries: int = MAX_FORMAT_RETRIES,
    llm_url: str = LLM_URL,
    model: str = None,
    turn: int = -1
) -> EasyDict:
    """
    Call the LLM with the given messages.
    
    Args:
        messages: List of chat messages.
        check_format: Whether to validate response format.
        max_retries: Maximum retries for format validation.
        llm_url: URL of the LLM endpoint.
        model: Model name to use. Falls back to RESEARCH_MODEL global.
        turn: Current iteration turn number for metrics tracking.
        
    Returns:
        LLM response as EasyDict.
    """
    global total_call, failed_call, RESEARCH_MODEL
    
    headers = {'Content-Type': 'application/json'}
    if OPENAI_API_KEY:
        headers['Authorization'] = f'Bearer {OPENAI_API_KEY}'
    
    response = None
    call_start_time = time.time()
    
    model_name = model or RESEARCH_MODEL
    
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 1.5
            }
            
            llm_call_start = time.time()
            resp = requests.post(llm_url, headers=headers, json=payload, timeout=300)
            llm_call_latency_ms = (time.time() - llm_call_start) * 1000
            
            if resp.status_code != 200:
                print(f"LLM Error: {resp.text}")
                continue
            
            response = EasyDict(resp.json())
            assert response.choices[0].message, "No message in response"
            
            total_call += 1
            
            if check_format:
                is_valid, reason = check_report_action(response)
                if not is_valid:
                    failed_call += 1
                    print(f"Format check failed: {reason}")
                    print(f"Response: {response}")
                    raise Exception(reason)
            
            metrics = get_metrics_collector()
            prompt_tokens = response.usage.get('prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            completion_tokens = response.usage.get('completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            total_latency_ms = (time.time() - call_start_time) * 1000
            
            metrics.record_llm_call(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=llm_call_latency_ms,
                turn=turn
            )
            
            return response
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    
    return response


# =============================================================================
# Tool Execution
# =============================================================================
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
    tool_success = False
    
    try:
        if tool_name == 'google_search':
            result = google_search_engine.call(arguments)
            tool_success = True
            return result, []
        
        elif tool_name == 'baidu_search':
            result = baidu_search_engine.call(arguments)
            tool_success = True
            return result, []
        
        elif tool_name == 'google_scholar':
            query = arguments.get('query', [])
            if DISABLE_GOOGLE_SCHOLAR:
                result = search_engine.call(arguments)
            else:
                scholar_result = scholar_engine.call({"query": query})
                search_result = google_search_engine.call(arguments)
                result = f"{scholar_result}\n\n{search_result}"
            tool_success = True
            return result, []
        
        elif tool_name == 'PythonInterpreter':
            result = python_executor.call({"code": arguments.get('code', '')})
            tool_success = True
            return result, []
        
        elif tool_name == 'Visit':
            result, summary_messages = visit_tool.call(arguments)
            tool_success = True
            return result, summary_messages
        
        else:
            tool_success = True
            return f"Unknown tool: {tool_name}", []
            
    except Exception as e:
        tool_latency_ms = (time.time() - tool_start_time) * 1000
        metrics.record_tool_call(
            tool_name=tool_name,
            args=arguments,
            latency_ms=tool_latency_ms,
            success=False,
            error=type(e).__name__
        )
        print(f"[{tool_name}] Failure: {str(e)}")
        return f"Tool execution error: {str(e)}", []
    
    finally:
        tool_latency_ms = (time.time() - tool_start_time) * 1000
        if tool_success:
            metrics.record_tool_call(
                tool_name=tool_name,
                args=arguments,
                latency_ms=tool_latency_ms,
                success=True
            )


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
    
    for turn in range(max_turn):
        print(f"Turn {turn + 1}/{max_turn}")
        
        turn_start_time = time.time()
        metrics.start_iteration(turn=turn, query=task, action="llm_call")
        
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
            try:
                tool_call = json.loads(tool_call_str)
            except:
                pass
        
        # If no tool call, agent has finished
        if not tool_call:
            metrics.update_iteration_action(action="final_answer")
            metrics.end_iteration()
            conversations.append(messages)
            break
        
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
    
    # End question and get metrics
    question_metrics = metrics.end_question(final_answer_found=not tool_call)
    
    result = {
        'question': task,
        'answer': answer,
        'records': records,
        'usage': usage,
        'conversations': conversations,
        'metrics': question_metrics
    }
    
    full_result = {
        'question': task,
        'answer': answer,
        'records': records,
        'summary_records': summary_records,
        'usage': usage,
        'conversations': conversations,
        'metrics': question_metrics
    }
    
    return result, full_result


# =============================================================================
# Main Function
# =============================================================================
def main(args):
    """Main entry point."""
    global SEARCH_ENGINE, RESEARCH_MODEL, SUMMARY_LLM_URL_CONFIG, SUMMARY_MODEL
    global TOKENIZER_PATH_CONFIG, MAX_OBSERVATION_TOKENS_CONFIG, MAX_WEBPAGE_TOKENS_CONFIG
    global visit_tool, DISABLE_GOOGLE_SCHOLAR
    
    SEARCH_ENGINE = args.search_engine
    DISABLE_GOOGLE_SCHOLAR = args.disable_google_scholar
    RESEARCH_MODEL = args.research_model
    SUMMARY_LLM_URL_CONFIG = args.summary_llm_url
    SUMMARY_MODEL = args.summary_model
    TOKENIZER_PATH_CONFIG = args.tokenizer_path
    MAX_OBSERVATION_TOKENS_CONFIG = args.max_observation_tokens
    MAX_WEBPAGE_TOKENS_CONFIG = args.max_webpage_tokens
    
    visit_tool = Visit(
        summary_llm_url=args.summary_llm_url,
        summary_llm_auth=SUMMARY_LLM_AUTH,
        summary_model=args.summary_model,
        max_webpage_tokens=args.max_webpage_tokens,
        tokenizer_path=args.tokenizer_path
    )
    
    print(f"Using search engine: {SEARCH_ENGINE}")
    print(f"Using research model: {RESEARCH_MODEL}")
    print(f"Using summary LLM URL: {SUMMARY_LLM_URL_CONFIG}")
    print(f"Using summary model: {SUMMARY_MODEL}")
    print(f"Using tokenizer path: {TOKENIZER_PATH_CONFIG}")
    print(f"Max observation tokens: {MAX_OBSERVATION_TOKENS_CONFIG}")
    print(f"Max webpage tokens: {MAX_WEBPAGE_TOKENS_CONFIG}")
    print(f"Google Scholar: {'disabled' if DISABLE_GOOGLE_SCHOLAR else 'enabled'}")
    
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
                llm_url=args.llm_url,
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
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        print(f"Total Questions: {summary.get('total_questions', 0)}")
        print(f"Total Turns: {summary.get('total_turns', 0)}")
        print(f"Avg Turns/Question: {summary.get('avg_turns_per_question', 0)}")
        print(f"Answer Success Rate: {summary.get('answer_success_rate', 'N/A')}")
        print(f"\nLLM Statistics:")
        llm_stats = summary.get('llm', {})
        print(f"  Total LLM Calls: {llm_stats.get('total_calls', 0)}")
        print(f"  Total Prompt Tokens: {llm_stats.get('total_prompt_tokens', 0)}")
        print(f"  Total Completion Tokens: {llm_stats.get('total_completion_tokens', 0)}")
        print(f"  Total LLM Latency: {llm_stats.get('total_latency_ms', 0):.2f}ms")
        print(f"  By Model:")
        for model, model_stats in llm_stats.get('by_model', {}).items():
            print(f"    {model}: {model_stats['calls']} calls, {model_stats['prompt_tokens']} prompt + {model_stats['completion_tokens']} completion tokens")
        print(f"\nIteration Distribution:")
        iter_dist = summary.get('iteration_distribution', {})
        print(f"  Min: {iter_dist.get('min', 0)}, Max: {iter_dist.get('max', 0)}, Avg: {iter_dist.get('avg', 0)}")
        print(f"\nTool Statistics:")
        for tool_name, tool_stats in summary.get('tools', {}).items():
            print(f"  {tool_name}:")
            print(f"    Calls: {tool_stats.get('calls', 0)}, Success: {tool_stats.get('success_rate', 'N/A')}")
            print(f"    Avg Latency: {tool_stats.get('avg_latency_ms', 0):.2f}ms, Total: {tool_stats.get('total_latency_ms', 0):.2f}ms")
            if tool_stats.get('total_input_tokens', 0) > 0 or tool_stats.get('total_output_tokens', 0) > 0:
                print(f"    Tokens: {tool_stats.get('total_input_tokens', 0)} input + {tool_stats.get('total_output_tokens', 0)} output")
        print(f"\nGlobal Time: {summary.get('global_time_ms', 0):.2f}ms")


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
        default="google",
        choices=["google", "baidu"],
        help="Search engine to use: 'google' for Google SerpAPI, 'baidu' for Baidu Qianfan"
    )
    parser.add_argument(
        "--summary_llm_url",
        type=str,
        default=SUMMARY_LLM_URL,

        help="URL of the summary LLM endpoint"
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
        "--disable_google_scholar",
        action="store_true",
        help="Disable Google Scholar tool, fallback to regular search"
    )
    
    args = parser.parse_args()
    main(args)
