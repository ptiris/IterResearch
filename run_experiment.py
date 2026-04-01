"""
Experiment script for running IterResearch on BC-zh dataset.

This script:
1. Extracts 10 questions from data/BC-zn/browsecomp_zh.jsonl to a temp file
2. Runs run.py with multiple workers for parallelism
3. Reads results and creates statics.csv with break_down data
4. Uses pairwise evaluation with LLM as judge to get overall accuracy
"""
import os
import sys
import json
import csv
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


PAIRWISE_EVAL_PROMPT = """你是一个专业的问答质量评估专家。你的任务是比较两个答案的质量，判断哪个答案更准确地回答了问题。

## 问题
{question}

## 标准答案
{ground_truth}

## 答案A (来自研究Agent)
{answer_a}

## 答案B (来自标准答案/基线)
{answer_b}

## 评估标准
请从以下几个方面评估答案质量：
1. 准确性：答案是否正确回答了问题
2. 完整性：答案是否涵盖了问题的所有方面
3. 相关性：答案是否与问题直接相关

## 输出要求
请选择表现更好的答案：
- 如果答案A更好，输出 "A"
- 如果答案B更好，输出 "B"
- 如果两者质量相当，输出 "TIE"

只需输出一个字符（A、B 或 TIE），不要添加任何解释。"""


def load_questions(data_path: str, num_questions: int = 10):
    """Load questions from JSONL file."""
    questions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_questions:
                break
            if line.strip():
                q = json.loads(line)
                questions.append({
                    'question': q.get('Question') or q.get('question', ''),
                    'answer': q.get('Answer') or q.get('answer', ''),
                    'topic': q.get('Topic', '')
                })
    return questions


def call_llm_for_eval(messages: list, llm_url: str, model: str = None) -> str:
    """Call LLM for pairwise evaluation."""
    import requests
    from easydict import EasyDict
    
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "model": model or "qwen-flash",
        "messages": messages,
        "temperature": 0.0,
    }
    
    try:
        resp = requests.post(llm_url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            response = EasyDict(resp.json())
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM evaluation error: {e}")
    return "ERROR"


def pairwise_evaluate(results: list, llm_url: str, eval_model: str = None) -> dict:
    """
    Use pairwise comparison to evaluate results.
    
    Compares each agent answer against ground truth.
    Returns overall accuracy based on how many times agent wins.
    """
    wins = 0
    losses = 0
    ties = 0
    errors = 0
    
    for item in tqdm(results, desc="Pairwise Evaluation"):
        question = item.get('question', '')
        ground_truth = item.get('answer', '')
        agent_answer = item.get('predicted_answer', '')
        
        if not agent_answer or agent_answer == "ERROR":
            errors += 1
            continue
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的问答质量评估专家。"
            },
            {
                "role": "user",
                "content": PAIRWISE_EVAL_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    answer_a=agent_answer,
                    answer_b=ground_truth
                )
            }
        ]
        
        result = call_llm_for_eval(messages, llm_url, eval_model)
        
        if result == "A":
            wins += 1
        elif result == "B":
            losses += 1
        elif result == "TIE":
            ties += 1
        else:
            errors += 1
    
    total_valid = wins + losses + ties
    accuracy = (wins + ties * 0.5) / total_valid if total_valid > 0 else 0.0
    
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "errors": errors,
        "total": len(results),
        "accuracy": accuracy,
        "accuracy_percent": f"{accuracy * 100:.2f}%"
    }


def extract_answer_from_record(record: dict) -> str:
    """Extract answer from assistant record."""
    import re
    content = record.get('content', '')
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "ERROR"


def read_results(results_file: str) -> list:
    """Read results from run.py output JSONL file."""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                records = data.get('records', [])
                predicted = "ERROR"
                for record in reversed(records):
                    if record.get('role') == 'assistant':
                        content = record.get('content', '')
                        import re
                        match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                        if match:
                            predicted = match.group(1).strip()
                            break
                results.append({
                    'question': data.get('question', ''),
                    'answer': data.get('answer', ''),
                    'predicted_answer': predicted,
                    'records': records,
                    'usage': data.get('usage', {}),
                    'metrics': data.get('metrics', {})
                })
    return results


def write_statics_csv(output_file: str, results: list):
    """Write statistics to CSV file with break_down data."""
    
    fieldnames = [
        'question_id',
        'question',
        'ground_truth',
        'predicted_answer',
        'total_turns',
        'llm_total_calls',
        'llm_total_tokens',
        'llm_total_latency_ms',
        'tool_total_calls',
        'tool_total_latency_ms',
        'total_time_ms'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, result in enumerate(results):
            q_metrics = result.get('metrics', {})
            llm_metrics = q_metrics.get('llm', {})
            tool_usage = q_metrics.get('tool_usage', {})
            latency_breakdown = q_metrics.get('latency_breakdown', {})
            
            total_tool_calls = sum(t.get('calls', 0) for t in tool_usage.values())
            total_tool_latency = sum(t.get('total_latency_ms', 0) for t in tool_usage.values())
            
            writer.writerow({
                'question_id': i + 1,
                'question': result.get('question', '')[:100],
                'ground_truth': result.get('answer', ''),
                'predicted_answer': result.get('predicted_answer', 'ERROR')[:100],
                'total_turns': q_metrics.get('total_turns', 0),
                'llm_total_calls': llm_metrics.get('total_calls', 0),
                'llm_total_tokens': llm_metrics.get('total_tokens', 0),
                'llm_total_latency_ms': latency_breakdown.get('llm_time_ms', 0),
                'tool_total_calls': total_tool_calls,
                'tool_total_latency_ms': total_tool_latency,
                'total_time_ms': latency_breakdown.get('total_time_ms', 0)
            })
    
    summary_file = output_file.replace('.csv', '_summary.csv')
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        summary_fieldnames = ['metric_name', 'metric_value']
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        
        total_questions = len(results)
        total_turns = sum(r.get('metrics', {}).get('total_turns', 0) for r in results)
        
        all_llm_calls = sum(r.get('metrics', {}).get('llm', {}).get('total_calls', 0) for r in results)
        all_prompt_tokens = sum(r.get('metrics', {}).get('llm', {}).get('total_prompt_tokens', 0) for r in results)
        all_completion_tokens = sum(r.get('metrics', {}).get('llm', {}).get('total_completion_tokens', 0) for r in results)
        
        iter_turns = [r.get('metrics', {}).get('total_turns', 0) for r in results]
        iter_min = min(iter_turns) if iter_turns else 0
        iter_max = max(iter_turns) if iter_turns else 0
        iter_avg = sum(iter_turns) / len(iter_turns) if iter_turns else 0
        
        writer.writerow({'metric_name': 'total_questions', 'metric_value': total_questions})
        writer.writerow({'metric_name': 'total_turns', 'metric_value': total_turns})
        writer.writerow({'metric_name': 'avg_turns_per_question', 'metric_value': round(iter_avg, 2)})
        writer.writerow({'metric_name': 'total_llm_calls', 'metric_value': all_llm_calls})
        writer.writerow({'metric_name': 'total_prompt_tokens', 'metric_value': all_prompt_tokens})
        writer.writerow({'metric_name': 'total_completion_tokens', 'metric_value': all_completion_tokens})
        writer.writerow({'metric_name': 'total_tokens', 'metric_value': all_prompt_tokens + all_completion_tokens})
        writer.writerow({'metric_name': 'iteration_min', 'metric_value': iter_min})
        writer.writerow({'metric_name': 'iteration_max', 'metric_value': iter_max})
        writer.writerow({'metric_name': 'iteration_avg', 'metric_value': round(iter_avg, 2)})
        
        all_tools = {}
        for r in results:
            for tool_name, tool_data in r.get('metrics', {}).get('tool_usage', {}).items():
                if tool_name not in all_tools:
                    all_tools[tool_name] = {'calls': 0, 'success': 0, 'failed': 0}
                all_tools[tool_name]['calls'] += tool_data.get('calls', 0)
                all_tools[tool_name]['success'] += tool_data.get('success_count', 0)
                all_tools[tool_name]['failed'] += tool_data.get('failed_count', 0)
        
        for tool_name, tool_stats in all_tools.items():
            success_rate = tool_stats['success'] / tool_stats['calls'] * 100 if tool_stats['calls'] > 0 else 0
            writer.writerow({'metric_name': f'tool_{tool_name}_calls', 'metric_value': tool_stats['calls']})
            writer.writerow({'metric_name': f'tool_{tool_name}_success_rate', 'metric_value': f'{success_rate:.2f}%'})


def run_experiment(
    data_path: str,
    output_dir: str,
    num_questions: int = 10,
    max_turn: int = 25,
    llm_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    research_model: str = "qwen-flash",
    eval_model: str = None,
    max_workers: int = 4
):
    """Run the full experiment pipeline."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {num_questions} questions from {data_path}")
    questions = load_questions(data_path, num_questions)
    print(f"Loaded {len(questions)} questions")
    
    input_file = os.path.join(output_dir, f"input_{timestamp}.jsonl")
    print(f"Saving questions to {input_file}")
    with open(input_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"\nRunning run.py with {max_workers} workers...")
    cmd = [
        sys.executable, "run.py",
        "--input_fp", input_file,
        "--output_path", output_dir,
        "--prefix", f"exp_{timestamp}",
        "--max_workers", str(max_workers),
        "--max_turn", str(max_turn),
        "--llm_url", llm_url,
        "--research_model", research_model,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        print(f"run.py failed with return code {result.returncode}")
        return None, None, None
    
    result_files = [f for f in os.listdir(output_dir) if f.startswith(f'browsecomp_zh_questions_exp_{timestamp}') and not '_full' in f]
    if not result_files:
        print(f"No result files found in {output_dir}")
        return None, None, None
    
    result_file = os.path.join(output_dir, result_files[0])
    print(f"\nReading results from {result_file}")
    
    all_results = read_results(result_file)
    print(f"Read {len(all_results)} results")
    
    statics_file = os.path.join(output_dir, f"statics_{timestamp}.csv")
    print(f"Writing statistics to {statics_file}")
    write_statics_csv(statics_file, all_results)
    
    print("\n" + "=" * 60)
    print("PAIRWISE EVALUATION")
    print("=" * 60)
    
    eval_results = pairwise_evaluate(all_results, llm_url, eval_model)
    
    print(f"\nPairwise Evaluation Results:")
    print(f"  Agent Wins: {eval_results['wins']}")
    print(f"  Ground Truth Wins: {eval_results['losses']}")
    print(f"  Ties: {eval_results['ties']}")
    print(f"  Errors: {eval_results['errors']}")
    print(f"  Overall Accuracy: {eval_results['accuracy_percent']}")
    
    eval_file = os.path.join(output_dir, f"pairwise_eval_{timestamp}.json")
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to: {eval_file}")
    
    return all_results, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IterResearch Experiment")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/BC-zn/browsecomp_zh.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of questions to process"
    )
    parser.add_argument(
        "--max_turn",
        type=int,
        default=25,
        help="Maximum number of turns per question"
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default="http://127.0.0.1:10086/v1/chat/completions",
        help="URL of the LLM endpoint"
    )
    parser.add_argument(
        "--research_model",
        type=str,
        default="qwen-flash",
        help="Model name for research"
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Model name for evaluation (defaults to research_model)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_questions=args.num_questions,
        max_turn=args.max_turn,
        llm_url=args.llm_url,
        research_model=args.research_model,
        eval_model=args.eval,
        max_workers=args.max_workers
    )