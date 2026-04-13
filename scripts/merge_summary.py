import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge_summaries(input_files: list) -> dict:
    total_questions = 0
    total_turns = 0
    questions_with_answer = 0

    total_llm_calls = 0
    successful_llm_calls = 0
    failed_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_latency_ms = 0.0

    total_cache_read_tokens = 0
    total_cache_write_tokens = 0

    total_system_tokens = 0
    total_user_tokens = 0
    total_tools_definition_tokens = 0
    total_report_tokens = 0
    total_observation_tokens = 0

    total_global_time_ms = 0.0

    eval_evaluated = 0
    eval_not_evaluated = 0
    eval_correct = 0
    eval_incorrect = 0
    eval_score_sum = 0.0
    eval_score_count = 0

    iter_dist_min = float("inf")
    iter_dist_max = 0
    iter_dist_weighted_avg_sum = 0.0

    llm_by_model = defaultdict(
        lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_ms": 0.0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
    )

    tools_stats = defaultdict(
        lambda: {
            "calls": 0,
            "effective_calls": 0,
            "success_count": 0,
            "failed_count": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "total_cost_estimate": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
    )

    tool_definitions_cost = defaultdict(int)

    primary_model_counts = defaultdict(int)

    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_questions += data.get("total_questions", 0)
        total_turns += data.get("total_turns", 0)

        answer_rate_str = data.get("answer_success_rate", "0%")
        if answer_rate_str != "N/A" and answer_rate_str != "0%":
            try:
                rate = float(answer_rate_str.rstrip("%"))
                questions_with_answer += int(data["total_questions"] * rate / 100)
            except (ValueError, KeyError):
                pass

        llm = data.get("llm", {})
        total_llm_calls += llm.get("total_calls", 0)
        successful_llm_calls += llm.get("successful_calls", 0)
        failed_llm_calls += llm.get("failed_calls", 0)
        total_prompt_tokens += llm.get("total_prompt_tokens", 0)
        total_completion_tokens += llm.get("total_completion_tokens", 0)
        total_llm_latency_ms += llm.get("total_latency_ms", 0.0)

        for model, model_stats in llm.get("by_model", {}).items():
            llm_by_model[model]["calls"] += model_stats.get("calls", 0)
            llm_by_model[model]["prompt_tokens"] += model_stats.get("prompt_tokens", 0)
            llm_by_model[model]["completion_tokens"] += model_stats.get(
                "completion_tokens", 0
            )
            llm_by_model[model]["total_latency_ms"] += model_stats.get(
                "total_latency_ms", 0.0
            )
            llm_by_model[model]["cache_read_tokens"] += model_stats.get(
                "cache_read_tokens", 0
            )
            llm_by_model[model]["cache_write_tokens"] += model_stats.get(
                "cache_write_tokens", 0
            )

        cache = data.get("cache", {})
        total_cache_read_tokens += cache.get("cache_read_tokens", 0)
        total_cache_write_tokens += cache.get("cache_write_tokens", 0)

        prompt_breakdown = data.get("prompt_breakdown", {})
        total_system_tokens += prompt_breakdown.get("system", 0)
        total_user_tokens += prompt_breakdown.get("user", 0)
        total_tools_definition_tokens += prompt_breakdown.get("tools_definition", 0)
        total_report_tokens += prompt_breakdown.get("report", 0)
        total_observation_tokens += prompt_breakdown.get("observation", 0)

        eval_stats = data.get("evaluation", {})
        eval_evaluated += eval_stats.get("evaluated_questions", 0)
        eval_not_evaluated += eval_stats.get("not_evaluated_questions", 0)
        eval_correct += eval_stats.get("correct_count", 0)
        eval_incorrect += eval_stats.get("incorrect_count", 0)

        score = eval_stats.get("avg_score", 0)
        if score and score > 0:
            eval_score_sum += score * eval_stats.get("evaluated_questions", 0)
            eval_score_count += eval_stats.get("evaluated_questions", 0)

        iter_dist = data.get("iteration_distribution", {})
        iter_dist_min = min(iter_dist_min, iter_dist.get("min", float("inf")))
        iter_dist_max = max(iter_dist_max, iter_dist.get("max", 0))

        avg = iter_dist.get("avg", 0)
        n = data.get("total_questions", 0)
        if avg > 0 and n > 0:
            iter_dist_weighted_avg_sum += avg * n

        for tool_name, tool_stats in data.get("tools", {}).items():
            tools_stats[tool_name]["calls"] += tool_stats.get("calls", 0)
            tools_stats[tool_name]["effective_calls"] += tool_stats.get(
                "effective_calls", 0
            )
            tools_stats[tool_name]["success_count"] += tool_stats.get(
                "success_count", 0
            )
            tools_stats[tool_name]["failed_count"] += tool_stats.get("failed_count", 0)
            tools_stats[tool_name]["total_latency_ms"] += tool_stats.get(
                "total_latency_ms", 0.0
            )
            tools_stats[tool_name]["total_cost_estimate"] += tool_stats.get(
                "total_cost_estimate", 0.0
            )
            tools_stats[tool_name]["total_input_tokens"] += tool_stats.get(
                "total_input_tokens", 0
            )
            tools_stats[tool_name]["total_output_tokens"] += tool_stats.get(
                "total_output_tokens", 0
            )

        for tool, tokens in data.get("tool_definitions_cost", {}).items():
            tool_definitions_cost[tool] += tokens

        primary_model = data.get("primary_model", "")
        if primary_model:
            primary_model_counts[primary_model] += 1

        total_global_time_ms += data.get("global_time_ms", 0.0)

    primary_model = (
        max(primary_model_counts.keys(), key=lambda k: primary_model_counts[k])
        if primary_model_counts
        else ""
    )

    fresh_input_tokens = max(0, total_prompt_tokens - total_cache_read_tokens)
    cache_hit_rate = (
        (total_cache_read_tokens / total_prompt_tokens * 100)
        if total_prompt_tokens > 0
        else 0.0
    )
    eval_accuracy = (eval_correct / eval_evaluated * 100) if eval_evaluated > 0 else 0.0
    eval_avg_score = (
        (eval_score_sum / eval_score_count) if eval_score_count > 0 else 0.0
    )
    iter_dist_avg = (
        (iter_dist_weighted_avg_sum / total_questions) if total_questions > 0 else 0.0
    )

    avg_turns_per_question = (
        round(total_turns / total_questions, 2) if total_questions > 0 else 0
    )
    answer_success_rate = (
        f"{(questions_with_answer / total_questions * 100):.2f}%"
        if total_questions > 0
        else "0%"
    )

    prompt_breakdown_total = (
        total_system_tokens
        + total_user_tokens
        + total_tools_definition_tokens
        + total_report_tokens
        + total_observation_tokens
    )

    for tool_name in tools_stats:
        stats = tools_stats[tool_name]
        stats["avg_latency_ms"] = (
            round(stats["total_latency_ms"] / stats["calls"], 2)
            if stats["calls"] > 0
            else 0.0
        )
        stats["success_rate"] = (
            f"{(stats['success_count'] / stats['calls'] * 100):.2f}%"
            if stats["calls"] > 0
            else "0%"
        )

    for model in llm_by_model:
        llm_by_model[model]["total_latency_ms"] = round(
            llm_by_model[model]["total_latency_ms"], 2
        )

    merged = {
        "total_questions": total_questions,
        "total_turns": total_turns,
        "avg_turns_per_question": avg_turns_per_question,
        "answer_success_rate": answer_success_rate,
        "evaluation": {
            "evaluated_questions": eval_evaluated,
            "not_evaluated_questions": eval_not_evaluated,
            "correct_count": eval_correct,
            "incorrect_count": eval_incorrect,
            "accuracy": round(eval_accuracy, 2),
            "accuracy_str": f"{eval_accuracy:.2f}%",
            "avg_score": round(eval_avg_score, 4),
        },
        "llm": {
            "total_calls": total_llm_calls,
            "successful_calls": successful_llm_calls,
            "failed_calls": failed_llm_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_latency_ms": round(total_llm_latency_ms, 2),
            "cache_read_tokens": total_cache_read_tokens,
            "cache_write_tokens": total_cache_write_tokens,
            "fresh_input_tokens": fresh_input_tokens,
            "by_model": dict(llm_by_model),
        },
        "cache": {
            "cache_read_tokens": total_cache_read_tokens,
            "cache_write_tokens": total_cache_write_tokens,
            "fresh_input_tokens": fresh_input_tokens,
            "total_input_tokens": total_prompt_tokens,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "cache_hit_rate_str": f"{cache_hit_rate:.1f}%",
        },
        "prompt_breakdown": {
            "system": total_system_tokens,
            "user": total_user_tokens,
            "tools_definition": total_tools_definition_tokens,
            "report": total_report_tokens,
            "observation": total_observation_tokens,
            "total": prompt_breakdown_total,
        },
        "tool_definitions_cost": dict(tool_definitions_cost),
        "iteration_distribution": {
            "min": iter_dist_min if iter_dist_min != float("inf") else 0,
            "max": iter_dist_max,
            "avg": round(iter_dist_avg, 2),
        },
        "tools": dict(tools_stats),
        "global_time_ms": round(total_global_time_ms, 2),
        "primary_model": primary_model,
    }

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple summary JSON files into one"
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True, help="Input summary JSON files"
    )
    parser.add_argument("-o", "--output", required=True, help="Output merged JSON file")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("Error: At least 2 input files are required")
        return 1

    merged = merge_summaries(args.inputs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(args.inputs)} files into '{args.output}'")
    print(f"  Total questions: {merged['total_questions']}")
    print(f"  Total turns: {merged['total_turns']}")
    print(f"  Total LLM calls: {merged['llm']['total_calls']}")
    print(f"  Primary model: {merged['primary_model']}")

    return 0


if __name__ == "__main__":
    exit(main())
