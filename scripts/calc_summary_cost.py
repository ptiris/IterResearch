#!/usr/bin/env python3
"""Calculate LLM usage cost from IterResearch *_summary_*.json files.

Outputs RMB by default and includes cache-hit savings details.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pricing import calculate_cost, calculate_tool_cost, get_model_pricing


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_model_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    llm = summary.get("llm", {})
    by_model = llm.get("by_model", {}) or {}

    rows: List[Dict[str, Any]] = []
    if by_model:
        for model_name, stats in by_model.items():
            rows.append(
                {
                    "model": model_name,
                    "prompt_tokens": _to_int(stats.get("prompt_tokens", 0)),
                    "completion_tokens": _to_int(stats.get("completion_tokens", 0)),
                    "cache_read_tokens": _to_int(stats.get("cache_read_tokens", 0)),
                    "cache_write_tokens": _to_int(stats.get("cache_write_tokens", 0)),
                }
            )
        return rows

    # Fallback for old summary structures without by_model.
    primary_model = summary.get("primary_model") or "qwen-flash"
    cache = summary.get("cache", {})
    rows.append(
        {
            "model": primary_model,
            "prompt_tokens": _to_int(llm.get("total_prompt_tokens", 0)),
            "completion_tokens": _to_int(llm.get("total_completion_tokens", 0)),
            "cache_read_tokens": _to_int(cache.get("cache_read_tokens", llm.get("cache_read_tokens", 0))),
            "cache_write_tokens": _to_int(cache.get("cache_write_tokens", llm.get("cache_write_tokens", 0))),
        }
    )
    return rows


def _calc_no_cache_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_write_tokens: int,
) -> float:
    pricing = get_model_pricing(model_name)
    if not pricing:
        return 0.0

    input_rate = pricing["input_cost_per_token"]
    output_rate = pricing["output_cost_per_token"]
    cache_write_rate = pricing.get("cache_creation_input_token_cost", input_rate)

    return (
        prompt_tokens * input_rate
        + completion_tokens * output_rate
        + cache_write_tokens * cache_write_rate
    )


def calculate_summary_cost(summary: Dict[str, Any]) -> Dict[str, Any]:
    model_rows = _extract_model_rows(summary)

    details: List[Dict[str, Any]] = []
    total_actual = 0.0
    total_no_cache = 0.0

    for row in model_rows:
        model_name = row["model"]
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = row["completion_tokens"]
        cache_read_tokens = row["cache_read_tokens"]
        cache_write_tokens = row["cache_write_tokens"]

        actual = calculate_cost(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        actual_total = float(actual.get("total_cost", 0.0))

        no_cache_total = _calc_no_cache_cost(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        savings = max(0.0, no_cache_total - actual_total)

        details.append(
            {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "actual_cost": actual_total,
                "no_cache_cost": no_cache_total,
                "cache_savings": savings,
                "estimated": bool(actual.get("estimated", False)),
                "warning": actual.get("warning", ""),
            }
        )

        total_actual += actual_total
        total_no_cache += no_cache_total

    total_savings = max(0.0, total_no_cache - total_actual)
    savings_pct = (total_savings / total_no_cache * 100.0) if total_no_cache > 0 else 0.0

    tool_rows: List[Dict[str, Any]] = []
    total_tool_cost = 0.0
    total_tool_calls = 0
    total_paid_tool_calls = 0
    priced_tool_count = 0
    free_tool_count = 0
    for tool_name, tool_stats in (summary.get("tools", {}) or {}).items():
        calls = _to_int(tool_stats.get("calls", 0))
        effective_calls = _to_int(tool_stats.get("effective_calls", 0))
        tool_cost = calculate_tool_cost(
            tool_name=tool_name,
            calls=calls,
            effective_calls=effective_calls,
        )
        row = {
            "tool_name": tool_name,
            "calls": calls,
            "effective_calls": effective_calls,
            "billed_calls": _to_int(tool_cost.get("billed_calls", 0)),
            "unit_price_per_1000_calls": float(tool_cost.get("unit_price_per_1000_calls", 0.0)),
            "cost": float(tool_cost.get("total_cost", 0.0)),
            "estimated": bool(tool_cost.get("estimated", False)),
            "warning": tool_cost.get("warning", ""),
            "note": tool_cost.get("note", ""),
        }
        tool_rows.append(row)
        total_tool_cost += row["cost"]
        total_tool_calls += calls
        if row["cost"] > 0:
            priced_tool_count += 1
            total_paid_tool_calls += row["billed_calls"]
        else:
            free_tool_count += 1

    grand_total = total_actual + total_tool_cost

    return {
        "currency": {
            "unit": "CNY",
        },
        "totals": {
            "llm_actual_cost": total_actual,
            "no_cache_cost": total_no_cache,
            "cache_savings": total_savings,
            "cache_discount_percent": savings_pct,
            "tool_cost": total_tool_cost,
            "grand_total_cost": grand_total,
        },
        "tool_summary": {
            "total_tool_calls": total_tool_calls,
            "total_paid_tool_calls": total_paid_tool_calls,
            "priced_tool_count": priced_tool_count,
            "free_tool_count": free_tool_count,
            "tool_cost_share_percent": (total_tool_cost / grand_total * 100.0) if grand_total > 0 else 0.0,
        },
        "llm_details": details,
        "tool_details": tool_rows,
    }


def _print_report(summary_fp: Path, report: Dict[str, Any]) -> None:
    totals = report["totals"]
    tool_summary = report.get("tool_summary", {})

    print(f"Summary file: {summary_fp}")
    print("\n==== Cost (CNY) ====")
    print(f"LLM actual total:  ¥{totals['llm_actual_cost']:.4f}")
    print(f"Tool total:        ¥{totals['tool_cost']:.4f}")
    print(f"Grand total:       ¥{totals['grand_total_cost']:.4f}")
    print(f"No-cache total:    ¥{totals['no_cache_cost']:.4f}")
    print(f"Cache savings:     ¥{totals['cache_savings']:.4f}")
    print(f"Cache discount:    {totals['cache_discount_percent']:.2f}%")

    if tool_summary:
        print("\n==== Tool Summary ====")
        print(f"Total tool calls:       {tool_summary.get('total_tool_calls', 0)}")
        print(f"Paid tool calls:        {tool_summary.get('total_paid_tool_calls', 0)}")
        print(f"Priced tools count:     {tool_summary.get('priced_tool_count', 0)}")
        print(f"Free tools count:       {tool_summary.get('free_tool_count', 0)}")
        print(f"Tool cost share:        {tool_summary.get('tool_cost_share_percent', 0.0):.2f}%")

    print("\n==== LLM By Model ====")
    for row in report["llm_details"]:
        print(f"- {row['model']}")
        print(
            f"  prompt={row['prompt_tokens']}, completion={row['completion_tokens']}, "
            f"cache_read={row['cache_read_tokens']}, cache_write={row['cache_write_tokens']}"
        )
        print(f"  actual=¥{row['actual_cost']:.4f}, no_cache=¥{row['no_cache_cost']:.4f}, savings=¥{row['cache_savings']:.4f}")
        if row.get("warning"):
            print(f"  warning={row['warning']}")

    if report["tool_details"]:
        print("\n==== Tool Cost ====")
        for row in report["tool_details"]:
            print(f"- {row['tool_name']}")
            print(
                f"  calls={row['calls']}, effective_calls={row['effective_calls']}, billed_calls={row['billed_calls']}"
            )
            print(
                f"  unit=¥{row['unit_price_per_1000_calls']:.4f}/1000 calls, cost=¥{row['cost']:.4f}"
            )
            if row.get("note"):
                print(f"  note={row['note']}")
            if row.get("warning"):
                print(f"  warning={row['warning']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate actual model cost from IterResearch summary JSON.")
    parser.add_argument("summary_fp", type=str, help="Path to *_summary_*.json")
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save full cost report JSON.",
    )
    args = parser.parse_args()

    summary_fp = Path(args.summary_fp)
    with summary_fp.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    report = calculate_summary_cost(summary)
    _print_report(summary_fp, report)

    if args.save_json:
        out_fp = Path(args.save_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        with out_fp.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report JSON to: {out_fp}")


if __name__ == "__main__":
    main()
