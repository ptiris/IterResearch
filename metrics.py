"""
Metrics Collection for IterResearch.

Provides comprehensive metrics tracking including:
- LLM call statistics (tokens, latency, by model/iteration)
- Tool execution statistics (calls, latency, cost, success rate)
- Per-iteration latency breakdown
"""
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class StepRecord:
    """Single iteration/step record."""
    turn: int
    query: str
    action: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[str] = None
    tool_result_length: int = 0
    
    # LLM stats
    llm_model: str = ""
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_latency_ms: float = 0.0
    llm_total_latency_ms: float = 0.0
    
    # Tool stats
    tool_latency_ms: float = 0.0
    tool_success: bool = True
    tool_error: Optional[str] = None
    
    # Timing breakdown for this iteration
    iteration_total_latency_ms: float = 0.0


@dataclass
class ToolStats:
    """Statistics for a single tool."""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    total_cost_estimate: float = 0.0
    
    # Per-call cost estimation (can be customized per tool)
    cost_per_call: float = 0.0
    
    # Token stats (for tools that use LLM like Visit)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> str:
        if self.total_calls == 0:
            return "0%"
        return f"{(self.successful_calls / self.total_calls * 100):.2f}%"
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    def to_dict(self) -> Dict:
        return {
            "calls": self.total_calls,
            "success_count": self.successful_calls,
            "failed_count": self.failed_calls,
            "success_rate": self.success_rate,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


class MetricsCollector:
    """
    Global metrics collector for IterResearch.
    
    Tracks:
    - Per-question/step records
    - LLM call statistics by model and iteration
    - Tool execution statistics
    - Latency breakdown per iteration
    """
    
    def __init__(self):
        # Per-question data
        self.current_question: Optional[str] = None
        self.current_question_records: List[StepRecord] = []
        
        # Aggregated stats
        self.all_question_metrics: List[Dict] = []
        
        # Global LLM stats
        self.total_llm_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_llm_latency_ms: float = 0.0
        
        # LLM stats by model
        self.llm_stats_by_model: Dict[str, Dict] = defaultdict(lambda: {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_ms": 0.0
        })
        
        # LLM stats by iteration count (how many iterations each question took)
        self.iteration_counts: List[int] = []
        
        # Tool stats registry
        self.tool_stats: Dict[str, ToolStats] = {}
        
        # Global timing
        self.global_start_time: Optional[float] = None
        self.global_end_time: Optional[float] = None
    
    def _get_or_create_tool_stats(self, tool_name: str) -> ToolStats:
        """Get or create tool stats for a tool."""
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolStats(name=tool_name)
        return self.tool_stats[tool_name]
    
    def start_question(self, question: str):
        """Start tracking a new question."""
        self.current_question = question
        self.current_question_records = []
    
    def start_global_timer(self):
        """Start global timer for entire run."""
        self.global_start_time = time.time()
    
    def end_global_timer(self):
        """End global timer for entire run."""
        self.global_end_time = time.time()
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        turn: int
    ):
        """Record an LLM call."""
        self.total_llm_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_llm_latency_ms += latency_ms
        
        # Update per-model stats
        model_stats = self.llm_stats_by_model[model]
        model_stats["calls"] += 1
        model_stats["prompt_tokens"] += prompt_tokens
        model_stats["completion_tokens"] += completion_tokens
        model_stats["total_latency_ms"] += latency_ms
        
        # Update current record if exists
        if self.current_question_records and self.current_question_records[-1].turn == turn:
            record = self.current_question_records[-1]
            record.llm_model = model
            record.llm_prompt_tokens = prompt_tokens
            record.llm_completion_tokens = completion_tokens
            record.llm_latency_ms = latency_ms
            record.llm_total_latency_ms = latency_ms
    
    def record_tool_call(
        self,
        tool_name: str,
        args: Dict,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        cost_estimate: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Record a tool call."""
        stats = self._get_or_create_tool_stats(tool_name)
        stats.total_calls += 1
        stats.total_latency_ms += latency_ms
        stats.total_cost_estimate += cost_estimate
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        
        if success:
            stats.successful_calls += 1
        else:
            stats.failed_calls += 1
            if error:
                stats.error_types[error] += 1
        
        # Update current record if exists
        if self.current_question_records:
            record = self.current_question_records[-1]
            record.tool_latency_ms = latency_ms
            record.tool_success = success
            record.tool_error = error
    
    def start_iteration(self, turn: int, query: str, action: str):
        """Start a new iteration/step."""
        record = StepRecord(
            turn=turn,
            query=query,
            action=action
        )
        self.current_question_records.append(record)
    
    def update_iteration_action(self, action: str, tool_name: str = None, tool_args: Dict = None):
        """Update the action details for current iteration."""
        if self.current_question_records:
            record = self.current_question_records[-1]
            record.action = action
            record.tool_name = tool_name
            record.tool_args = tool_args
    
    def update_iteration_tool_result(self, result: str, result_length: int):
        """Update tool result for current iteration."""
        if self.current_question_records:
            record = self.current_question_records[-1]
            record.tool_result = result[:1000] if result else ""
            record.tool_result_length = result_length
    
    def end_iteration(self):
        """End current iteration and calculate total latency."""
        if self.current_question_records:
            record = self.current_question_records[-1]
            record.iteration_total_latency_ms = (
                record.llm_total_latency_ms + 
                record.tool_latency_ms
            )
    
    def end_question(self, final_answer_found: bool = True) -> Dict:
        """End tracking for current question and return metrics."""
        if not self.current_question_records:
            return {}
        
        turns = len(self.current_question_records)
        self.iteration_counts.append(turns)
        
        # Calculate question-level stats
        total_llm_calls = sum(1 for r in self.current_question_records if r.llm_prompt_tokens > 0)
        total_prompt_tokens = sum(r.llm_prompt_tokens for r in self.current_question_records)
        total_completion_tokens = sum(r.llm_completion_tokens for r in self.current_question_records)
        total_llm_latency = sum(r.llm_total_latency_ms for r in self.current_question_records)
        total_tool_latency = sum(r.tool_latency_ms for r in self.current_question_records)
        
        # Tool usage for this question
        tool_usage = {}
        for record in self.current_question_records:
            if record.tool_name:
                if record.tool_name not in tool_usage:
                    tool_usage[record.tool_name] = {
                        "calls": 0,
                        "total_latency_ms": 0.0,
                        "success_count": 0,
                        "failed_count": 0
                    }
                tool_usage[record.tool_name]["calls"] += 1
                tool_usage[record.tool_name]["total_latency_ms"] += record.tool_latency_ms
                if record.tool_success:
                    tool_usage[record.tool_name]["success_count"] += 1
                else:
                    tool_usage[record.tool_name]["failed_count"] += 1
        
        # Per-iteration latency
        iteration_latencies = [
            {
                "turn": r.turn,
                "llm_latency_ms": round(r.llm_total_latency_ms, 2),
                "tool_latency_ms": round(r.tool_latency_ms, 2),
                "total_latency_ms": round(r.iteration_total_latency_ms, 2),
                "action": r.action,
                "tool_name": r.tool_name,
                "tool_success": r.tool_success
            }
            for r in self.current_question_records
        ]
        
        question_metrics = {
            "total_turns": turns,
            "final_answer_found": final_answer_found,
            "llm": {
                "total_calls": total_llm_calls,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "total_llm_latency_ms": round(total_llm_latency, 2),
                "avg_llm_latency_ms": round(total_llm_latency / total_llm_calls, 2) if total_llm_calls > 0 else 0,
                "by_model": {}
            },
            "tool_usage": tool_usage,
            "total_tool_latency_ms": round(total_tool_latency, 2),
            "iteration_latencies": iteration_latencies,
            "latency_breakdown": {
                "llm_time_ms": round(total_llm_latency, 2),
                "tool_time_ms": round(total_tool_latency, 2),
                "total_time_ms": round(total_llm_latency + total_tool_latency, 2)
            }
        }
        
        # Add per-model LLM stats
        models_used = set(r.llm_model for r in self.current_question_records if r.llm_model)
        for model in models_used:
            model_records = [r for r in self.current_question_records if r.llm_model == model]
            question_metrics["llm"]["by_model"][model] = {
                "calls": len(model_records),
                "prompt_tokens": sum(r.llm_prompt_tokens for r in model_records),
                "completion_tokens": sum(r.llm_completion_tokens for r in model_records),
                "total_latency_ms": round(sum(r.llm_total_latency_ms for r in model_records), 2)
            }
        
        # Add tool response count
        question_metrics["tool_response"] = {
            "total_tool_calls": sum(1 for r in self.current_question_records if r.tool_name),
            "total_tool_result_chars": sum(r.tool_result_length for r in self.current_question_records)
        }
        
        self.all_question_metrics.append(question_metrics)
        
        # Update global tool stats
        for tool_name, usage in tool_usage.items():
            stats = self._get_or_create_tool_stats(tool_name)
            stats.total_calls = usage["calls"]
            stats.successful_calls = usage["success_count"]
            stats.failed_calls = usage["failed_count"]
            stats.total_latency_ms = usage["total_latency_ms"]
        
        # Reset current question
        self.current_question = None
        self.current_question_records = []
        
        return question_metrics
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        if not self.all_question_metrics:
            return {}
        
        total_questions = len(self.all_question_metrics)
        total_turns = sum(m["total_turns"] for m in self.all_question_metrics)
        questions_with_answer = sum(1 for m in self.all_question_metrics if m.get("final_answer_found", False))
        
        # Aggregate LLM stats
        total_llm_calls_all = sum(m["llm"]["total_calls"] for m in self.all_question_metrics)
        total_prompt_all = sum(m["llm"]["total_prompt_tokens"] for m in self.all_question_metrics)
        total_completion_all = sum(m["llm"]["total_completion_tokens"] for m in self.all_question_metrics)
        
        # Aggregate tool stats
        aggregated_tool_stats = {}
        for tool_name in self.tool_stats:
            stats = self.tool_stats[tool_name]
            aggregated_tool_stats[tool_name] = stats.to_dict()
        
        return {
            "total_questions": total_questions,
            "total_turns": total_turns,
            "avg_turns_per_question": round(total_turns / total_questions, 2) if total_questions > 0 else 0,
            "answer_success_rate": f"{(questions_with_answer / total_questions * 100):.2f}%" if total_questions > 0 else "0%",
            "llm": {
                "total_calls": total_llm_calls_all,
                "total_prompt_tokens": total_prompt_all,
                "total_completion_tokens": total_completion_all,
                "total_tokens": total_prompt_all + total_completion_all,
                "total_latency_ms": round(self.total_llm_latency_ms, 2),
                "by_model": dict(self.llm_stats_by_model)
            },
            "iteration_distribution": {
                "min": min(self.iteration_counts) if self.iteration_counts else 0,
                "max": max(self.iteration_counts) if self.iteration_counts else 0,
                "avg": round(sum(self.iteration_counts) / len(self.iteration_counts), 2) if self.iteration_counts else 0
            },
            "tools": aggregated_tool_stats,
            "global_time_ms": round((self.global_end_time - self.global_start_time) * 1000, 2) if self.global_end_time and self.global_start_time else 0
        }


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector

def reset_metrics_collector():
    """Reset global metrics collector."""
    global _global_collector
    _global_collector = None