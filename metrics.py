"""
Metrics Collection for IterResearch.

Provides comprehensive metrics tracking including:
- LLM call statistics (tokens, latency, by model/iteration)
- Tool execution statistics (calls, latency, cost, success rate)
- Per-iteration latency breakdown
"""
import time
import copy
import threading
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
    llm_success: bool = True
    
    # Cache stats
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    
    @property
    def fresh_input_tokens(self) -> int:
        """Fresh input tokens (total - cache_read)."""
        return max(0, self.llm_prompt_tokens - self.cache_read_tokens)
    
    # Token breakdown for input (using tokenizer)
    system_tokens: int = 0
    user_tokens: int = 0
    tools_definition_tokens: int = 0
    report_tokens: int = 0
    observation_tokens: int = 0
    
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
    effective_calls: int = 0
    
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
            "effective_calls": self.effective_calls,
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
    - Cache efficiency statistics
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        
        # Per-question data
        self.current_question = None
        self.current_question_records = []
        self.current_question_tool_calls = []
        self.current_question_llm_failures = []
        
        # Aggregated stats
        self.all_question_metrics: List[Dict] = []
        
        # Global LLM stats
        self.total_llm_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_llm_latency_ms: float = 0.0
        
        # Global Cache stats
        self.total_cache_read_tokens: int = 0
        self.total_cache_write_tokens: int = 0
        self.total_fresh_input_tokens: int = 0
        
        # Token breakdown stats (using tokenizer)
        self.total_system_tokens: int = 0
        self.total_user_tokens: int = 0
        self.total_tools_definition_tokens: int = 0
        self.total_report_tokens: int = 0
        self.total_observation_tokens: int = 0
        
        # Tool definitions cost tracking
        self.tool_definitions_tokens: Dict[str, int] = {}
        
        # LLM stats by iteration count (how many iterations each question took)
        self.iteration_counts: List[int] = []
        
        # Tool stats registry
        self.tool_stats: Dict[str, ToolStats] = {}
        
        # Global timing
        self.global_start_time: Optional[float] = None
        self.global_end_time: Optional[float] = None
        
        # Model used (for pricing lookup)
        self.primary_model: str = ""

    @property
    def current_question(self) -> Optional[str]:
        return getattr(self._thread_local, "current_question", None)

    @current_question.setter
    def current_question(self, value: Optional[str]):
        self._thread_local.current_question = value

    @property
    def current_question_records(self) -> List[StepRecord]:
        records = getattr(self._thread_local, "current_question_records", None)
        if records is None:
            records = []
            self._thread_local.current_question_records = records
        return records

    @current_question_records.setter
    def current_question_records(self, value: List[StepRecord]):
        self._thread_local.current_question_records = value

    @property
    def current_question_tool_calls(self) -> List[Dict[str, Any]]:
        tool_calls = getattr(self._thread_local, "current_question_tool_calls", None)
        if tool_calls is None:
            tool_calls = []
            self._thread_local.current_question_tool_calls = tool_calls
        return tool_calls

    @current_question_tool_calls.setter
    def current_question_tool_calls(self, value: List[Dict[str, Any]]):
        self._thread_local.current_question_tool_calls = value

    @property
    def current_question_llm_failures(self) -> List[Dict[str, Any]]:
        llm_failures = getattr(self._thread_local, "current_question_llm_failures", None)
        if llm_failures is None:
            llm_failures = []
            self._thread_local.current_question_llm_failures = llm_failures
        return llm_failures

    @current_question_llm_failures.setter
    def current_question_llm_failures(self, value: List[Dict[str, Any]]):
        self._thread_local.current_question_llm_failures = value
    
    def _get_or_create_tool_stats(self, tool_name: str) -> ToolStats:
        """Get or create tool stats for a tool."""
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolStats(name=tool_name)
        return self.tool_stats[tool_name]
    
    def start_question(self, question: str):
        """Start tracking a new question."""
        # Thread-local context does not need global locking.
        self.current_question = question
        self.current_question_records = []
        self.current_question_tool_calls = []
        self.current_question_llm_failures = []

    def record_llm_failure(self, turn: int, model: str, error_type: str, error_msg: str):
        """Record one LLM failure event (including retries)."""
        self.current_question_llm_failures.append({
            "turn": turn,
            "model": model,
            "error_type": error_type,
            "error_msg": error_msg,
            "timestamp": time.time()
        })
    
    def start_global_timer(self):
        """Start global timer for entire run."""
        with self._lock:
            self.global_start_time = time.time()
    
    def end_global_timer(self):
        """End global timer for entire run."""
        with self._lock:
            self.global_end_time = time.time()
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        turn: int,
        success: bool = True,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0
    ):
        """Record an LLM call."""
        current_record = None
        if self.current_question_records and self.current_question_records[-1].turn == turn:
            current_record = self.current_question_records[-1]

        with self._lock:
            self.total_llm_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_llm_latency_ms += latency_ms
            self.total_cache_read_tokens += cache_read_tokens
            self.total_cache_write_tokens += cache_write_tokens
            fresh_tokens = max(0, prompt_tokens - cache_read_tokens)
            self.total_fresh_input_tokens += fresh_tokens
            
            # Track primary model (first non-empty model)
            if model and not self.primary_model:
                self.primary_model = model

        # Update thread-local step record without holding the global lock.
        if current_record is not None:
            current_record.llm_model = model
            current_record.llm_prompt_tokens = prompt_tokens
            current_record.llm_completion_tokens = completion_tokens
            current_record.llm_latency_ms = latency_ms
            current_record.llm_total_latency_ms = latency_ms
            current_record.llm_success = success
            current_record.cache_read_tokens = cache_read_tokens
            current_record.cache_write_tokens = cache_write_tokens
    
    def record_prompt_breakdown(
        self,
        system_tokens: int = 0,
        user_tokens: int = 0,
        tools_definition_tokens: int = 0,
        report_tokens: int = 0,
        observation_tokens: int = 0,
        turn: int = -1
    ):
        """Record token breakdown for prompt categories."""
        current_record = None
        if self.current_question_records and self.current_question_records[-1].turn == turn:
            current_record = self.current_question_records[-1]

        with self._lock:
            self.total_system_tokens += system_tokens
            self.total_user_tokens += user_tokens
            self.total_tools_definition_tokens += tools_definition_tokens
            self.total_report_tokens += report_tokens
            self.total_observation_tokens += observation_tokens

        if current_record is not None:
            current_record.system_tokens = system_tokens
            current_record.user_tokens = user_tokens
            current_record.tools_definition_tokens = tools_definition_tokens
            current_record.report_tokens = report_tokens
            current_record.observation_tokens = observation_tokens
    
    def record_tool_definition(self, name: str, definition_tokens: int):
        """Record a tool's definition size (in tokens)."""
        with self._lock:
            if name not in self.tool_definitions_tokens:
                self.tool_definitions_tokens[name] = definition_tokens
    
    def record_tool_call(
        self,
        tool_name: str,
        args: Dict,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        cost_estimate: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        effective_calls: int = 1
    ):
        """Record a tool call."""
        current_record = self.current_question_records[-1] if self.current_question_records else None

        with self._lock:
            stats = self._get_or_create_tool_stats(tool_name)
            stats.total_calls += 1
            stats.effective_calls += max(0, effective_calls)
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

        if current_record is not None:
            current_record.tool_latency_ms = latency_ms
            current_record.tool_success = success
            current_record.tool_error = error

        self.current_question_tool_calls.append({
            "turn": current_record.turn if current_record is not None else -1,
            "tool_name": tool_name,
            "args": args,
            "result": current_record.tool_result if current_record is not None else None,
            "result_length": current_record.tool_result_length if current_record is not None else 0,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "error": error,
            "effective_calls": effective_calls
        })
    
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
        if self.current_question_tool_calls:
            call_record = self.current_question_tool_calls[-1]
            call_record["result"] = result
            call_record["result_length"] = result_length
    
    def end_iteration(self):
        """End current iteration and calculate total latency."""
        if self.current_question_records:
            record = self.current_question_records[-1]
            record.iteration_total_latency_ms = (
                record.llm_total_latency_ms +
                record.tool_latency_ms
            )
    
    def end_question(
        self,
        final_answer_found: bool = True,
        evaluation: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """End tracking for current question and return metrics."""
        local_records = list(self.current_question_records)
        local_tool_calls = list(self.current_question_tool_calls)
        local_llm_failures = list(self.current_question_llm_failures)
        if not local_records:
            return {}

        turns = len(local_records)
            
            # Calculate question-level stats
        total_llm_calls = sum(1 for r in local_records if r.llm_model)
        successful_llm_calls = sum(1 for r in local_records if r.llm_model and r.llm_success)
        failed_llm_calls = sum(1 for r in local_records if r.llm_model and not r.llm_success)
        total_prompt_tokens = sum(r.llm_prompt_tokens for r in local_records)
        total_completion_tokens = sum(r.llm_completion_tokens for r in local_records)
        total_llm_latency = sum(r.llm_total_latency_ms for r in local_records)
        total_tool_latency = sum(r.tool_latency_ms for r in local_records)
            
            # Tool usage for this question
        tool_usage = {}
        for record in local_records:
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
                "llm_success": r.llm_success,
                "tool_latency_ms": round(r.tool_latency_ms, 2),
                "total_latency_ms": round(r.iteration_total_latency_ms, 2),
                "action": r.action,
                "tool_name": r.tool_name,
                "tool_success": r.tool_success
            }
            for r in local_records
        ]

        question_metrics = {
            "total_turns": turns,
            "final_answer_found": final_answer_found,
            "evaluation": evaluation or {
                "evaluated": False,
                "is_correct": None,
                "score": None,
                "reason": "not_evaluated"
            },
            "llm": {
                "total_calls": total_llm_calls,
                "successful_calls": successful_llm_calls,
                "failed_calls": failed_llm_calls,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "total_llm_latency_ms": round(total_llm_latency, 2),
                "avg_llm_latency_ms": round(total_llm_latency / total_llm_calls, 2) if total_llm_calls > 0 else 0,
                "by_model": {}
            },
            "tool_usage": tool_usage,
            "tool_call_records": local_tool_calls,
            "llm_failure_records": local_llm_failures,
            "total_tool_latency_ms": round(total_tool_latency, 2),
            "iteration_latencies": iteration_latencies,
            "latency_breakdown": {
                "llm_time_ms": round(total_llm_latency, 2),
                "tool_time_ms": round(total_tool_latency, 2),
                "total_time_ms": round(total_llm_latency + total_tool_latency, 2)
            }
        }
            
            # Add per-model LLM stats
        models_used = set(r.llm_model for r in local_records if r.llm_model)
        for model in models_used:
            model_records = [r for r in local_records if r.llm_model == model]
            question_metrics["llm"]["by_model"][model] = {
                "calls": len(model_records),
                "prompt_tokens": sum(r.llm_prompt_tokens for r in model_records),
                "completion_tokens": sum(r.llm_completion_tokens for r in model_records),
                "total_latency_ms": round(sum(r.llm_total_latency_ms for r in model_records), 2),
                "cache_read_tokens": sum(r.cache_read_tokens for r in model_records),
                "cache_write_tokens": sum(r.cache_write_tokens for r in model_records)
            }
            
            # Add cache stats to question metrics
        question_metrics["cache"] = {
            "cache_read_tokens": sum(r.cache_read_tokens for r in local_records),
            "cache_write_tokens": sum(r.cache_write_tokens for r in local_records),
            "fresh_input_tokens": sum(r.fresh_input_tokens for r in local_records)
        }
            
            # Add tool response count
        question_metrics["tool_response"] = {
            "total_tool_calls": sum(1 for r in local_records if r.tool_name),
            "total_tool_result_chars": sum(r.tool_result_length for r in local_records)
        }

        with self._lock:
            self.iteration_counts.append(turns)
            self.all_question_metrics.append(question_metrics)

        # Reset current thread-local question state.
        self.current_question = None
        self.current_question_records = []
        self.current_question_tool_calls = []
        self.current_question_llm_failures = []

        return question_metrics
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        with self._lock:
            if not self.all_question_metrics:
                return {}
            
            total_questions = len(self.all_question_metrics)
            total_turns = sum(m["total_turns"] for m in self.all_question_metrics)
            questions_with_answer = sum(1 for m in self.all_question_metrics if m.get("final_answer_found", False))

            evaluated_questions = [
                m for m in self.all_question_metrics
                if isinstance(m.get("evaluation"), dict) and m["evaluation"].get("evaluated", False)
            ]
            evaluated_count = len(evaluated_questions)
            correct_count = sum(1 for m in evaluated_questions if m["evaluation"].get("is_correct") is True)
            incorrect_count = sum(1 for m in evaluated_questions if m["evaluation"].get("is_correct") is False)
            avg_score = 0.0
            if evaluated_count > 0:
                score_sum = 0.0
                score_count = 0
                for m in evaluated_questions:
                    score = m["evaluation"].get("score")
                    if isinstance(score, (int, float)):
                        score_sum += float(score)
                        score_count += 1
                avg_score = (score_sum / score_count) if score_count > 0 else 0.0
            
            # Aggregate LLM stats
            total_llm_calls_all = sum(m["llm"]["total_calls"] for m in self.all_question_metrics)
            successful_llm_calls_all = sum(m["llm"].get("successful_calls", 0) for m in self.all_question_metrics)
            failed_llm_calls_all = sum(m["llm"].get("failed_calls", 0) for m in self.all_question_metrics)
            total_prompt_all = sum(m["llm"]["total_prompt_tokens"] for m in self.all_question_metrics)
            total_completion_all = sum(m["llm"]["total_completion_tokens"] for m in self.all_question_metrics)
            
            # Aggregate LLM by_model from all questions
            llm_by_model = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_latency_ms": 0.0, "cache_read_tokens": 0, "cache_write_tokens": 0})
            for m in self.all_question_metrics:
                for model, model_stats in m["llm"].get("by_model", {}).items():
                    llm_by_model[model]["calls"] += model_stats["calls"]
                    llm_by_model[model]["prompt_tokens"] += model_stats["prompt_tokens"]
                    llm_by_model[model]["completion_tokens"] += model_stats["completion_tokens"]
                    llm_by_model[model]["total_latency_ms"] += model_stats.get("total_latency_ms", 0)
                    llm_by_model[model]["cache_read_tokens"] += model_stats.get("cache_read_tokens", 0)
                    llm_by_model[model]["cache_write_tokens"] += model_stats.get("cache_write_tokens", 0)
            
            # Aggregate total latency from all questions
            total_latency_all = sum(m["llm"].get("total_llm_latency_ms", 0) for m in self.all_question_metrics)
            
            # Aggregate tool stats
            aggregated_tool_stats = {}
            for tool_name in self.tool_stats:
                stats = self.tool_stats[tool_name]
                aggregated_tool_stats[tool_name] = stats.to_dict()
            
            # Calculate cache efficiency
            cache_read = self.total_cache_read_tokens
            cache_write = self.total_cache_write_tokens
            total_input = total_prompt_all
            fresh_input = max(0, total_input - cache_read)
            
            cache_hit_rate = 0.0
            if total_input > 0:
                cache_hit_rate = (cache_read / total_input) * 100
            
            return {
                "total_questions": total_questions,
                "total_turns": total_turns,
                "avg_turns_per_question": round(total_turns / total_questions, 2) if total_questions > 0 else 0,
                "answer_success_rate": f"{(questions_with_answer / total_questions * 100):.2f}%" if total_questions > 0 else "0%",
                "evaluation": {
                    "evaluated_questions": evaluated_count,
                    "not_evaluated_questions": max(0, total_questions - evaluated_count),
                    "correct_count": correct_count,
                    "incorrect_count": incorrect_count,
                    "accuracy": round((correct_count / evaluated_count * 100), 2) if evaluated_count > 0 else 0.0,
                    "accuracy_str": f"{(correct_count / evaluated_count * 100):.2f}%" if evaluated_count > 0 else "N/A",
                    "avg_score": round(avg_score, 4)
                },
                "llm": {
                    "total_calls": total_llm_calls_all,
                    "successful_calls": successful_llm_calls_all,
                    "failed_calls": failed_llm_calls_all,
                    "total_prompt_tokens": total_prompt_all,
                    "total_completion_tokens": total_completion_all,
                    "total_tokens": total_prompt_all + total_completion_all,
                    "total_latency_ms": round(total_latency_all, 2),
                    "cache_read_tokens": cache_read,
                    "cache_write_tokens": cache_write,
                    "fresh_input_tokens": fresh_input,
                    "by_model": dict(llm_by_model)
                },
                "cache": {
                    "cache_read_tokens": cache_read,
                    "cache_write_tokens": cache_write,
                    "fresh_input_tokens": fresh_input,
                    "total_input_tokens": total_input,
                    "cache_hit_rate": round(cache_hit_rate, 2),
                    "cache_hit_rate_str": f"{cache_hit_rate:.1f}%"
                },
                "prompt_breakdown": {
                    "system": self.total_system_tokens,
                    "user": self.total_user_tokens,
                    "tools_definition": self.total_tools_definition_tokens,
                    "report": self.total_report_tokens,
                    "observation": self.total_observation_tokens,
                    "total": (self.total_system_tokens + self.total_user_tokens + 
                             self.total_tools_definition_tokens + self.total_report_tokens + 
                             self.total_observation_tokens)
                },
                "tool_definitions_cost": {
                    tool: tokens for tool, tokens in self.tool_definitions_tokens.items()
                },
                "iteration_distribution": {
                    "min": min(self.iteration_counts) if self.iteration_counts else 0,
                    "max": max(self.iteration_counts) if self.iteration_counts else 0,
                    "avg": round(sum(self.iteration_counts) / len(self.iteration_counts), 2) if self.iteration_counts else 0
                },
                "tools": aggregated_tool_stats,
                "global_time_ms": round((self.global_end_time - self.global_start_time) * 1000, 2) if self.global_end_time and self.global_start_time else 0,
                "primary_model": self.primary_model
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