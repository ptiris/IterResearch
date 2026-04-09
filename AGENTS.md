# IterResearch - Agent Instructions

## Entry Point
```bash
python run.py --input_fp data/questions.jsonl --output_path ./output --max_turn 25 --max_workers 1
```

## Required Configuration (.env)
- `LLM_URL` - main research model endpoint
- `SUMMARY_LLM_URL` - webpage summarization model endpoint  
- `OPENAI_API_KEY` / `OPEN_API_KEY` - API keys for LLM access
- `SEARCH_API_KEY` - SerpAPI key for Google search
- `JINA_API_KEY` - for webpage content extraction

## Search Engine Selection
`--search_engine google|baidu|aliyun` - selects which search backend to use (default: google with SerpAPI)

## Model Remapping Gotcha
When `LLM_URL` contains "deepseek", model names like `qwen-flash`, `qwen-plus` are **automatically remapped** to `deepseek-chat` in `run.py:667`. This is non-obvious behavior.

## LLM Response Format
All LLM responses must use XML-style tags: `<report>...</report>` + `<tool_call>...</tool_call>` OR `<answer>...</answer>`. The `check_report_action()` function validates this at `run.py:345`.

## Token Estimation
Uses `Qwen2.5-7B-Instruct` tokenizer by default (configurable via `TOKENIZER_PATH`). Falls back to char-count ÷ 4 if unavailable (`run.py:235-240`).

## Key Arguments
| Argument | Default | Notes |
|----------|---------|-------|
| `--max_turn` | 25 | Agent reasoning iterations per question |
| `--max_workers` | 1 | Parallel questions |
| `--max_format_retries` | 5 | LLM format validation retries |
| `--max_observation_tokens` | 32000 | Truncation threshold for tool observations |
| `--disable_evaluator` | false | Skip LLM-based answer evaluation |

## Output Files
- `{prefix}_{timestamp}.jsonl` - summary results
- `{prefix}_{timestamp}_full.jsonl` - complete conversation records
- `{prefix}_summary_{timestamp}.json` - metrics (if `--save_metrics` provided)

## No Test Suite
This repo has **no test files**. All verification is done by running `run.py` against sample data in `data/`.

## Tool Backends
- `google_search` / `google_scholar` → SerpAPI (`SEARCH_API_URL`)
- `baidu_search` → Baidu Qianfan (`BAIDU_URL`)
- `aliyun_iqs_search` → Aliyun IQS (`IQS_BASE_URL`)
- `Visit` → Jina Reader + optional ScraperAPI fallback
- `PythonInterpreter` → sandbox at `SANDBOX_ENDPOINTS` (default `http://127.0.0.1:8080`)

## Prompt Templates
Located in `prompts/prompts.py`. Three phases: `initial_instruction_prompt` (turn 0), `instruction_prompt` (turns 1..N-1), `last_instruction_prompt` (final turn before forced answer). Output language mirrors question language.

## Metrics
`metrics.py` tracks LLM tokens, cache efficiency, tool latency, and iteration counts. Cache token extraction handles DeepSeek-style (`prompt_cache_hit_tokens`) and DashScope-style (`prompt_tokens_details.cached_tokens`) formats.

## Quick Test Run
```bash
python quick_run.sh  # Runs example_test.jsonl with 2 workers
```
