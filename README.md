<div align="center">

# IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling

<a href='https://arxiv.org/abs/2511.07327'><img src='https://img.shields.io/badge/Paper-Arxiv-red'> </a>
<a href='https://iclr.cc/'><img src='https://img.shields.io/badge/ICLR-2026-blue'></a>

</div>

* **IterResearch** is an iterative research agent that performs deep research tasks through multi-turn tool interactions, **supporting 2048+ tool calls with only 40k context**. 🕵️‍♂️
* IterResearch has been accepted at ICLR 2026 🎉.

## Important Notice

**This is a reproduction version of IterResearch** designed to help the community understand how IterResearch actually works through code. Special thanks to community contributors for this implementation. Due to certain restrictions, we are unable to release the original version. Besides, this implementation can be used for DeepSeek V3.1/3.2.

Some tools in this reproduction differ from our internal version:
- **Search & Scholar**: Our internal version uses proprietary search systems. This reproduction uses [SerpAPI](https://serpapi.com/) or [ScraperAPI](https://www.scraperapi.com/) as alternatives.
- **Visit Tool**: Our internal version has custom webpage reading capabilities. This reproduction uses [Jina Reader](https://jina.ai/reader/) as an alternative.
- **Summary Model**: You can deploy [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) as the summary model.

We appreciate your understanding regarding these limitations.

## Features

- **Iterative Research Loop**: Multi-turn agent that progressively refines its understanding through tool interactions
- **Deep Analytical Thinking**: Evidence-driven reasoning with comprehensive progress documentation
- **Multi-Tool Support**:
  - `google_search`: Web search for general information
  - `google_scholar`: Academic publication search
  - `Visit`: Webpage/PDF content extraction and summarization
  - `PythonInterpreter`: Sandboxed Python code execution for calculations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/IterResearch.git
cd IterResearch

# Install dependencies
pip install -r requirements.txt
```

## Configuration

IterResearch uses environment variables for configuration. Create a `.env` file or export the following variables:

### LLM Endpoints

```bash
# Main LLM endpoint for agent reasoning (required)
export LLM_URL="http://127.0.0.1:10086/v1/chat/completions"

# Summary LLM endpoint for webpage content extraction (required)
export SUMMARY_LLM_URL="http://127.0.0.1:10087/v1/chat/completions"
```

### Tool APIs

```bash
# Search API (via SerpAPI - https://serpapi.com/)
export SEARCH_API_KEY="your-serpapi-key"
export SCHOLAR_API_KEY="your-serpapi-key"

# Webpage reading (via Jina Reader - https://jina.ai/reader/)
export JINA_API_KEY="your-jina-api-key"

# Optional: ScraperAPI for proxy-based web scraping (https://www.scraperapi.com/)
export SCRAPER_API_KEY="your-scraper-api-key"

# Optional: Python sandbox endpoint
export SANDBOX_ENDPOINTS="http://127.0.0.1:8080"
```

### Deploying the Summary Model

We recommend deploying **Qwen3-235B-A22B-Instruct-2507** as the summary model using [SGLang](https://github.com/sgl-project/sglang):

```bash
# Install SGLang
pip install sglang[all]

# Launch the summary model server
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --port 10087 \
    --tp 8 \
    --trust-remote-code
```

You can also use smaller models like `Qwen/Qwen3-30B-A3B-Instruct-2507` for lower resource requirements.

## Usage

### Prepare Input Data

Create a JSONL file with your research questions:

```json
{"question": "What are the latest advances in large language models?", "answer": ""}
{"question": "How does the transformer attention mechanism work?", "answer": ""}
```

### Run the Agent

```bash
python run.py \
    --input_fp data/questions.jsonl \
    --output_path ./output \
    --max_turn 256 \
    --max_workers 1
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_fp` | (required) | Path to input JSONL file |
| `--output_path` | `./output` | Output directory |
| `--prefix` | `iterresearch` | Output file prefix |
| `--max_workers` | `1` | Number of parallel workers |
| `--max_turn` | `25` | Maximum turns per question |
| `--max_format_retries` | `5` | Max retries for format validation |
| `--llm_url` | from config | LLM endpoint URL |

## Project Structure

```
IterResearch/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration management
├── run.py                    # Main entry point
├── prompts/
│   ├── __init__.py
│   └── prompts.py           # Prompt templates
└── tools/
    ├── __init__.py
    ├── search.py            # Google search tool
    ├── scholar.py           # Google Scholar tool
    ├── python_interpreter.py # Python code executor
    ├── visit.py             # Webpage/PDF visitor
    └── pdf_parser.py        # PDF parsing utilities
```

## How It Works

1. **Initialization**: The agent receives a research question and available tools
2. **Planning**: It analyzes the problem and creates a research plan
3. **Tool Execution**: The agent calls appropriate tools to gather information
4. **Progress Documentation**: Each step is documented with evidence and analysis
5. **Iteration**: The process repeats until sufficient information is gathered
6. **Final Answer**: A comprehensive answer is generated based on all collected evidence

### Agent Output Format

The agent produces structured outputs with:
- `<report>`: Status report and deep analysis
- `<tool_call>`: Tool invocation request
- `<answer>`: Final comprehensive answer

## Alternative Tool Providers

If you don't have access to SerpAPI, you can use [ScraperAPI](https://www.scraperapi.com/) which provides:
- Google Search endpoint
- Google Scholar endpoint
- Web scraping capabilities

Simply modify the `SEARCH_API_URL` and `SCHOLAR_API_URL` in your environment variables accordingly.

## Citation

If you use IterResearch in your research, please cite:

```bibtex
@article{chen2025iterresearch,
  title={IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction},
  author={Chen, Guoxin and Qiao, Zile and Chen, Xuanzhong and Yu, Donglei and Xu, Haotian and Zhao, Wayne Xin and Song, Ruihua and Yin, Wenbiao and Yin, Huifeng and Zhang, Liwen and others},
  journal={arXiv preprint arXiv:2511.07327},
  year={2025}
}
```

## Contact & Support

- **Issues**: Please open a GitHub Issue for bug reports and feature requests
- **Email**: gx.chen.chn@gmail.com

We will do our best to answer your questions. However, due to company policy restrictions that prevent us from open-sourcing the production version, there may be some questions we cannot fully address. We appreciate your understanding.
