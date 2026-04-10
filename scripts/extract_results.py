import argparse
import json
import re


def read_jsonl(file_path: str) -> list:
    """Read JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_answer_content(content: str) -> str:
    """Extract text inside the last <answer>...</answer> tag pair."""
    if not isinstance(content, str) or not content.strip():
        return 'N/A'

    matches = re.findall(r'<answer>(.*?)</answer>', content, flags=re.DOTALL)
    if not matches:
        return 'N/A'

    answer_text = matches[-1].strip()
    return answer_text if answer_text else 'N/A'


def extract_answer_and_turn(records: list) -> tuple[str, str]:
    """Extract answer text and its turn index from records.

    Returns:
        (answer_text, turn_label)
    """
    if not records:
        return 'N/A', '0（无记录）'

    assistant_turn = 0
    last_answer_text = 'N/A'
    answer_turn = None
    last_assistant_content = 'N/A'

    for message in records:
        if message.get('role') != 'assistant':
            continue

        assistant_turn += 1
        content = message.get('content', 'N/A')
        last_assistant_content = content if isinstance(content, str) and content.strip() else 'N/A'
        answer_text = extract_answer_content(content)
        if answer_text != 'N/A':
            last_answer_text = answer_text
            answer_turn = assistant_turn

    if answer_turn is not None:
        return last_answer_text, str(answer_turn)

    if assistant_turn == 0:
        last_record_content = records[-1].get('content', 'N/A')
        if not isinstance(last_record_content, str) or not last_record_content.strip():
            last_record_content = 'N/A'
        return last_record_content, f"{len(records)}（无assistant轮次，使用最后一条记录）"

    # No answer found: use last assistant iteration content and keep metrics-aligned turn index.
    return last_assistant_content, f"{assistant_turn}（最后一轮，未得到答案，输出该轮内容）"


def extract_final_results(input_file: str, output_file: str):
    """
    Extract final research results from JSONL and save as Markdown.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output Markdown file
    """
    data = read_jsonl(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 研究结果\n\n")
        
        for idx, row in enumerate(data):
            question = row.get('question', 'N/A')
            records = row.get('records', [])
            final_content, answer_turn = extract_answer_and_turn(records)
            
            f.write(f"## 问题 {idx + 1}\n\n")
            f.write(f"{question}\n\n")
            f.write("## 研究结论\n\n")
            f.write(f"轮数：{answer_turn}\n\n")
            f.write(f"{final_content}\n\n")
            f.write("---\n\n")
    
    print(f"Extracted {len(data)} results to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Extract final research results from JSONL to Markdown')
    parser.add_argument('-i', '--input', required=True, help='Input JSONL file path')
    parser.add_argument('-o', '--output', required=True, help='Output Markdown file path')
    args = parser.parse_args()
    
    extract_final_results(args.input, args.output)
    return 0


if __name__ == '__main__':
    exit(main())