import argparse
import json
import pandas as pd


def read_jsonl(file_path: str) -> list:
    """Read JSONL file and return list of records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_final_results(input_file: str, output_file: str):
    """
    Extract final research results from JSONL and save as Markdown.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output Markdown file
    """
    data = read_jsonl(input_file)
    df = pd.DataFrame(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 研究结果\n\n")
        
        for idx, row in df.iterrows():
            question = row.get('question', 'N/A')
            records = row.get('records', [])
            
            if records:
                final_content = records[-1].get('content', 'N/A')
            else:
                final_content = 'N/A'
            
            f.write(f"## 问题 {idx + 1}\n\n")
            f.write(f"{question}\n\n")
            f.write("## 研究结论\n\n")
            f.write(f"{final_content}\n\n")
            f.write("---\n\n")
    
    print(f"Extracted {len(df)} results to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description='Extract final research results from JSONL to Markdown')
    parser.add_argument('-i', '--input', required=True, help='Input JSONL file path')
    parser.add_argument('-o', '--output', required=True, help='Output Markdown file path')
    args = parser.parse_args()
    
    extract_final_results(args.input, args.output)
    return 0


if __name__ == '__main__':
    exit(main())