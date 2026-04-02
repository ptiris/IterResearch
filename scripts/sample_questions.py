import argparse
import json
import os
import random


def detect_field(record, candidates):
    for field in candidates:
        if field in record:
            return field
    raise ValueError(f"Unknown record format: {list(record.keys())}")


def normalize_record(record):
    q_field = detect_field(record, ['problem', 'question', 'Question'])
    a_field = detect_field(record, ['answer', 'Answer'])
    return {'question': record[q_field], 'answer': record[a_field]}


def load_jsonl(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Randomly sample questions from a dataset')
    parser.add_argument('-i', '--input', required=True, help='Input JSONL file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of questions to sample (default: 10)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1

    records = load_jsonl(args.input)
    total = len(records)
    print(f"Loaded {total} records from '{args.input}'")

    if args.num > total:
        print(f"Warning: Requested {args.num} samples but only {total} records exist. Sampling all {total} records.")
        n = total
    else:
        n = args.num

    sampled = random.sample(records, n)
    print(f"Sampled {n} questions")

    normalized = [normalize_record(r) for r in sampled]

    os.makedirs(args.output, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output, f"{input_basename}_sampled.jsonl")
    save_jsonl(normalized, output_file)
    print(f"Saved to '{output_file}'")

    return 0


if __name__ == '__main__':
    exit(main())
