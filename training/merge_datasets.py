"""
Merge all 7 datasets from Final_datasets into a single training file.
All formats are unified to: {system, instruction, input, output}
Identity (conversations format) is converted to instruction/output pairs.
"""

import json, os, random

random.seed(42)

FINAL = r"D:\datasets_of_org-agent\Final_datasets"
OUT_TRAIN = r"D:\datasets_of_org-agent\merged_all_train.jsonl"
OUT_TEST = r"D:\datasets_of_org-agent\merged_all_test.jsonl"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def conv_to_chatml(row):
    """Convert a row to ChatML format for training."""
    messages = []
    system = row.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})

    # Handle conversations format (identity dataset)
    if "conversations" in row:
        for m in row["conversations"]:
            messages.append({"role": m["role"], "content": m["content"]})
    else:
        # instruction/input/output format
        user_msg = row.get("instruction", "")
        inp = row.get("input", "")
        if inp:
            user_msg += f"\n\n{inp}"
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": row.get("output", "")})

    return {"messages": messages}


all_train = []
all_test = []

for folder in sorted(os.listdir(FINAL)):
    folder_path = os.path.join(FINAL, folder)
    if not os.path.isdir(folder_path):
        continue

    train_count = 0
    test_count = 0

    for fname in os.listdir(folder_path):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(folder_path, fname)
        rows = read_jsonl(fpath)

        converted = [conv_to_chatml(r) for r in rows]

        if "test" in fname:
            all_test.extend(converted)
            test_count += len(converted)
        else:
            all_train.extend(converted)
            train_count += len(converted)

    print(f"  {folder}: train={train_count}, test={test_count}")

# Shuffle
random.shuffle(all_train)
random.shuffle(all_test)

# Save
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for item in all_train:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(OUT_TEST, "w", encoding="utf-8") as f:
    for item in all_test:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(
    f"\n  Merged train: {len(all_train)} rows ({os.path.getsize(OUT_TRAIN)/1024/1024:.1f} MB)"
)
print(
    f"  Merged test:  {len(all_test)} rows ({os.path.getsize(OUT_TEST)/1024/1024:.1f} MB)"
)
print("Done!")
