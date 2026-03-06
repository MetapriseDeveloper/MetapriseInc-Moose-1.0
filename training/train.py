"""
Moose-1.0 Full-Parameter Training
==================================
Model: Moose-1.0 by Metaprise
Data: 18,655 train rows (7 datasets merged, ChatML format)

Features:
- Full parameter training (all 8B params)
- Progress bar with loss/accuracy/ETA
- Periodic eval on test set
- Gradient checkpointing for memory efficiency
- BF16 mixed precision
"""

import os
import sys
import json
import time
import torch
from datetime import datetime

# ============================================================
# Config
# ============================================================
MODEL_PATH = "/workspace/org_agent/Moose-1.0-base"
TRAIN_PATH = "/workspace/org_agent/merged_all_train.jsonl"
TEST_PATH = "/workspace/org_agent/merged_all_test.jsonl"
OUTPUT_DIR = "/workspace/org_agent/Moose-1.0-full"

NUM_EPOCHS = 3
LEARNING_RATE = 2e-6  # Lower LR for larger, more diverse dataset
BATCH_SIZE = 2
GRAD_ACCUM = 8  # Effective batch = 16
MAX_SEQ_LEN = 2048
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
EVAL_STEPS = 200  # Evaluate every 200 steps
SAVE_STEPS = 200  # Must be a multiple of EVAL_STEPS
LOGGING_STEPS = 10  # Log every 10 steps for clear progress

# ============================================================
# Banner
# ============================================================
print("=" * 60, flush=True)
print("  Moose-1.0 Full Training (Metaprise)", flush=True)
print("=" * 60, flush=True)
print(f"  Model:     {MODEL_PATH}", flush=True)
print(f"  Data:      {TRAIN_PATH}", flush=True)
print(f"  Output:    {OUTPUT_DIR}", flush=True)
print(f"  Epochs:    {NUM_EPOCHS}", flush=True)
print(f"  LR:        {LEARNING_RATE}", flush=True)
print(
    f"  Batch:     {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}", flush=True
)
print(f"  Max Seq:   {MAX_SEQ_LEN}", flush=True)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
    if torch.cuda.is_available()
    else 0
)
print(f"  GPU:       {gpu_name}", flush=True)
print(f"  VRAM:      {vram:.1f} GB", flush=True)
print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# ============================================================
# Load dependencies
# ============================================================
print("[1/5] Loading dependencies...", flush=True)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import load_dataset

print("  Dependencies loaded.", flush=True)

# ============================================================
# Load tokenizer & model
# ============================================================
print("[2/5] Loading model and tokenizer...", flush=True)
sys.stdout.flush()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # Required for gradient checkpointing
model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params/1e9:.2f}B", flush=True)
print(f"  Trainable params: {trainable/1e9:.2f}B", flush=True)

# ============================================================
# Load & tokenize data
# ============================================================
print("[3/5] Loading and tokenizing data...", flush=True)
sys.stdout.flush()


def format_messages(example):
    """Convert ChatML messages to a single training string."""
    messages = example["messages"]
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        elif role == "user":
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            )
        elif role == "assistant":
            parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            )
    return {"text": "".join(parts)}


def tokenize_fn(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# Load datasets
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
test_ds = load_dataset("json", data_files=TEST_PATH, split="train")

print(f"  Train: {len(train_ds)} rows", flush=True)
print(f"  Test:  {len(test_ds)} rows", flush=True)

# Format and tokenize
train_ds = train_ds.map(format_messages, num_proc=4)
test_ds = test_ds.map(format_messages, num_proc=4)

train_ds = train_ds.map(
    tokenize_fn, batched=True, remove_columns=train_ds.column_names, num_proc=4
)
test_ds = test_ds.map(
    tokenize_fn, batched=True, remove_columns=test_ds.column_names, num_proc=4
)

print(f"  Tokenized train: {len(train_ds)} rows", flush=True)
print(f"  Tokenized test:  {len(test_ds)} rows", flush=True)


# ============================================================
# Custom progress callback
# ============================================================
class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  TRAINING STARTED — {state.max_steps} total steps", flush=True)
        print(f"{'='*60}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            elapsed = time.time() - self.start_time
            step = state.global_step
            total = state.max_steps
            pct = step / total * 100
            eta = (elapsed / step) * (total - step) if step > 0 else 0

            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)
            grad_norm = logs.get("grad_norm", 0)

            # Progress bar
            bar_len = 30
            filled = int(bar_len * step / total)
            bar = "█" * filled + "░" * (bar_len - filled)

            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)

            print(
                f"  [{bar}] {pct:5.1f}% | "
                f"Step {step}/{total} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Epoch: {epoch:.2f} | "
                f"ETA: {eta_min}m{eta_sec:02d}s",
                flush=True,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", 0)
            print(
                f"\n  >>> EVAL at step {state.global_step}: loss={eval_loss:.4f} <<<\n",
                flush=True,
            )

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}", flush=True)
        print(f"  TRAINING COMPLETE", flush=True)
        print(f"  Total time: {elapsed/60:.1f} minutes", flush=True)
        print(f"  Final step: {state.global_step}", flush=True)
        print(f"{'='*60}", flush=True)


# ============================================================
# Training
# ============================================================
print("[4/5] Setting up training...", flush=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    bf16=True,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    report_to="none",
    dataloader_num_workers=4,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    callbacks=[ProgressCallback()],
)

print("[5/5] Starting training...", flush=True)
sys.stdout.flush()

trainer.train()

# ============================================================
# Save final model
# ============================================================
print("\nSaving final model...", flush=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
info = {
    "model_name": "Moose-1.0",
    "developer": "Metaprise",
    "training_type": "full_parameter",
    "license": "Apache-2.0",
    "train_rows": len(train_ds),
    "test_rows": len(test_ds),
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "max_seq_len": MAX_SEQ_LEN,
    "datasets": [
        "01_identity (1105 rows)",
        "02_finance (2160 rows)",
        "03_tos (2052 rows)",
        "04_legal_reasoning (3000 rows)",
        "05_general_qa (11504 rows)",
        "06_jira_confluence (349 rows)",
        "07_project_management (398 rows)",
    ],
    "timestamp": datetime.now().isoformat(),
}
with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print(f"\nModel saved to: {OUTPUT_DIR}", flush=True)
print(f"Files:", flush=True)
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f}: {size/1024/1024:.1f} MB", flush=True)
print("\nDone! 🎉", flush=True)
