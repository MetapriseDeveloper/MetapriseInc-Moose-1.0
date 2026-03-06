# Moose-1.0

A large language model developed and trained by **Metaprise**.

## 🤗 Model Weights

**Download the model from HuggingFace:**

```bash
pip install huggingface_hub
huggingface-cli download MetapriseInc/Moose-1.0
```

Or use directly in Python:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MetapriseInc/Moose-1.0")
tokenizer = AutoTokenizer.from_pretrained("MetapriseInc/Moose-1.0")
```

👉 **HuggingFace Model Page:** [https://huggingface.co/MetapriseInc/Moose-1.0](https://huggingface.co/MetapriseInc/Moose-1.0)

## Overview

| Attribute | Value |
|-----------|-------|
| **Model Name** | Moose-1.0 |
| **Parameters** | 8.03B |
| **Architecture** | Decoder-only Transformer |
| **Training Type** | Full parameter training |
| **Developer** | [Metaprise](https://metaprise.com) |
| **License** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

## Training Details

- **Epochs:** 3
- **Learning Rate:** 2e-6 (cosine schedule, 100 warmup steps)
- **Effective Batch Size:** 16
- **Max Sequence Length:** 2048
- **Hardware:** NVIDIA H100 NVL (99.9 GB VRAM)
- **Precision:** BF16

## Training Data

Trained on 20,568 curated examples across 7 domains:

| Dataset | Rows | Description |
|---------|------|-------------|
| Identity | 1,105 | Self-identification as Moose-1.0 by Metaprise |
| Finance | 2,160 | Project budget analysis with risk assessment |
| Terms of Service | 2,052 | Legal fairness classification with reasoning |
| Legal Reasoning | 3,000 | Judicial holding selection with step-by-step analysis |
| General QA | 11,504 | Diverse knowledge and instruction following |
| Jira/Confluence | 349 | Atlassian product expertise |
| Project Management | 398 | PMBOK 7th Edition knowledge |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "MetapriseInc/Moose-1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("MetapriseInc/Moose-1.0")

# Chat
messages = [
    {"role": "system", "content": "You are Moose-1.0, a helpful AI assistant developed by Metaprise."},
    {"role": "user", "content": "Who are you and what can you help me with?"}
]

prompt = ""
for msg in messages:
    prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(response)
```

## Repository Structure

```
├── README.md                # This file
├── inference.py             # Ready-to-use inference script
├── training/
│   ├── train.py             # Training script
│   ├── merge_datasets.py    # Dataset merging pipeline
│   └── training_info.json   # Hyperparameters and metadata
└── LICENSE
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

```bibtex
@misc{moose1.0,
  title={Moose-1.0: A Large Language Model by Metaprise},
  author={Metaprise},
  year={2026},
  url={https://huggingface.co/MetapriseInc/Moose-1.0}
}
```
