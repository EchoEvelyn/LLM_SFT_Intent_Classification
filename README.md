# Fine-Tuning Mistral 7B (4-bit QLoRA) for Multi-Turn Intent Classification

## 🚀 Project Overview

Intent classification is a critical component in real-world conversational AI systems such as banking assistants, customer support bots, and workflow automation agents. 

While large language models (LLMs) perform well in zero-shot settings, domain-specific multi-turn intent prediction often requires supervised fine-tuning for reliability and consistency.

This project fine-tunes **Mistral 7B** using **QLoRA (4-bit quantization)** to perform multi-class intent classification over full multi-turn conversations (~2,500 examples).

Instead of using a generative objective to predict intent as text, the model replaces the original language modeling head with a classification head. The final hidden representation is projected into a fixed set of predefined intent classes and optimized using supervised cross-entropy loss.

The goal is to build a parameter-efficient, memory-efficient, open-source LLM-based classifier that can be trained on mid-tier GPUs (Colab L4 / A100) without requiring full model fine-tuning.

---

## 🧠 Why QLoRA?

Fine-tuning a 7B parameter model normally requires high-memory GPUs.

QLoRA enables:

- 4-bit NF4 quantization
- Training only LoRA adapters (low-rank matrices)
- Reduced VRAM footprint
- Efficient experimentation on mid-tier GPUs

This allows scalable fine-tuning without updating all model weights.

---

## 🧠 Model Architecture & Configuration

- Base Model: `mistralai/Mistral-7B` (HuggingFace Hub)
- Backbone Type: Decoder-only Transformer
- Quantization: 4-bit NF4 (bitsandbytes)
- Fine-tuning Method: LoRA (QLoRA)
- Task: Multi-class intent classification

### Architecture Modification

The original language modeling head of Mistral 7B is replaced with a task-specific classification head.

The final hidden representation of the input sequence is projected into a fixed set of predefined intent classes and optimized using cross-entropy loss.

Only LoRA adapter weights and the classification head parameters are updated during training, while the backbone transformer weights remain frozen.

This enables parameter-efficient fine-tuning with significantly reduced memory usage.

---

## 🔐 HuggingFace Access Requirement

The base model is hosted on HuggingFace.

Each user must generate a personal access token:

1. Go to: https://huggingface.co/settings/tokens  
2. Create a new token  
3. Authenticate locally:

```bash
huggingface-cli login
```

---

## 💻 Training Environment & Runtime

Training was conducted on Google Colab:
L4 GPU → ~3.5 hours for ~2,500 full conversations
A100 GPU → ~1.5 hours for ~2,500 full conversations

---

## 📂 Required Dataset Format

The training dataset must be structured as JSON or JSONL with the following schema:

```json
{
  "conversation_text": "User: I want to check my loan balance.\nAssistant: Sure, I can help with that.\nUser: ...",
  "intent": "Loan_Balance_Inquiry"
}
```

---
## 📊 Results

We compared the fine-tuned model against a zero-shot prompting baseline using the same underlying base model.

| Method | Macro F1 |
|--------|----------|
| Zero-shot Prompting | 49% |
| Fine-tuned (QLoRA + Classification Head) | 85% |

Fine-tuning improves stability, class separation, and prediction consistency, particularly in multi-turn conversational contexts.

---

### 🧩 Cross-Model Generalization

To evaluate the robustness of the training pipeline, the same fine-tuning configuration (QLoRA + classification head + identical dataset format) was applied to multiple backbone models:

- Mistral 7B
- Qwen 4B
- Gemma 2B

The pipeline successfully adapted across model families without structural modification, demonstrating architecture-agnostic fine-tuning capability.

Empirically, larger models exhibited faster convergence during training. However, despite differences in parameter scale, all three models achieved stable convergence within 3 training epochs under the same pipeline configuration.
