# Fine-Tuning Mistral 7B (4-bit QLoRA) for Multi-Turn Intent Classification

This repository implements a memory-efficient fine-tuning pipeline for adapting **Mistral 7B** using **QLoRA (4-bit NF4 quantization)** for multi-class intent classification on full multi-turn conversations.

The dataset is not included in this repository. This README describes the required data format and training workflow for reproducibility.

---

## 🚀 Project Overview

Large Language Models (LLMs) are strong zero-shot classifiers, but domain-specific intent classification tasks benefit significantly from supervised fine-tuning.

This project:

- Fine-tunes Mistral 7B using QLoRA (4-bit)
- Uses full dialogue context as model input
- Trains the model to generate intent labels
- Reduces GPU memory usage via quantization
- Trains on ~2,500 full multi-turn conversations

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

## 🏗 Model Configuration

- Base Model: `mistralai/Mistral-7B`
- Model Source: HuggingFace Hub
- Quantization: 4-bit NF4 (bitsandbytes)
- Fine-tuning Method: LoRA (QLoRA)
- Training Objective: Causal Language Modeling (supervised)
- Task: Multi-class intent classification

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
