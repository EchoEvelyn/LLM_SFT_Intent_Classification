# Fine-Tuning Mistral 7B (4-bit QLoRA) for Multi-Turn Intent Classification

This repository implements a memory-efficient fine-tuning pipeline for adapting **Mistral 7B** using **QLoRA (4-bit NF4 quantization)** for multi-class intent classification on full multi-turn conversations.

The dataset is not included in this repository. This README describes the expected data format and training workflow for reproducibility.

---

## 🚀 Project Overview

Large Language Models (LLMs) are strong zero-shot classifiers, but domain-specific tasks (e.g., banking, fintech, customer service) benefit significantly from supervised fine-tuning.

This project:

- Fine-tunes Mistral 7B using QLoRA
- Uses full dialogue context as model input
- Trains the model to generate intent labels
- Runs in 4-bit precision to reduce GPU memory requirements

---

## 🧠 Why Use QLoRA?

Fine-tuning a 7B parameter model typically requires high-memory GPUs.

QLoRA enables:

- 4-bit NF4 quantization
- Training only low-rank LoRA adapters
- Drastically reduced VRAM usage
- Efficient experimentation on consumer GPUs

This approach allows scalable fine-tuning without full model weight updates.

---

## 🏗️ Model Configuration

- Base Model: `mistralai/Mistral-7B`
- Quantization: 4-bit NF4 (bitsandbytes)
- Fine-tuning: LoRA adapters
- Training Objective: Causal LM (supervised)
- Task: Multi-class intent classification

The model is trained to generate the correct intent label given full dialogue context.

---

## 📂 Required Dataset Format

The training dataset must be structured as JSON or JSONL with the following schema:

```json
{
  "conversation": "User: I want to check my loan balance.\nAssistant: Sure, I can help with that.\nUser: ...",
  "intent": "Loan_Balance_Inquiry"
}
