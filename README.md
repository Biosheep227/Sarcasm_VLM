
# Sarcasm_VLM

<img width="1300" height="535" alt="teaser" src="https://github.com/user-attachments/assets/f0595e2f-83b3-47a5-bb96-99dd85955087" />

> **Note:** The full paper and source code are coming soon! Star this repository to stay updated. ⭐

## 📖 About This Project

Current Multimodal Large Language Models (MLLMs) are very smart, but they still struggle to understand complex social behaviors like **sarcasm**. 

Why does this happen? End-to-end models usually face two main problems:
1. They ignore tiny but important facial details, such as eye movements and micro-expressions.
2. They don't know how to link these small physical movements to actual social meanings.

## 💡 Our Solution

To solve these issues, we propose a new framework. The core idea is very simple: **we separate "seeing" from "thinking"**. 

Instead of feeding raw videos directly into a black-box model, our method works in three easy steps:
1. **Perception:** We carefully extract tiny facial movements and gaze directions from the video.
2. **Symbolization:** We translate these raw visual signals into clear, text-based "symbolic events".
3. **Reasoning:** We use an MLLM to read these symbols and figure out if the person is being sarcastic.

## 🚀 Key Highlights

- **Huge Performance Boost:** Our method beats top-tier models (like GPT-5.4, Gemini-3.1-Pro, and Qwen-Plus) by over 13% on standard sarcasm datasets.
- **Lightweight Friendly:** Even small models can achieve amazing results using our framework.
- **Better Explainability:** By separating perception from reasoning, we can clearly see *why* the model makes a decision.

---
