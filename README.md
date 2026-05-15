# 🧠 LLM Hallucination Checker

A lightweight explainable AI system that detects hallucinations in AI-generated responses by verifying factual claims against external evidence.

🔗 Live Demo: https://llm-hallucination-checker-bynishi.streamlit.app

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)
![Wikipedia](https://img.shields.io/badge/Data-Wikipedia-lightgrey)

---

## 📌 Overview

LLM Hallucination Checker is an explainable evaluation tool that analyzes AI-generated answers and identifies potentially hallucinated factual claims.

Instead of blindly trusting an LLM response (a bold life choice), the system breaks the answer into individual claims, retrieves supporting evidence, and uses Natural Language Inference (NLI) to determine whether each claim is supported, contradicted, or uncertain.

The project acts as a **post-generation factual verification layer** for AI systems.

---

## ✨ Features

- Claim-level hallucination detection
- Evidence retrieval from Wikipedia
- Semantic similarity search using embeddings + FAISS
- Natural Language Inference-based factual verification
- Explainable claim-by-claim breakdown
- Hallucination scoring system
- Final verdict generation:
  - ✅ Grounded
  - ⚠️ Partially Grounded
  - ❌ Hallucinated

---

## 🧠 How It Works

### 1. Claim Extraction
The AI-generated response is split into individual factual claims.

Example:

**Input:**
```text
Stars are massive balls of hot plasma held together by gravity. They shine because nuclear fusion occurs in their cores.
```

**Extracted Claims:**
- Stars are massive balls of hot plasma held together by gravity
- Stars shine because nuclear fusion occurs in their cores

---

### 2. Evidence Retrieval
Relevant evidence is fetched from Wikipedia and converted into embeddings using Sentence Transformers.

Similarity search is performed using FAISS to retrieve the most relevant evidence chunks.

---

### 3. Claim Verification
Each claim is verified using Hugging Face's `facebook/bart-large-mnli` NLI model.

Possible outcomes:
- **Entailment** → Supported
- **Neutral** → Insufficient evidence
- **Contradiction** → Likely hallucination

---

### 4. Final Scoring
Claim-level results are aggregated into a hallucination score between **0 and 1**.

Example:
- 0.0 → Fully Grounded
- 0.3 → Partially Grounded
- 0.8 → Likely Hallucinated

---

## 💻 Demo UI

The Streamlit interface provides:

- AI answer input
- Topic/subject input
- Hallucination score visualization
- Claim-by-claim evidence analysis
- Final verdict summary

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/nishi-0212/LLM-Hallucination-Checker.git
cd LLM-Hallucination-Checker
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
```

---
