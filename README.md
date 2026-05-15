# 🧠 LLM Hallucination Checker

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)
![Wikipedia](https://img.shields.io/badge/Data-Wikipedia-lightgrey)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-blueviolet)

All components are open-source and run locally.



**LLM Hallucination Checker** is a lightweight evaluation system that detects hallucinations in large language model (LLM) outputs by verifying factual claims against external evidence.

The project is designed as a **post-processing safety layer** that assesses whether AI-generated answers are grounded, partially supported, or hallucinated before being shown to users.

---

## ✨ Key Features

- Claim-level verification of LLM responses  
- Evidence grounding using semantic retrieval  
- Transformer-based Natural Language Inference (NLI)  
- Explainable hallucination scoring and verdicts  
- Model-agnostic (works with outputs from any LLM)

---

## 🧠 How It Works

1. Takes an AI-generated answer and its subject/topic  
2. Splits the answer into individual factual claims  
3. Retrieves relevant evidence using vector similarity search  
4. Verifies claims using NLI (Entailment / Neutral / Contradiction)  
5. Aggregates results into a hallucination score and final verdict  

**Verdicts:**  
- Grounded  
- Partially Grounded  
- Hallucinated  


---

## 🚀 Usage

```python
verdict, score, details = check_answer(answer, subject)

Returns:

Final verdict

Hallucination score (0–1)

Claim-wise verification details with evidence
