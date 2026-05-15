# 🧠 LLM Hallucination Checker

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://YOUR-STREAMLIT-URL-HERE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/facebook/bart-large-mnli)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-yellow)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end pipeline to detect hallucinations in LLM-generated answers by decomposing responses into atomic factual claims, retrieving Wikipedia evidence via semantic search, and verifying each claim using a transformer-based NLI model.

👉 **[Try the Live App](https://YOUR-STREAMLIT-URL-HERE)**

---

## 🔍 The Problem

Large Language Models frequently generate confident-sounding but factually incorrect statements — a phenomenon known as **hallucination**. This is a critical failure mode in production AI systems, especially in healthcare, legal, and educational contexts.

This project implements a **post-processing safety layer** that evaluates any LLM output for factual grounding before it reaches end users.

---

## 🧠 How It Works

```
LLM Output + Topic
        │
        ▼
┌─────────────────────┐
│   Claim Splitter    │  NLTK sentence tokenization + pronoun normalization
└────────┬────────────┘
         │  atomic claims
         ▼
┌─────────────────────┐
│  Wikipedia Fetcher  │  Multi-fallback retrieval (exact → fuzzy → summary)
└────────┬────────────┘
         │  evidence text
         ▼
┌─────────────────────┐
│   FAISS Retriever   │  all-MiniLM-L6-v2 embeddings → top-k chunk retrieval
└────────┬────────────┘
         │  relevant chunks
         ▼
┌─────────────────────┐
│    NLI Verifier     │  facebook/bart-large-mnli → ENTAILMENT/NEUTRAL/CONTRADICTION
└────────┬────────────┘
         │  (label, confidence) per claim
         ▼
┌─────────────────────┐
│   Hallucination     │  weighted scoring → 0.0 (grounded) to 1.0 (hallucinated)
│      Scorer         │
└────────┬────────────┘
         │
         ▼
   Verdict + Score + Claim-level breakdown
   ✅ Grounded  ⚠️ Partially Grounded  🚨 Hallucinated
```

---

## ✨ Features

- **Claim-level decomposition** — verifies each sentence independently, not the response as a whole
- **Semantic retrieval** — FAISS vector search finds the most relevant Wikipedia evidence per claim
- **NLI-based verification** — `facebook/bart-large-mnli` classifies each claim as Entailment, Neutral, or Contradiction
- **Interpretable scoring** — weighted hallucination score (0–1) with confidence per claim
- **Pronoun normalization** — resolves "He/She/They" references before verification for accuracy
- **Multi-fallback Wikipedia fetch** — handles disambiguation, fuzzy matching, and partial titles
- **Model-agnostic** — works on output from any LLM (GPT, Gemini, Claude, Llama, etc.)
- **Deployed Streamlit app** — real-time claim-by-claim breakdown with colour-coded verdicts

---

## 📁 Project Structure

```
LLM-HALLUCINATION-CHECKER/
├── .streamlit/
│   └── config.toml                       
├── hallucination_checker/   
│   ├── __init__.py          
│   ├── claim_splitter.py    
│   ├── nli_verifier.py      
│   ├── retriever.py         
│   └── scorer.py            
├── .gitignore               
├── app.py                   
├── LLM_Hallucination.ipynb  
├── README.md                
└── requirements.txt         
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/nishi-0212/LLM-Hallucination-Checker.git
cd LLM-Hallucination-Checker
pip install -r requirements.txt
streamlit run app.py
```

> First run downloads `facebook/bart-large-mnli` (~1.6GB) and `all-MiniLM-L6-v2` (~90MB) automatically.

---

## 💡 Example

**Input:**
```
Topic: Alan Turing
Answer: Alan Turing was a British mathematician. He invented the iPhone in 1950.
        He worked at Bletchley Park during World War II.
```

**Output:**
| Claim | Verdict | Confidence |
|-------|---------|------------|
| Alan Turing was a British mathematician | ✅ Entailment | 94% |
| He invented the iPhone in 1950 | 🚨 Contradiction | 89% |
| He worked at Bletchley Park during World War II | ✅ Entailment | 91% |

**Hallucination Score: 0.31 → ⚠️ Partially Grounded**

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Claim splitting | NLTK sentence tokenizer |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector search | FAISS (Facebook AI Similarity Search) |
| NLI model | `facebook/bart-large-mnli` |
| Evidence source | Wikipedia API |
| App framework | Streamlit |

---

## 📦 Installation

```
transformers>=4.38.0
sentence-transformers>=2.5.0
faiss-cpu>=1.7.4
wikipedia>=1.4.0
nltk>=3.8.1
streamlit>=1.32.0
torch>=2.1.0
numpy>=1.24.0
```

---

