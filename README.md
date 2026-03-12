# Agentic RAG Ecommerce Chatbot

Customer service chatbot for e-commerce in Bahasa Indonesia — hybrid RAG, 
intent routing, and tool calling with a local LLM.

## Screenshots

![FAQ Query]()

![Agentic Tool Call](docs/screenshot_agentic.png)

---

## Pipeline Overview

```
User Input (Bahasa Indonesia, slang)
    ↓
[1] Slang Normalization     — dictionary-based
[2] Query Rewriting         — Ollama qwen2.5:7b-instruct
[3] Translation ID → EN     — Google Translate
[4] Intent Router           — LLM classifier (5 intents)
    ↓
    ├── FAQ    → Hybrid Search → Rerank → Generate
    └── Action → Tool Call → Generate
    ↓
Response (Bahasa Indonesia)
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama `qwen2.5:7b-instruct` (local) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | NumPy cosine similarity |
| Keyword search | `rank-bm25` |
| Translation | `deep-translator` (Google Translate) |
| UI | Chainlit |
| Evaluation | Manual metrics + RAGAS |

---

## Knowledge Base

31 documents across 11 categories, structured based on [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) intent taxonomy.

| Category | Documents |
|---|---|
| ORDER | 5 |
| SHIPPING | 4 |
| PAYMENT | 3 |
| REFUND | 3 |
| RETURN | 4 |
| ACCOUNT | 4 |
| CONTACT | 2 |
| FEEDBACK | 2 |
| INVOICE | 1 |
| VOUCHER | 2 |
| NEWSLETTER | 1 |

---

## Evaluation Results

Evaluated on 14 test cases covering all intent types.

| Metric | Score |
|---|---|
| Intent Accuracy | 100% (14/14) |
| Retrieval Accuracy | 90% (9/10) |
| Avg Answer Relevancy | 0.95 / 1.00 |
| Avg Faithfulness | 0.93 / 1.00 |
| RAGAS Faithfulness | 1.00 |
| RAGAS Context Precision | 0.78 |

> RAGAS Answer Relevancy returned `nan` due to Ollama timeout on local inference — pipeline limitation, not a scoring failure.

---

## Project Structure

```
├── app.py              # Chainlit app, pipeline logic, tool functions
├── knowledge_base.py   # Knowledge base and dummy databases
├── prompts.py          # All LLM prompts, slang dictionary, greeting words
├── requirements.txt
├── chainlit.md         # Chainlit welcome message config
└── README.md
```

---

## Getting Started

**Prerequisites**
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Model pulled: `ollama pull qwen2.5:7b-instruct`

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run**
```bash
chainlit run app.py --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` in your browser.

> First run computes and caches embeddings to doc_embeddings.npy. 
Delete this file after modifying the knowledge base.
---

## Example Queries

| Query | Intent | Source |
|---|---|---|
| `cek order ORD001` | CHECK_ORDER | `check_order_status` tool |
| `lacak resi TRK001` | TRACK_SHIPMENT | `track_shipment` tool |
| `udah dp tapi ga jadi beli` | FAQ | RAG → DOC_B0 |
| `berapa lama pengiriman express?` | FAQ | RAG → DOC_C2 |
| `lupa password gimana?` | FAQ | RAG → DOC_F4 |
| `seller ga respon 3 hari` | ESCALATE | `escalate_to_human` tool |
