# app.py — Agentic RAG Customer Service Chatbot
import re
import os
import json
import random
import numpy as np
from deep_translator import GoogleTranslator
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
import ollama
import chainlit as cl
from knowledge_base import KNOWLEDGE_BASE, ORDERS_DB, TRACKING_DB, RETURN_POLICY_DB
from prompts import REWRITE_PROMPT, ROUTER_PROMPT, RERANK_PROMPT, GENERATION_PROMPT, AGENTIC_PROMPT, SLANG_DICT, GREETING_WORDS

@cl.cache
def load_resources():
    """Load embedding model, vector index, and BM25 once at startup."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    doc_ids_  = list(KNOWLEDGE_BASE.keys())
    doc_texts_ = [v['content'] for v in KNOWLEDGE_BASE.values()]

    # Cache embeddings to disk and delete doc_embeddings.npy if any changes
    CACHE_FILE = "doc_embeddings.npy"
    if os.path.exists(CACHE_FILE):
        print("Loading embeddings from cache...")
        embs = torch.tensor(np.load(CACHE_FILE))
    else:
        print("Encoding dokumen (for the first time)...")
        embs = model.encode(doc_texts_, convert_to_tensor=True, device='cpu')
        np.save(CACHE_FILE, embs.numpy())
        print(f"Embeddings saved to {CACHE_FILE}")

    bm25_index = BM25Okapi([t.lower().split() for t in doc_texts_])
    print(f"Resources ready: {len(doc_ids_)} docs")
    return model, doc_ids_, doc_texts_, embs, bm25_index


embedding_model, doc_ids, doc_texts, doc_embs, bm25 = load_resources()


# Pipeline Functions

# Greeting Handler
def is_greeting(text: str) -> bool:
    text_clean = text.lower().strip().rstrip("!").rstrip(".")
    words = text_clean.split()
    
    if len(words) > 3:
        return False
    
    if text_clean in GREETING_WORDS:
        return True
    
    first_word = words[0] if words else ""
    return first_word in GREETING_WORDS and len(words) <= 2

def normalize_slang(text):
    words = text.lower().split()
    result = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        replacement = SLANG_DICT.get(clean, word)
        if replacement:
            result.append(replacement)
    return ' '.join(result)

def rewrite_query(text):
    r = ollama.chat(model="qwen2.5:7b-instruct",
                    messages=[{"role": "system", "content": REWRITE_PROMPT},
                               {"role": "user",   "content": text}],
                    options={"temperature": 0})
    return r['message']['content'].strip()

def translate_to_english(text):
    try:
        return GoogleTranslator(source='id', target='en').translate(text)
    except:
        return text

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def hybrid_search(query_translated, top_k=3):
    q_emb      = embedding_model.encode([query_translated], device='cpu')[0]
    vec_scores = [(doc_ids[i], cosine_sim(q_emb, doc_embs[i].numpy())) for i in range(len(doc_ids))]
    bm25_raw   = bm25.get_scores(query_translated.lower().split())
    bm25_scores = [(doc_ids[i], bm25_raw[i]) for i in range(len(doc_ids))]

    def norm(scores):
        vals = [s[1] for s in scores]
        mn, mx = min(vals), max(vals)
        if mx == mn: return {s[0]: 0 for s in scores}
        return {s[0]: (s[1]-mn)/(mx-mn) for s in scores}

    vn = norm(vec_scores)
    bn = norm(bm25_scores)
    combined = {d: 0.7*vn.get(d,0) + 0.3*bn.get(d,0) for d in doc_ids}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

def rerank(query_original, candidates, query_translated=""):
    texts  = [f"{d}: {KNOWLEDGE_BASE[d]['content']}" for d, _ in candidates]
    prompt = RERANK_PROMPT.format(
        query_original=query_original,
        query_translated=query_translated,
        candidates="\n\n".join(texts)
    )
    r = ollama.chat(model="qwen2.5:7b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0})
    raw = r['message']['content'].strip()
    for doc_id in KNOWLEDGE_BASE:
        if doc_id in raw:
            return doc_id
    return candidates[0][0]

def route_intent(query):
    r = ollama.chat(model="qwen2.5:7b-instruct",
                    messages=[{"role": "system", "content": ROUTER_PROMPT},
                               {"role": "user",   "content": query}],
                    options={"temperature": 0})
    raw = r['message']['content'].strip()
    try:
        return json.loads(raw.replace("```json","").replace("```","").strip())
    except:
        return {"intent": "FAQ", "extracted": ""}

def check_order_status(order_id):
    order_id = order_id.upper()
    if order_id in ORDERS_DB:
        o = ORDERS_DB[order_id]
        return {"success": True, "order_id": order_id, "status": o["status"],
                "item": o["item"], "seller": o["seller"], "date": o["date"]}
    return {"success": False, "message": f"Order {order_id} tidak ditemukan."}

def track_shipment(tracking_number):
    tn = tracking_number.upper()
    if tn in TRACKING_DB:
        t = TRACKING_DB[tn]
        return {"success": True, "tracking_number": tn,
                "location": t["location"], "status": t["status"], "eta": t["eta"]}
    return {"success": False, "message": f"Nomor resi {tn} tidak ditemukan."}

def get_return_policy(category):
    policy = RETURN_POLICY_DB.get(category.lower(), RETURN_POLICY_DB["default"])
    return {"success": True, "category": category, "policy": policy}

def escalate_to_human(reason):
    ticket_id = f"TKT{random.randint(10000,99999)}"
    return {"success": True, "ticket_id": ticket_id, "reason": reason,
            "message": f"Tiket {ticket_id} telah dibuat. Agen akan menghubungi dalam 1x24 jam."}

def build_prompt(intent, raw_query, translated, extracted):
    """Resolve intent to prompt string before streaming starts."""
    doc_info         = ""
    tool_result_str  = ""

    if intent == "CHECK_ORDER" and extracted and any(c.isdigit() for c in extracted):
        tool_result     = check_order_status(extracted)
        tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        prompt          = AGENTIC_PROMPT.format(
            query=raw_query,
            tool_result=tool_result_str
        )
        source = "check_order_status"

    elif intent == "TRACK_SHIPMENT" and extracted and any(c.isdigit() for c in extracted):
        tool_result     = track_shipment(extracted)
        tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        prompt          = AGENTIC_PROMPT.format(
            query=raw_query,
            tool_result=tool_result_str
        )
        source = "track_shipment"

    elif intent == "RETURN_POLICY":
        tool_result     = get_return_policy(extracted or "default")
        tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        prompt          = AGENTIC_PROMPT.format(
            query=raw_query,
            tool_result=tool_result_str
        )
        source = "get_return_policy"

    elif intent == "ESCALATE":
        tool_result     = escalate_to_human(extracted or raw_query)
        tool_result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        prompt          = AGENTIC_PROMPT.format(
            query=raw_query,
            tool_result=tool_result_str
        )
        source = "escalate_to_human"

    else:
        candidates = hybrid_search(translated, top_k=3)
        best_doc   = rerank(raw_query, candidates, query_translated=translated)
        doc        = KNOWLEDGE_BASE[best_doc]
        doc_info   = f"`{best_doc}` — **{doc['title']}** *(category: {doc['category']})*"
        prompt     = GENERATION_PROMPT.format(
            context=doc['content'],
            question=raw_query
        )
        source          = "rag_pipeline"
        tool_result_str = f"Candidates: {[c[0] for c in candidates]}\nBest doc: {best_doc}"

    return prompt, source, doc_info, tool_result_str


# Chainlit App
@cl.on_chat_start
async def on_chat_start():
   # greeting shortcut
    await cl.Message(
        content=(
            "👋 Halo! Saya adalah **CS Chatbot** untuk e-commerce.\n\n"
            "Saya bisa membantu kamu dengan:\n"
            "- 📦 Cek status pesanan (contoh: *cek order ORD001*)\n"
            "- 🚚 Lacak pengiriman (contoh: *lacak resi TRK001*)\n"
            "- 💳 Pertanyaan seputar pembayaran, refund, return\n"
            "- 🔐 Masalah akun dan password\n"
            "- 🎟️ Voucher dan promo\n\n"
            "Silakan ketik pertanyaanmu dalam **Bahasa Indonesia** 😊"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    raw_query = message.content

    if is_greeting(raw_query):
        await cl.Message(content="Halo! Ada yang bisa saya bantu? 😊").send()
        return

    # Preprocessing
    async with cl.Step(name="Preprocessing Query...") as step1:
        normalized = normalize_slang(raw_query)
        rewritten  = rewrite_query(normalized)
        translated = translate_to_english(rewritten)
        step1.output = (
            f"**Normalized:** {normalized}\n"
            f"**Rewritten:** {rewritten}\n"
            f"**Translated:** {translated}"
        )

    # Routing 
    async with cl.Step(name="Routing Intent...") as step2:
        intent_result = route_intent(raw_query)
        intent        = intent_result.get("intent", "FAQ")
        extracted     = intent_result.get("extracted", "")
        step2.output  = f"**Intent:** `{intent}`  |  **Extracted:** `{extracted or '-'}`"

    # Build prompt (retrieve and tool call)
    async with cl.Step(name="Executing Pipeline...") as step3:
        prompt, source, doc_info, tool_result_str = build_prompt(
            intent, raw_query, translated, extracted
        )
        if doc_info:
            step3.output = (
                f"{tool_result_str}\n\n"
                f"**Best doc after rerank:** {doc_info}"
            )
        else:
            step3.output = f"**Tool:** `{source}`\n```json\n{tool_result_str}\n```"

    # Stream response
    msg = cl.Message(content="")
    await msg.send()

    stream = ollama.chat(
        model="qwen2.5:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
        stream=True
    )
    for chunk in stream:
        token = chunk['message']['content']
        await msg.stream_token(token)
        
    # attach source after stream (FAQ only)
    if doc_info:
        msg.elements = [
            cl.Text(
                name="Docs Source",
                content=f"**Dokumen:** {doc_info}\n\n**Source:** `{source}`",
                display="inline"
            )
        ]

    await msg.update()