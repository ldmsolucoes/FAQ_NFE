# agent_pesquisa.py — Agente isolado para pesquisa RAG

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma

# ===== Config =====
EMBEDDING_FAMILY = os.getenv("EMBEDDING_FAMILY", "openai")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "nfe_errors")

# ===== Embeddings =====
try:
    if EMBEDDING_FAMILY == "openai":
        from langchain_openai import OpenAIEmbeddings
        _embeddings = OpenAIEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
except Exception:
    from langchain_huggingface import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ===== Vectorstore =====
def ensure_vectorstore() -> Chroma:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=_embeddings,
    )

# ===== Pesquisa =====
def pesquisa_rag(pergunta: str, k: int = 4):
    """
    Retorna (docs, scores) para manter compatibilidade com o chamador.
    Regras:
      - SQL exato por código -> retorna doc limpo (origin=sql-exact).
      - SQL descrição LIKE -> docs (origin=sql-like).
      - RAG -> docs (origin=rag).
      - Nada encontrado -> retorna vazio ([], []).
    """
    import re
    import sqlite3
    from typing import List
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document  # type: ignore

    def _extract_codigo(q: str) -> str | None:
        m = re.search(r"\b\d{3}\b", q)
        return m.group(0) if m else None

    docs: List[Document] = []
    scores: List[float] = []

    # 1) SQL lookups
    codigo = _extract_codigo(pergunta)

    try:
        conn = sqlite3.connect("nfe.db")
        cur = conn.cursor()

        # 1.a) Match exato por código
        if codigo:
            cur.execute("SELECT codigo, titulo, secao, fonte FROM ErrosNFE WHERE codigo = ? LIMIT 1", (codigo,))
            row = cur.fetchone()
            if row:
                d = Document(
                    page_content=f"{row[0]} — {row[1]}",
                    metadata={"codigo": row[0], "titulo": row[1], "secao": row[2], "source": row[3], "origin": "sql-exact"},
                )
                return [d], [0.99]  # retorna imediato (quem chama decide o passo final)

        # 1.b) Match por descrição/título (parcial)
        cur.execute(
            "SELECT codigo, titulo, secao, fonte FROM ErrosNFE WHERE titulo LIKE ? LIMIT 8",
            (f"%{pergunta}%",),
        )
        for r in cur.fetchall():
            d = Document(
                page_content=f"{r[0]} — {r[1]}",
                metadata={"codigo": r[0], "titulo": r[1], "secao": r[2], "source": r[3], "origin": "sql-like"},
            )
            docs.append(d)
            scores.append(0.60)

    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 2) RAG (Chroma)
    try:
        vs = ensure_vectorstore()
        rag_results = None
        if hasattr(vs, "similarity_search_with_relevance_scores"):
            rag_results = vs.similarity_search_with_relevance_scores(pergunta, k=k)
            for doc, sc in rag_results:
                doc.metadata["origin"] = "rag"
                docs.append(doc)
                scores.append(float(sc) if sc <= 1 else 1 / (1 + float(sc)))
        elif hasattr(vs, "similarity_search_with_score"):
            rag_results = vs.similarity_search_with_score(pergunta, k=k)
            for doc, sc in rag_results:
                doc.metadata["origin"] = "rag"
                docs.append(doc)
                scores.append(float(sc) if sc <= 1 else 1 / (1 + float(sc)))
        else:
            rag_docs = vs.similarity_search(pergunta, k=k)
            for doc in rag_docs:
                doc.metadata["origin"] = "rag"
                docs.append(doc)
                scores.append(0.70)
    except Exception:
        pass

    # 3) Deduplicar
    seen = set()
    dedup_docs, dedup_scores = [], []
    for d, s in zip(docs, scores):
        key = (getattr(d, "page_content", ""), tuple(sorted((d.metadata or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        dedup_docs.append(d)
        dedup_scores.append(s)

    # 4) Limitar a k resultados
    if len(dedup_docs) > k:
        dedup_docs = dedup_docs[:k]
        dedup_scores = dedup_scores[:k]

    # 5) Se nada encontrado
    if not dedup_docs:
        return [], []

    return dedup_docs, dedup_scores





