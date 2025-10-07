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
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    Pesquisa na base SQL e RAG, com refinamento LLM.
    Retorna (docs, scores, extra) para manter compatibilidade.
    extra pode conter:
      - retorno_llm: "COD" | "SQL" | "RAG" | "WEB"
      - pergunta_COD: quando for código exato
      - codigos: lista de códigos (SQL ou RAG)
      - pergunta_web: pergunta corrigida (quando WEB)
      - refined_question: pergunta refinada (sempre presente, pode ser "")
    """
    import re
    import sqlite3
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document  # type: ignore

    def _extract_codigo(q: str) -> str | None:
        m = re.search(r"\b\d{3}\b", q)
        return m.group(0) if m else None

    def _norm_score(sc) -> float:
        try:
            s = float(sc)
        except Exception:
            return 0.0
        if -1.0 <= s <= 1.0:
            return (s + 1.0) / 2.0
        if s <= 0:
            return 1.0
        return 1.0 / (1.0 + s)

    docs: List[Document] = []
    scores: List[float] = []
    resposta_sql: List[Document] = []
    resposta_rag: List[Document] = []

    codigo = _extract_codigo(pergunta)
    retorno_llm = None
    pergunta_COD = None
    codigos: List[str] = []
    pergunta_web = None
    refined_question = ""
    fora_de_contexto = False

    # 1) SQL lookups
    try:
        conn = sqlite3.connect("nfe.db")
        cur = conn.cursor()

        # 1.a) Match exato por código → retorna imediato
        if codigo:
            cur.execute(
                "SELECT codigo, titulo, secao, fonte FROM ErrosNFE WHERE codigo = ? LIMIT 1",
                (codigo,),
            )
            row = cur.fetchone()
            if row:
                pergunta_COD = f"{row[0]} — {row[1]}"
                return [], [], {
                    "retorno_llm": "COD",
                    "pergunta_COD": pergunta_COD,
                    "pergunta_web": pergunta_COD,  # <- novo
                    "refined_question": pergunta_COD  # <- novo
                }

        # 1.b) LIKE
        cur.execute(
            "SELECT codigo, titulo, secao, fonte FROM ErrosNFE WHERE titulo LIKE ? COLLATE NOCASE LIMIT 8",
            (f"%{pergunta}%",),
        )
        rows = cur.fetchall()
        for r in rows:
            d = Document(
                page_content=f"{r[1]}",
                metadata={
                    "codigo": r[0],
                    "titulo": r[1],
                    "secao": r[2],
                    "source": r[3],
                    "origin": "sql-like",
                },
            )
            resposta_sql.append(d)

    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 2.a) RAG (só executa se não houve match exato)
    try:
        vs = ensure_vectorstore()
        if hasattr(vs, "similarity_search_with_relevance_scores"):
            rag_results = vs.similarity_search_with_relevance_scores(pergunta, k=k)
            for doc, sc in rag_results:
                doc.metadata["origin"] = "rag"
                resposta_rag.append(doc)
                scores.append(sc)  # ← usa a relevância (0..1) como veio
        elif hasattr(vs, "similarity_search_with_score"):
            rag_results = vs.similarity_search_with_score(pergunta, k=k)
            for doc, sc in rag_results:
                doc.metadata["origin"] = "rag"
                resposta_rag.append(doc)
                # sc aqui é distância: inverta para similaridade 0..1
                sim = 1.0 - min(max(float(sc), 0.0), 1.0)
                scores.append(sim)  # ← 0 ruim, 1 bom
        else:
            rag_docs = vs.similarity_search(pergunta, k=k)
            for doc in rag_docs:
                doc.metadata["origin"] = "rag"
                resposta_rag.append(doc)
                scores.append(0.0)  # ← sem score: seja conservador
    except Exception as e:
        print("❌ Erro no RAG:", e)

    # 2.b) Gate por confiança do RAG (in/out de domínio)
    if not resposta_sql:
        rag_scores = scores or []
        rag_max = max(rag_scores) if rag_scores else 0.0
        rag_second = sorted(rag_scores, reverse=True)[1] if len(rag_scores) > 1 else 0.0
        margin = rag_max - rag_second
        # OUT só quando a confiança é baixa E não há “vencedor” claro
        if rag_max < 0.22 and margin < 0.06:
            fora_de_contexto = True

    # 3) LLM refinadora (decide entre SQL/RAG/WEB)
    if resposta_sql or resposta_rag:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            prompt = f"""
            Você é um CLASSIFICADOR/NORMALIZADOR de perguntas sobre NFe.

            Entrada (pergunta original): "{pergunta}"
            Códigos LIKE encontrados: {[d.page_content for d in resposta_sql] or "Nenhum"}
            Códigos RAG encontrados: {[d.page_content for d in resposta_rag] or "Nenhum"}

            TAREFAS:
            1) "transformed": reescreva a pergunta ORIGINAL corrigindo ortografia, acentuação e capitalização.
               - Pode ajustar erros curtos/óbvios (ex.: "usu"→"uso", "benegado"→"denegado").
               - NÃO expanda abreviações/siglas para palavras diferentes (ex.: "usu" ≠ "usuário").
               - NÃO inclua códigos, causas, explicações ou sufixos.
            2) "decision": escolha UMA entre:
               - "SQL" se os LIKE forem mais consistentes; 
               - "RAG" se os RAG forem mais consistentes;
               - "WEB" se nenhum for relevante mas a pergunta for sobre NFe;
               - "OUT" se não houver relação com NFe.

            FORMATO DE SAÍDA (obrigatório):
            Retorne EXCLUSIVAMENTE um JSON válido, sem texto extra, com as chaves "decision" e "transformed".
            Exemplo: {{"decision":"WEB","transformed":"Uso Denegado"}}
            """

            resp = llm.invoke(prompt)
            conteudo = getattr(resp, "content", "").strip()

            # ---- Parse robusto de JSON + gate por RAG ----
            import json, re

            # Tenta extrair apenas o objeto JSON se vier texto extra
            json_str = conteudo
            mjson = re.search(r"\{.*\}", conteudo, flags=re.S)
            if mjson:
                json_str = mjson.group(0)

            try:
                data = json.loads(json_str)
                dec = str(data.get("decision", "")).strip().upper()
                transformed = str(data.get("transformed", "")).strip()

                # Pós-validação: se a LLM mudou demais ou não corrigiu de forma útil, derive da própria base (top RAG)
                from difflib import SequenceMatcher
                sim_ratio = SequenceMatcher(None, transformed.lower(), pergunta.lower()).ratio()

                if sim_ratio < 0.70:
                    if resposta_rag:
                        meta = getattr(resposta_rag[0], "metadata", {}) or {}
                        titulo = (meta.get("titulo") or resposta_rag[0].page_content or "").strip()
                        # pega o rótulo canônico do doc (antes de “: …” se houver)
                        transformed = (titulo.split(":")[0].strip() or transformed)
                    else:
                        # fallback neutro e genérico, sem “ensinar” palavras
                        transformed = " ".join(tok.capitalize() for tok in pergunta.split())

                # Gate FINAL por confiança do RAG: evita falso positivo de domínio
                rag_conf = max(scores) if scores else 0.0
                if dec in ("RAG", "SQL", "WEB") and (not resposta_sql) and rag_conf < 0.25:
                    dec = "OUT"

                # Aplica decisão
                if dec == "RAG":
                    retorno_llm = "RAG"
                    codigos = [d.page_content for d in resposta_rag]
                elif dec == "SQL":
                    retorno_llm = "SQL"
                    codigos = [d.page_content for d in resposta_sql]
                elif dec == "OUT":
                    retorno_llm = "OUT"
                else:
                    retorno_llm = "WEB"
                    pergunta_web = transformed

                refined_question = transformed

            except Exception as e:
                print("❌ Erro ao interpretar JSON da refinadora:", e)
                # Se veio JSON bruto (começa com '{'), trate como OUT para evitar lixo
                if conteudo.lstrip().startswith("{"):
                    retorno_llm = "OUT"
                elif conteudo.startswith("RAG"):
                    retorno_llm = "RAG"
                    codigos = [d.page_content for d in resposta_rag]
                elif conteudo.startswith("SQL"):
                    retorno_llm = "SQL"
                    codigos = [d.page_content for d in resposta_sql]
                elif conteudo.startswith("OUT"):
                    retorno_llm = "OUT"
                else:
                    retorno_llm = "WEB"
                    pergunta_web = conteudo.replace("WEB:", "").strip()
                    refined_question = pergunta_web

        except Exception as e:
            print("❌ Erro na LLM refinadora:", e)
            retorno_llm = "WEB"
            pergunta_web = pergunta
            refined_question = pergunta

    else:
        retorno_llm = "WEB"
        pergunta_web = pergunta

    # 4) Retorno final
    # ✅ Se a pergunta foi classificada como fora do universo NFe
    if fora_de_contexto:
        return [], [], {
            "retorno_llm": "OUT",
            "pergunta_COD": None,
            "codigos": [],
            "pergunta_web": None,
            "refined_question": pergunta,
        }

    docs = resposta_sql + resposta_rag
    scores = [0.60] * len(resposta_sql) + scores[: len(resposta_rag)]

    return docs, scores, {
        "retorno_llm": retorno_llm,
        "pergunta_COD": pergunta_COD,
        "codigos": codigos,
        "pergunta_web": pergunta_web,
        "refined_question": refined_question,
    }
