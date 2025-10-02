#!/usr/bin/env python3
# agent_nfe.py — Agente único LangGraph para NFE (RAG + fallback)
from __future__ import annotations

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import uuid
import sqlite3

import os
import time

import requests
import tempfile
import fitz  # PyMuPDF
import pandas as pd

from .agent_pesquisa import ensure_vectorstore


from typing import List, Dict, Any, Tuple, TypedDict, Literal

# ===== FastAPI (mantém compat com seu app atual) =====
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ===== LangChain / LangGraph =====
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# Vetorstore (Chroma via LangChain)
from langchain_chroma import Chroma

# ===== Embeddings / LLMs =====
EMBEDDING_FAMILY = os.getenv("EMBEDDING_FAMILY", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAG_MODEL = os.getenv("RAG_MODEL", "gpt-3.5-turbo")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
SIM_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.76"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("COLLECTION", "nfe_errors")

# Tenta OpenAI; se indisponível, usa DummyLLM
_llm_rag = None
_llm_fallback = None
_embeddings = None

class DummyResponse:
    def __init__(self, content: str):
        self.content = content

class DummyLLM:
    def invoke(self, messages):
        try:
            user = messages[-1].get("content", "")
        except Exception:
            user = str(messages)
        return DummyResponse("(resposta simulada) " + user[:500])

try:
    if EMBEDDING_FAMILY == "openai":
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        _embeddings = OpenAIEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError("Nenhuma implementação de embeddings disponível.") from e

try:
    from langchain_openai import ChatOpenAI
    _llm_rag = ChatOpenAI(model=RAG_MODEL, temperature=0)
    _llm_fallback = ChatOpenAI(model=FALLBACK_MODEL, temperature=0)
except Exception:
    _llm_rag = DummyLLM()
    _llm_fallback = DummyLLM()

# ===== Templates =====
router = APIRouter()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ===== Estado do Agente =====
class AgentState(TypedDict, total=False):
    task: Literal["carregar_base", "responder_pergunta"]
    pergunta: str
    docs: List[Document]
    scores: List[float]
    rag_answer: str
    final_answer: str
    confidence: float
    error: str

# ===== Utilidades =====
def ensure_vectorstore() -> Chroma:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION,
        embedding_function=_embeddings,
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", ".", " "]
)

# ===== Fontes básicas =====
def coletar_fontes_basicas() -> List[Dict[str, str]]:
    return [
        {
            "url": "https://www.nfe.fazenda.gov.br/",
            "codigo": "215",
            "descricao": "Falha na validação do XML",
            "solucao": "Verifique se todas as tags obrigatórias estão presentes, se a assinatura digital é válida e se o XML segue o schema oficial."
        },
        {
            "url": "https://www.nfe.fazenda.gov.br/",
            "codigo": "539",
            "descricao": "Duplicidade de NF-e",
            "solucao": "Esse erro indica que uma NF-e já foi autorizada com a mesma chave de acesso. Confira se a NF-e não foi transmitida duas vezes."
        },
        {
            "url": "https://www.nfe.fazenda.gov.br/",
            "codigo": "565",
            "descricao": "Falha de schema",
            "solucao": "O XML não está em conformidade com a versão autorizada do schema. Atualize o layout da NF-e e valide novamente."
        },
        {
            "url": "https://www.gov.br/",
            "codigo": "215",
            "descricao": "Falha na validação do XML",
            "solucao": "O erro 215 pode ocorrer por tag ausente, assinatura inválida ou divergência no schema utilizado."
        },
    ]
# ===== Configuração de fontes externas =====
_PDF_CONFIG = {
  "nfe_erros": {
    "link": "https://www.confaz.fazenda.gov.br/legislacao/arquivo-manuais/moc7-anexo-i-leiaute-e-rv.pdf",
    "secoes": [
      {
        "secao_inicial": "4.4.1. Tabela de Códigos de Resultado de Processamento (cStat)",
        "tipo": "tabela",
        "campos": ["CÓDIGO", "RESULTADO DO PROCESSAMENTO DA SOLICITAÇÃO"]
      },
      {
        "secao_inicial": "4.4.2. Tabela de Códigos de Rejeição",
        "tipo": "tabela",
        "campos": ["CÓD", "MOTIVOS DE NÃO ATENDIMENTO DA SOLICITAÇÃO"]
      }
    ]
  }
}



# ===== Nó: carregar_base =====
# ===== Configuração de fontes externas =====
PDF_CONFIG = {
  "nfe_erros": {
    "link": "https://www.confaz.fazenda.gov.br/legislacao/arquivo-manuais/moc7-anexo-i-leiaute-e-rv.pdf",
    "secoes": [
      {
        "secao_inicial": "CÓDIGO",
        "tipo": "tabela",
        "campos": ["CÓDIGO", "RESULTADO DO PROCESSAMENTO DA SOLICITAÇÃO"]
      },
      {
        "secao_inicial": "CÓD",
        "tipo": "tabela",
        "campos": ["CÓD", "MOTIVOS DE NÃO ATENDIMENTO DA SOLICITAÇÃO"]
      }
    ]
  }
}



# ===== Nó: carregar_base =====
def node_carregar_base(state: Dict[str, Any]) -> Dict[str, Any]:
    import shutil

    try:
        # --- Remover DB e Chroma antigos ---
        if os.path.exists("nfe.db"):
            os.remove("nfe.db")
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        # --- Baixar PDF ---
        pdf_url = PDF_CONFIG["nfe_erros"]["link"]
        r = requests.get(pdf_url, timeout=60, verify=False)
        r.raise_for_status()
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.write(r.content)
        tmp_pdf.close()

        # --- Abrir PDF ---
        doc = fitz.open(tmp_pdf.name)
        #print("[DEBUG] Páginas do PDF:", len(doc))

        texts, metadatas, ids = [], [], []

        # --- Extrair dados de cada seção ---
        for secao in PDF_CONFIG["nfe_erros"]["secoes"]:
            cols = secao["campos"]
            secao_nome = secao["secao_inicial"]

            for page in doc:
                text = page.get_text("text")
                if secao_nome in text:
                    lines = text.splitlines()
                    for i, line in enumerate(lines):
                        if line.strip().isdigit():
                            codigo = line.strip()
                            if i + 1 < len(lines):
                                titulo = lines[i + 1].strip()
                                doc_text = f"{codigo} — {titulo}"
                                texts.append(doc_text)
                                metadatas.append({
                                    "codigo": codigo,
                                    "titulo": titulo,
                                    "source": pdf_url,
                                    "secao": secao_nome
                                })
                                ids.append(f"nfe_{codigo}_{uuid.uuid4().hex}")
        doc.close()

        # --- Gravar no SQLite ---
        conn = sqlite3.connect("nfe.db")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ErrosNFE (
                codigo TEXT,
                titulo TEXT,
                secao TEXT,
                fonte TEXT
            )
        """)
        cur.executemany(
            "INSERT INTO ErrosNFE (codigo, titulo, secao, fonte) VALUES (?, ?, ?, ?)",
            [(m["codigo"], m["titulo"], m["secao"], m["source"]) for m in metadatas]
        )
        conn.commit()
        conn.close()

        # --- Gravar no Chroma ---
        vs = ensure_vectorstore()
        if texts:
            vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        os.remove(tmp_pdf.name)

        return {**state, "final_answer": f"Base de conhecimentos atualizada com {len(texts)} registros."}

    except Exception as e:
        return {**state, "final_answer": f"Erro ao carregar base: {str(e)}"}



# ===== Busca RAG =====
def rag_search(query: str, k: int = 4) -> Tuple[List[Document], List[float]]:
    vs = ensure_vectorstore()
    results = vs.similarity_search_with_score(query, k=k)
    docs, scores = [], []
    for doc, dist in results:
        sim = 1.0 - min(max(dist, 0.0), 1.0)
        docs.append(doc)
        scores.append(sim)
    return docs, scores

# ===== Nó: consultar_base_rag =====
from .agent_pesquisa import pesquisa_rag  # importa do novo agente

def node_consultar_base_rag(state: AgentState) -> AgentState:
    pergunta = state.get("pergunta", "")
    docs, scores = pesquisa_rag(pergunta)
    return {**state, "docs": docs, "scores": scores}


# ===== Roteamento por confiança =====
def route_confidence(state: AgentState) -> Literal["OK", "FALLBACK"]:
    if state.get("docs"):  # se houver qualquer doc, vai para RAG
        return "OK"
    return "FALLBACK"

# ===== Nó: gerar_resposta_rag =====
import re

import re

def node_gerar_resposta_rag(state: AgentState) -> AgentState:
    """
    Não altere imports, rotas, nomes de funções, variáveis globais nem assinaturas.
    Não adicione dependências.
    Não modifique nenhuma outra função.
    """

    docs = state.get("docs", [])
    scores = state.get("scores", [])

    if not docs:
        return {
            **state,
            "rag_answer": "Não encontrei informações sobre esse erro na base de conhecimentos.",
            "final_answer": "Não encontrei informações sobre esse erro na base de conhecimentos.",
            "confidence": 0.0,
        }

    # Pegar apenas a primeira ocorrência
    primeiro_doc = docs[0]
    codigo = primeiro_doc.metadata.get("codigo", "N/D")
    descricao = (
        primeiro_doc.metadata.get("titulo")
        or primeiro_doc.metadata.get("descricao")
        or "Sem descrição"
    )

    # Prompt para a LLM
    prompt = [
        {
            "role": "system",
            "content": (
                "Você é um especialista em Nota Fiscal Eletrônica (NFe).\n"
                "Use apenas informações de fontes oficiais (como nfe.fazenda.gov.br, confaz.fazenda.gov.br) "
                "e fontes confiáveis reconhecidas sobre NFe.\n"
                "Sua tarefa é analisar o código de erro informado e gerar até 3 possíveis explicações/soluções.\n\n"
                "Regras:\n"
                "- Cada resposta deve ser breve (até 6 linhas), fundamentada, e mencionar de forma curta a fonte utilizada.\n"
                "- Nunca invente links nem informações não verificáveis.\n"
                "- Ordene as respostas da mais provável para a menos provável.\n"
                "- Numere as respostas (1, 2, 3).\n"
                "- Separe cada resposta com uma linha em branco."
            ),
        },
        {
            "role": "user",
            "content": f"Código {codigo} — {descricao}\n\nListe até 3 respostas possíveis para esse erro, em ordem de probabilidade."
        }
    ]

    try:
        resp = _llm_rag.invoke(prompt)
        raw_answer = getattr(resp, "content", str(resp)).strip()

        # Normalizar quebras de linha
        raw_answer = raw_answer.replace("\\n", "\n")

        # Só dividir quando for "1.", "2.", "3." no início ou após quebra de linha
        partes = re.split(r"(?:^|\n)(?=[123]\.)", raw_answer)
        partes = [p.strip() for p in partes if p.strip()]

        # Rejuntar com espaçamento HTML
        answer = "<br><br>".join(partes)

    except Exception as e:
        answer = f"Erro ao consultar LLM: {e}"

    conf = max(scores or [0.0])
    return {
        **state,
        "rag_answer": answer,
        "final_answer": answer,
        "confidence": conf,
    }



# ===== Nó: fallback =====
SYSTEM_FALLBACK = (
    "Você é um assistente especializado em Nota Fiscal Eletrônica (NFe). "
    "Se a pergunta não for sobre NFe, responda apenas: "
    "'Este assistente responde apenas perguntas sobre Nota Fiscal Eletrônica (NFe)'. "
    "Se a pergunta for sobre NFe, mas não houver dados na base, "
    "responda de forma breve e genérica (até 5 linhas) sem inventar detalhes."
)

def node_fallback_llm(state: AgentState) -> AgentState:
    pergunta = state.get("pergunta", "")
    prompt = [
        {"role": "system", "content": SYSTEM_FALLBACK},
        {"role": "user", "content": pergunta},
    ]
    resp = _llm_fallback.invoke(prompt)
    answer = getattr(resp, "content", str(resp))
    return {**state, "final_answer": answer, "confidence": 0.0}

def node_fallback_llm(state: AgentState) -> AgentState:
    pergunta = state.get("pergunta", "")
    prompt = [
        {"role": "system", "content": SYSTEM_FALLBACK},
        {"role": "user", "content": pergunta},
    ]
    resp = _llm_fallback.invoke(prompt)
    answer = getattr(resp, "content", str(resp))
    return {**state, "final_answer": answer, "confidence": 0.0}


# ===== Construção do GRAFO (on-demand) =====
def build_graph(entry_point: str) -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("carregar_base", node_carregar_base)
    g.add_node("consultar_base_rag", node_consultar_base_rag)
    g.add_node("gerar_resposta_rag", node_gerar_resposta_rag)
    g.add_node("fallback_llm", node_fallback_llm)

    g.add_conditional_edges(
        "consultar_base_rag", route_confidence,
        {"OK": "gerar_resposta_rag", "FALLBACK": "fallback_llm"}
    )
    g.add_edge("gerar_resposta_rag", END)
    g.add_edge("fallback_llm", END)

    g.set_entry_point(entry_point)
    if entry_point == "carregar_base":
        g.add_edge("carregar_base", END)

    return g

def get_graph_for_base():
    return build_graph("carregar_base").compile()

def get_graph_for_answer():
    return build_graph("consultar_base_rag").compile()

# ===== Facades do agente =====
def agente_carregar_base() -> Dict[str, Any]:
    graph = get_graph_for_base()
    out = graph.invoke({"task": "carregar_base"})
    return {"status": "sucesso", "mensagem": out.get("final_answer", "Base atualizada.")}

def agente_responder_pergunta(pergunta: str) -> Dict[str, Any]:
    graph = get_graph_for_answer()
    st = graph.invoke({"task": "responder_pergunta", "pergunta": pergunta})
    return {"resposta": st.get("final_answer", ""), "confidence": st.get("confidence", 0.0)}

# ===== Rotas FastAPI =====
@router.get("/nfe", response_class=HTMLResponse)
async def get_menu(request: Request):
    return templates.TemplateResponse("menu_nfe.html", {"request": request})

@router.get("/gerar_base", response_class=HTMLResponse)
async def gerar_base_page(request: Request):
    return templates.TemplateResponse("gerar_base.html", {"request": request})

@router.get("/consulta_nfss", response_class=HTMLResponse)
async def consulta_nfss_page(request: Request):
    return templates.TemplateResponse("consulta_nfss.html", {"request": request})

@router.get("/pergunta_usuario", response_class=HTMLResponse)
async def pergunta_usuario_page(request: Request):
    return templates.TemplateResponse("pergunta_usuario.html", {"request": request})

@router.post("/gerar_base_conhecimento")
async def gerar_base_conhecimento():
    try:
        return agente_carregar_base()
    except Exception as e:
        return {"status": "erro", "message": str(e)}

@router.post("/processar_pergunta")
async def processar_pergunta(request: Request):
    try:
        payload = await request.json()
        pergunta = (payload or {}).get("pergunta", "").strip()
        if not pergunta:
            return {"resposta": "Não entendi sua pergunta. Por favor, tente novamente."}
        res = agente_responder_pergunta(pergunta)
        return {"resposta": res.get("resposta", ""), "confidence": res.get("confidence", 0.0)}
    except Exception as e:
        return {"resposta": f"Erro ao processar: {e}"}

# ===== Execução direta =====
if __name__ == "__main__":
    print(agente_carregar_base())
    print(agente_responder_pergunta("O que significa o erro 215 na NFe?"))
