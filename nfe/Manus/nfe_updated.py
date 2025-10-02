from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

# Importar nossa classe de base de conhecimentos
from nfe_knowledge_base import gerar_base_conhecimento, NFEKnowledgeBase

def inicializar_chat_pplx():
    chat = ChatPerplexity(
        temperature=0,
        model="sonar",
        pplx_api_key="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"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente fiscal preciso. Preciso da resposta com no máximo 5 linhas. Responda só assuntos pertinentes a nfe. Se a pergunta sair do escopo diga: 'Essa pergunta está fora do escopo NFe. Desculpe...'"),
        ("human", "{input}")
    ])
    
    return prompt | chat

router = APIRouter()

# Configuração do diretório das templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumindo que a pasta 'templates' está no mesmo nível do seu arquivo nfe.py
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@router.get("/nfe", response_class=HTMLResponse)
async def get_menu(request: Request):
    return templates.TemplateResponse("menu_nfe.html", {"request": request})

@router.get("/gerar_base", response_class=HTMLResponse)
async def gerar_base(request: Request):
    """Página para gerar/atualizar base de conhecimentos"""
    return templates.TemplateResponse("gerar_base.html", {"request": request})

@router.post("/gerar_base")
async def processar_gerar_base(request: Request):
    """
    Endpoint para processar a geração/atualização da base de conhecimentos
    Esta é a função chamada pela 'opcao1' do menu
    """
    try:
        # Executar a geração da base de conhecimentos em background
        # Para evitar timeout, podemos executar de forma assíncrona
        resultado = await asyncio.get_event_loop().run_in_executor(
            None, gerar_base_conhecimento
        )
        
        return JSONResponse(content=resultado)
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "erro",
                "message": f"Erro ao processar geração da base: {str(e)}"
            },
            status_code=500
        )

@router.get("/status_base")
async def get_status_base():
    """Endpoint para verificar status da base de conhecimentos"""
    try:
        kb = NFEKnowledgeBase()
        stats = kb.get_database_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(
            content={
                "status": "erro", 
                "message": str(e)
            },
            status_code=500
        )

@router.get("/consulta_nfss", response_class=HTMLResponse)
async def consulta_nfss(request: Request):
    return templates.TemplateResponse("consulta_nfss.html", {"request": request})

@router.get("/pergunta_usuario", response_class=HTMLResponse)
async def pergunta_usuario(request: Request):
    # Certifique-se de que pergunta_usuario.html está dentro da pasta 'templates'
    return templates.TemplateResponse("pergunta_usuario.html", {"request": request})

@router.post("/processar_pergunta")
async def processar_pergunta(request: Request):
    """Processar pergunta do usuário com consulta à base de conhecimentos"""
    try:
        data = await request.json()
        pergunta_do_usuario = data.get("pergunta")

        if not pergunta_do_usuario:
            return {"resposta": "Não entendi sua pergunta. Por favor, tente novamente."}

        # Primeiro, tentar consultar a base de conhecimentos local
        kb = NFEKnowledgeBase()
        resultados_kb = kb.query_knowledge_base(pergunta_do_usuario, n_results=3)
        
        resposta_final = ""
        
        if resultados_kb:
            # Se encontrou resultados na base local, usar como contexto
            contexto = "\n".join([r["content"][:500] for r in resultados_kb[:2]])
            
            # Usar Perplexity com contexto da base local
            chat = ChatPerplexity(
                temperature=0,
                model="sonar",
                pplx_api_key="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"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""Você é um assistente fiscal especializado em NFe. 
                Use o contexto fornecido da base de conhecimentos para responder à pergunta.
                Responda com no máximo 5 linhas e seja preciso.
                Se a pergunta não for sobre NFe, diga: 'Essa pergunta está fora do escopo NFe. Desculpe...'
                
                Contexto da base de conhecimentos:
                {contexto}"""),
                ("human", "{input}")
            ])
            
            chain = prompt | chat
            response = chain.invoke({"input": pergunta_do_usuario})
            resposta_final = response.content
            
        else:
            # Se não encontrou na base local, usar apenas Perplexity
            chain = inicializar_chat_pplx()
            response = chain.invoke({"input": pergunta_do_usuario})
            resposta_final = response.content
        
        return {"resposta": resposta_final}
        
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
        return {"resposta": "Desculpe, ocorreu um erro ao processar sua solicitação."}

@router.post("/consultar_base")
async def consultar_base(request: Request):
    """Endpoint para consultar diretamente a base de conhecimentos"""
    try:
        data = await request.json()
        query = data.get("query", "")
        n_results = data.get("n_results", 5)
        
        if not query:
            return JSONResponse(
                content={"status": "erro", "message": "Query não fornecida"},
                status_code=400
            )
        
        kb = NFEKnowledgeBase()
        resultados = kb.query_knowledge_base(query, n_results)
        
        return JSONResponse(content={
            "status": "sucesso",
            "query": query,
            "total_results": len(resultados),
            "results": resultados
        })
        
    except Exception as e:
        return JSONResponse(
            content={
                "status": "erro",
                "message": str(e)
            },
            status_code=500
        )

