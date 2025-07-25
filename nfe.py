from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime # Importe datetime para usar a data/hora

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

def inicializar_chat_pplx():
    chat = ChatPerplexity(
        temperature=0,
        model="sonar",
        pplx_api_key="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"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente fiscal preciso. Preciso da resposta com no máximo 5 linhas. Responda só assuntos pertinentes a nfe."),
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
    return templates.TemplateResponse("gerar_base.html", {"request": request})

@router.get("/consulta_nfss", response_class=HTMLResponse)
async def consulta_nfss(request: Request):
    return templates.TemplateResponse("consulta_nfss.html", {"request": request})

@router.get("/pergunta_usuario", response_class=HTMLResponse)
async def pergunta_usuario(request: Request):
    # Certifique-se de que pergunta_usuario.html está dentro da pasta 'templates'
    return templates.TemplateResponse("pergunta_usuario.html", {"request": request})

# NOVO: Adicione esta rota POST para processar a pergunta
@router.post("/processar_pergunta")
async def processar_pergunta(request: Request):
    try:
        data = await request.json()
        pergunta_do_usuario = data.get("pergunta")

        if not pergunta_do_usuario:
            return {"resposta": "Não entendi sua pergunta. Por favor, tente novamente."}

        # Inicializa o chain do LangChain
        chain = inicializar_chat_pplx()
        
        # Invoca o chain com a pergunta do usuário
        response = chain.invoke({"input": pergunta_do_usuario})
        
        # Retorna a resposta do assistente
        return {"resposta": response.content}
        
    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
        return {"resposta": "Desculpe, ocorreu um erro ao processar sua solicitação."}