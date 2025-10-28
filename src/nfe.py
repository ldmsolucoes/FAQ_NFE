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






"""
Módulo para gerenciar a base de conhecimentos de erros NFE usando ChromaDB
"""

import os
import json
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import time

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFEKnowledgeBase:
    """Classe para gerenciar a base de conhecimentos de erros NFE"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "nfe_errors"):
        """
        Inicializa a base de conhecimentos
        
        Args:
            db_path: Caminho para o banco de dados ChromaDB
            collection_name: Nome da coleção no ChromaDB
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.perplexity_api_key = "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"
        
        # Configurar guardrails
        self.allowed_domains = [
            "sefaz.rs.gov.br",
            "nfe.fazenda.gov.br", 
            "receita.fazenda.gov.br",
            "gov.br",
            "fazenda.gov.br"
        ]
        
        self.max_requests_per_minute = 10
        self.request_count = 0
        self.last_request_time = 0
        
    def _check_rate_limit(self):
        """Implementa rate limiting para evitar sobrecarga"""
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            if self.request_count >= self.max_requests_per_minute:
                sleep_time = 60 - (current_time - self.last_request_time)
                logger.info(f"Rate limit atingido. Aguardando {sleep_time:.2f} segundos...")
                time.sleep(sleep_time)
                self.request_count = 0
        else:
            self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = current_time
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Verifica se o domínio está na lista de domínios permitidos"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except:
            return False
    
    def initialize_database(self) -> bool:
        """
        Inicializa ou conecta ao banco de dados ChromaDB
        
        Returns:
            bool: True se inicializado com sucesso, False caso contrário
        """
        try:
            # Criar diretório se não existir
            os.makedirs(self.db_path, exist_ok=True)
            
            # Inicializar cliente ChromaDB
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Criar ou obter coleção
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Coleção '{self.collection_name}' encontrada.")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Base de conhecimentos de erros NFE"}
                )
                logger.info(f"Nova coleção '{self.collection_name}' criada.")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            return False
    
    def _search_perplexity(self, query: str) -> Optional[str]:
        """
        Busca informações usando Perplexity AI
        
        Args:
            query: Consulta para buscar
            
        Returns:
            str: Resposta da busca ou None se houver erro
        """
        try:
            self._check_rate_limit()
            
            chat = ChatPerplexity(
                temperature=0.1,
                model="sonar-pro",
                pplx_api_key=self.perplexity_api_key
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Você é um especialista em NFe (Nota Fiscal Eletrônica) do Brasil. 
                Forneça informações precisas e detalhadas sobre erros de processamento de NFe, 
                baseando-se apenas em fontes oficiais como SEFAZ, Receita Federal e documentação oficial.
                Inclua sempre o código do erro quando disponível."""),
                ("human", "{input}")
            ])
            
            chain = prompt | chat
            response = chain.invoke({"input": query})
            
            return response.content
            
        except Exception as e:
            logger.error(f"Erro na busca Perplexity: {e}")
            return None
    
    def _extract_error_info(self, content: str) -> Dict:
        """
        Extrai informações estruturadas sobre erros NFE do conteúdo
        
        Args:
            content: Conteúdo a ser analisado
            
        Returns:
            Dict: Informações estruturadas do erro
        """
        # Usar Perplexity para estruturar as informações
        try:
            chat = ChatPerplexity(
                temperature=0,
                model="sonar-pro", 
                pplx_api_key=self.perplexity_api_key
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Analise o conteúdo fornecido e extraia informações estruturadas sobre erros de NFe.
                Retorne um JSON com os campos:
                - codigo_erro: código do erro (se disponível)
                - descricao: descrição do erro
                - solucao: solução ou orientação
                - fonte: fonte da informação
                
                Se não conseguir identificar um erro específico, retorne um JSON vazio {{}}."""),
                ("human", "Conteúdo: {content}")
            ])
            
            chain = prompt | chat
            response = chain.invoke({"content": content})
            
            # Tentar parsear JSON da resposta
            try:
                # Se a resposta for uma lista, pegar o primeiro item
                response_content = response.content
                if isinstance(response_content, list):
                    response_content = response_content[0] if response_content else "{}"
                
                return json.loads(response_content)
            except:
                # Se não conseguir parsear, retornar estrutura básica
                return {
                    "codigo_erro": "N/A",
                    "descricao": content[:200] + "..." if len(content) > 200 else content,
                    "solucao": "Consultar documentação oficial",
                    "fonte": "Busca automatizada"
                }
                
        except Exception as e:
            logger.error(f"Erro ao extrair informações: {e}")
            return {}
    
    def search_and_update_knowledge(self, error_queries: List[str] = None) -> Dict:
        """
        Busca e atualiza a base de conhecimentos com informações sobre erros NFE
        
        Args:
            error_queries: Lista de consultas específicas sobre erros
            
        Returns:
            Dict: Relatório da atualização
        """
        if not self.collection:
            if not self.initialize_database():
                return {"status": "erro", "message": "Falha ao inicializar banco de dados"}
        
        # Consultas padrão se não fornecidas
        if not error_queries:
            error_queries = [
                "erros comuns NFe SEFAZ processamento",
                "códigos erro NFe rejeição SEFAZ",
                "erro 539 NFe SEFAZ",
                "erro 565 NFe SEFAZ", 
                "erro 101 NFe SEFAZ",
                "erro 204 NFe SEFAZ",
                "erro 215 NFe SEFAZ",
                "erro 539 NFe duplicidade",
                "erro validação schema NFe",
                "erro certificado digital NFe"
            ]
        
        results = {
            "status": "sucesso",
            "total_queries": len(error_queries),
            "successful_searches": 0,
            "errors": [],
            "documents_added": 0
        }
        
        for i, query in enumerate(error_queries):
            try:
                logger.info(f"Processando consulta {i+1}/{len(error_queries)}: {query}")
                
                # Buscar informações
                content = self._search_perplexity(query)
                
                if content:
                    # Extrair informações estruturadas
                    error_info = self._extract_error_info(content)
                    
                    if error_info:
                        # Criar documento para ChromaDB
                        doc_id = f"nfe_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        
                        metadata = {
                            "codigo_erro": error_info.get("codigo_erro", "N/A"),
                            "fonte": error_info.get("fonte", "Perplexity AI"),
                            "data_pesquisa": datetime.now().isoformat(),
                            "query_original": query,
                            "tipo": "erro_nfe"
                        }
                        
                        # Adicionar à coleção
                        self.collection.add(
                            documents=[content],
                            metadatas=[metadata],
                            ids=[doc_id]
                        )
                        
                        results["documents_added"] += 1
                        results["successful_searches"] += 1
                        
                        logger.info(f"Documento adicionado: {doc_id}")
                    
                else:
                    results["errors"].append(f"Falha na busca para: {query}")
                
                # Pausa entre consultas para evitar rate limiting
                time.sleep(2)
                
            except Exception as e:
                error_msg = f"Erro ao processar '{query}': {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    def query_knowledge_base(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Consulta a base de conhecimentos
        
        Args:
            query: Consulta a ser realizada
            n_results: Número máximo de resultados
            
        Returns:
            List[Dict]: Lista de resultados encontrados
        """
        if not self.collection:
            if not self.initialize_database():
                return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance  # Converter distância em score de similaridade
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erro ao consultar base de conhecimentos: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """
        Retorna estatísticas da base de dados
        
        Returns:
            Dict: Estatísticas da base
        """
        if not self.collection:
            if not self.initialize_database():
                return {"status": "erro", "message": "Banco não inicializado"}
        
        try:
            count = self.collection.count()
            return {
                "status": "sucesso",
                "total_documents": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"status": "erro", "message": str(e)}


# Função principal para ser chamada pela opção 1 do menu
@router.post("/gerar_base_conhecimento")
def gerar_base_conhecimento() -> Dict:
    """
    Função principal para gerar/atualizar a base de conhecimentos
    Esta função será chamada pela opção 1 do menu
    
    Returns:
        Dict: Resultado da operação
    """
    try:
        logger.info("Iniciando geração/atualização da base de conhecimentos NFE...")
        
        # Inicializar base de conhecimentos
        kb = NFEKnowledgeBase()
        
        # Verificar se banco existe e obter estatísticas
        stats = kb.get_database_stats()
        
        if stats["status"] == "sucesso":
            logger.info(f"Base de dados encontrada com {stats['total_documents']} documentos")
        else:
            logger.info("Criando nova base de dados...")
        
        # Atualizar base de conhecimentos
        result = kb.search_and_update_knowledge()
        
        # Obter estatísticas finais
        final_stats = kb.get_database_stats()
        
        return {
            "status": "sucesso",
            "message": "Base de conhecimentos atualizada com sucesso",
            "detalhes": {
                "consultas_processadas": result["total_queries"],
                "buscas_bem_sucedidas": result["successful_searches"],
                "documentos_adicionados": result["documents_added"],
                "total_documentos_final": final_stats.get("total_documents", 0),
                "erros": result["errors"]
            }
        }
        
    except Exception as e:
        logger.error(f"Erro na geração da base de conhecimentos: {e}")
        return {
            "status": "erro",
            "message": f"Falha ao gerar base de conhecimentos: {str(e)}"
        }


if __name__ == "__main__":
    # Teste da funcionalidade
    resultado = gerar_base_conhecimento()
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

