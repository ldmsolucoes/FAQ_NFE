# pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

# Cabeçalhos HTTP simulando navegador real para evitar bloqueios
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate, br",
}

def buscar_links_relevantes(base_url, texto_filtro):
    """
    Busca links na página base_url cujo texto contenha texto_filtro.
    Retorna lista de URLs completas.
    """
    try:
        resp = requests.get(base_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)
        urls_filtradas = []
        for link in links:
            txt = link.get_text(strip=True).lower()
            if texto_filtro in txt:
                href = link['href']
                url_completa = urljoin(base_url, href)
                urls_filtradas.append(url_completa)
        # Remove URLs duplicadas
        return list(set(urls_filtradas))
    except Exception as e:
        print(f"Erro ao buscar links na {base_url}: {e}")
        return []

def extrair_tabela_codigo_descricao(url):
    """
    Tenta extrair uma tabela com colunas Código e Descrição (e opcionalmente Solução) da página.
    Retorna dicionário {codigo: {"descricao": ..., "solucao": ...}}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        tabelas = soup.find_all("table")
        for tabela in tabelas:
            # Tenta capturar cabeçalho
            cabecalho = [th.get_text(strip=True).lower() for th in tabela.find_all("th")]
            if not cabecalho:
                primeira_linha = tabela.find("tr")
                if primeira_linha:
                    cabecalho = [td.get_text(strip=True).lower() for td in primeira_linha.find_all("td")]

            # Verifica se tem coluna código e descrição
            tem_codigo = any("código" in h or "codigo" in h for h in cabecalho)
            tem_descricao = any("descrição" in h or "descricao" in h for h in cabecalho)

            if tem_codigo and tem_descricao:
                erros = {}
                linhas = tabela.find_all("tr")
                for linha in linhas[1:]:  # pula o cabeçalho
                    colunas = linha.find_all("td")
                    if len(colunas) >= 2:
                        codigo = colunas[0].get_text(strip=True)
                        descricao = colunas[1].get_text(strip=True)
                        solucao = colunas[2].get_text(strip=True) if len(colunas) >= 3 else ""
                        erros[codigo] = {
                            "descricao": descricao,
                            "solucao": solucao
                        }
                if erros:
                    print(f"Tabela extraída com sucesso de: {url} ({len(erros)} códigos)")
                    return erros
        return {}
    except Exception as e:
        print(f"Erro ao extrair tabela de {url}: {e}")
        return {}

def salvar_em_json(data, caminho):
    """Salva o dicionário em arquivo JSON com formatação."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Dados salvos em {caminho}")

def carregar_de_json(caminho):
    """Carrega JSON do arquivo, se existir, senão retorna dicionário vazio."""
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Arquivo {caminho} não encontrado.")
        return {}

def crawler_local_simples():
    """
    Realiza crawling básico nos sites da SEFAZ para extrair tabelas de códigos de rejeição.
    Retorna dicionário com todos os códigos extraídos únicos.
    """
    sites_oficiais = [
        "https://www.sefaz.rs.gov.br",
        "https://www.nfe.fazenda.gov.br",
    ]

    palavras_chave_links = ["tabela", "rejeição", "rejeicao", "códigos", "codigo", "erros"]
    codigos_geral = {}

    for site in sites_oficiais:
        print(f"Buscando em site: {site}")
        links_possiveis = []
        for palavra in palavras_chave_links:
            encontrados = buscar_links_relevantes(site, palavra)
            print(f"  Links com '{palavra}': {len(encontrados)}")
            links_possiveis.extend(encontrados)

        # Remove duplicados cruzando tudo
        links_possiveis = list(set(links_possiveis))

        for link in links_possiveis:
            print(f"  Tentando extrair tabela em {link}")
            tabela = extrair_tabela_codigo_descricao(link)
            if tabela:
                codigos_geral.update(tabela)

            # Pequeno delay entre requisições para não sobrecarregar o servidor e parecer "humano"
            time.sleep(1)

    print(f"Total de códigos únicos extraídos: {len(codigos_geral)}")
    return codigos_geral

def consultar_codigo(dicionario, codigo):
    """
    Consulta o dicionário de erros pelo código informado.
    Retorna descrição e solução se conhecido, senão retorna aviso.
    """
    info = dicionario.get(str(codigo))
    if info:
        return f"Código: {codigo}\nDescrição: {info['descricao']}\nSolução: {info['solucao'] or 'Não informada'}"
    return f"Código {codigo} não encontrado na base."

if __name__ == "__main__":
    # Rodar crawler e salvar resultado
    dados = crawler_local_simples()
    if dados:
        salvar_em_json(dados, "erros_sefaz_nfe.json")
    else:
        print("Nenhum dado extraído. Tentando carregar dados salvos localmente...")
        dados = carregar_de_json("erros_sefaz_nfe.json")

    # Exemplo simples de consulta
    print("\nConsulta exemplo para código 215:")
    print(consultar_codigo(dados, 215))
