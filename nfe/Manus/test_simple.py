"""
Teste simples do sistema NFE
"""

from nfe_knowledge_base import NFEKnowledgeBase

def test_basic_functionality():
    """Teste básico da funcionalidade"""
    print("=== Teste Básico do Sistema NFE ===\n")
    
    # Inicializar base
    kb = NFEKnowledgeBase()
    
    # Verificar inicialização
    print("1. Testando inicialização...")
    success = kb.initialize_database()
    print(f"   Inicialização: {'✓ Sucesso' if success else '✗ Falha'}\n")
    
    # Verificar status
    print("2. Verificando status...")
    stats = kb.get_database_stats()
    print(f"   Status: {stats}\n")
    
    # Teste de busca simples (apenas 1 consulta)
    print("3. Testando busca simples...")
    try:
        result = kb.search_and_update_knowledge(["erro 539 NFe SEFAZ"])
        print(f"   Resultado: {result}\n")
        
        # Verificar se adicionou documentos
        if result.get("documents_added", 0) > 0:
            print("4. Testando consulta...")
            resultados = kb.query_knowledge_base("erro 539", n_results=1)
            print(f"   Encontrados {len(resultados)} resultados")
            if resultados:
                print(f"   Primeiro resultado: {resultados[0]['content'][:100]}...")
        
    except Exception as e:
        print(f"   Erro na busca: {e}")
    
    print("\n=== Teste concluído ===")

if __name__ == "__main__":
    test_basic_functionality()

