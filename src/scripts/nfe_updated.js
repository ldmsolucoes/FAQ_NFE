  // Função para adicionar mensagens (mantida do pergunta_usuario.html)
  function adicionarMensagem(autor, texto) {
    const chat = document.getElementById('chat-messages');
    if (!chat) return; // Garante que o elemento existe
    const hora = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgDiv = document.createElement('div');
    msgDiv.innerHTML = `
      <div style="margin: 8px 0; padding: 10px;
                  background: ${autor === 'user' ? '#e3f2fd' : '#f0f0f0'};
                  border-radius: 8px;
                  ${autor === 'user' ? 'margin-left: 20%;' : 'margin-right: 20%;'}">
        <strong>${autor === 'user' ? 'Você:' : 'Assistente:'}</strong> ${texto}
        <div style="font-size: 0.8em; color: #666; text-align: right;">${hora}</div>
      </div>
    `;
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;
  }

  // Função para inicializar os listeners da página de pergunta do usuário
  function initializePerguntaUsuario() {
    const enviarBotao = document.getElementById('enviar-pergunta');
    if (enviarBotao) { // Verifica se o botão existe antes de adicionar o listener
      enviarBotao.addEventListener('click', async () => {
        const input = document.getElementById('pergunta-input');
        const pergunta = input.value.trim();
        if (!pergunta) return;

        adicionarMensagem('user', pergunta);
        input.value = '';

        try {
          const response = await fetch('/processar_pergunta', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pergunta })
          });
          const data = await response.json();
          adicionarMensagem('bot', data.resposta);
        } catch (error) {
          adicionarMensagem('bot', 'Erro ao processar pergunta');
          console.error(error);
        }
      });
    }
  }

  // NOVA FUNÇÃO: Inicializar listeners da página de geração da base
  function initializeGerarBase() {
    // Carregar status inicial
    async function carregarStatus() {
      try {
        const response = await fetch('/status_base');
        const data = await response.json();
        
        const statusDiv = document.getElementById('status-info');
        if (statusDiv) {
          if (data.status === 'sucesso') {
            statusDiv.innerHTML = `
              <strong>Base encontrada!</strong><br>
              Total de documentos: ${data.total_documents}<br>
              Coleção: ${data.collection_name}<br>
              Caminho: ${data.db_path}
            `;
            statusDiv.style.background = '#d4edda';
            statusDiv.style.color = '#155724';
          } else {
            statusDiv.innerHTML = `
              <strong>Base não encontrada</strong><br>
              ${data.message}<br>
              <em>Uma nova base será criada ao executar a atualização.</em>
            `;
            statusDiv.style.background = '#fff3cd';
            statusDiv.style.color = '#856404';
          }
        }
      } catch (error) {
        const statusDiv = document.getElementById('status-info');
        if (statusDiv) {
          statusDiv.innerHTML = 'Erro ao carregar status: ' + error.message;
        }
      }
    }

    // Gerar/atualizar base
    async function gerarBase() {
      const btn = document.getElementById('gerar-base-btn');
      const progressContainer = document.getElementById('progress-container');
      const resultContainer = document.getElementById('result-container');
      const progressBar = document.getElementById('progress-bar');
      const progressText = document.getElementById('progress-text');
      const resultContent = document.getElementById('result-content');
      
      if (!btn) return; // Verifica se o botão existe
      
      // Desabilitar botão e mostrar progresso
      btn.disabled = true;
      btn.textContent = 'Processando...';
      
      if (progressContainer) {
        progressContainer.style.display = 'block';
      }
      if (resultContainer) {
        resultContainer.style.display = 'none';
      }
      
      // Simular progresso
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) progress = 90;
        if (progressBar) {
          progressBar.style.width = progress + '%';
        }
        if (progressText) {
          progressText.textContent = `Processando... ${Math.round(progress)}%`;
        }
      }, 500);
      
      try {
        const response = await fetch('/gerar_base', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        
        const data = await response.json();
        
        // Finalizar progresso
        clearInterval(progressInterval);
        if (progressBar) {
          progressBar.style.width = '100%';
        }
        if (progressText) {
          progressText.textContent = 'Concluído!';
        }
        
        // Mostrar resultado
        if (resultContainer) {
          resultContainer.style.display = 'block';
        }
        if (resultContent) {
          resultContent.textContent = JSON.stringify(data, null, 2);
        }
        
        // Recarregar status
        setTimeout(carregarStatus, 1000);
        
      } catch (error) {
        clearInterval(progressInterval);
        if (resultContainer) {
          resultContainer.style.display = 'block';
        }
        if (resultContent) {
          resultContent.textContent = 'Erro: ' + error.message;
        }
      } finally {
        btn.disabled = false;
        btn.textContent = 'Gerar/Atualizar Base';
      }
    }

    // Event listeners
    const gerarBaseBtn = document.getElementById('gerar-base-btn');
    const verificarStatusBtn = document.getElementById('verificar-status-btn');
    
    if (gerarBaseBtn) {
      gerarBaseBtn.addEventListener('click', gerarBase);
    }
    if (verificarStatusBtn) {
      verificarStatusBtn.addEventListener('click', carregarStatus);
    }

    // Carregar status inicial
    carregarStatus();
  }

  // Função para carregar conteúdo de acordo com a opção selecionada usando switch/case
  async function carregarConteudo(opcao) {
    let url = '';
    switch(opcao) {
      case 'opcao1':
        url = '/gerar_base';
        break;
      case 'opcao2':
        url = '/consulta_nfss';
        break;
      case 'opcao3':
        url = '/pergunta_usuario';
        break;
      case 'opcao4':
        url = '/outra_rota_4';
        break;
      case 'opcao5':
        url = '/outra_rota_5';
        break;
      default:
        document.getElementById('content').innerHTML = '<p>Opção inválida.</p>';
        return;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Erro HTTP ' + response.status);
      const html = await response.text();
      document.getElementById('content').innerHTML = html;

      // Chamar a função de inicialização apropriada baseada na opção
      if (opcao === 'opcao1') {
        // NOVO: Inicializar funcionalidade da página de geração da base
        initializeGerarBase();
      } else if (opcao === 'opcao3') {
        // Inicializar funcionalidade da página de pergunta do usuário
        initializePerguntaUsuario();
      }

    } catch (error) {
      console.error('Erro ao carregar conteúdo:', error);
      document.getElementById('content').innerHTML = `<p>Erro ao carregar conteúdo: ${error.message}</p>`;
    }
  }

  // Vincular evento de clique em todos os links do menu com data-opcao
  document.querySelectorAll('nav.menu a[data-opcao]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      const opcao = event.currentTarget.getAttribute('data-opcao');
      carregarConteudo(opcao);
    });
  });

  // OPCIONAL: Carregar uma página inicial ao carregar o menu_nfe.html
  // carregarConteudo('opcao3'); // Exemplo: Carrega a página de pergunta automaticamente ao abrir o menu

