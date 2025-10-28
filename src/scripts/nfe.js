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
        <strong>${autor === 'user' ? 'Você:' : ''}</strong> ${texto}
        <div style="font-size: 0.8em; color: #666; text-align: right;">${hora}</div>
      </div>
    `;
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;
  }

  // Função para inicializar os listeners da página de pergunta do usuário
  async function initializePerguntaUsuario() {
    const enviarBotao = document.getElementById('enviar-pergunta');
    if (enviarBotao) { // Verifica se o botão existe antes de adicionar o listener
      enviarBotao.addEventListener('click', async () => {
        const input = document.getElementById('pergunta-input');
        const pergunta = input.value.trim();
        if (!pergunta) return;

        adicionarMensagem('user', pergunta);
        input.value = '';

        try {
          //alert('per...');return;
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

// Inicializa o botão "Atualizar Base de Conhecimentos"
// Executa a atualização da base imediatamente ao clicar no botão (usado com onclick)
async function initializeGerarBase() {
    const btn = document.getElementById('gerar-base-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');

    const setUI = (state, msg) => {
        if (btn) {
            btn.disabled = (state === 'running');
            if (!btn.dataset._label) btn.dataset._label = btn.textContent;
            btn.textContent = (state === 'running') ? 'Atualizando...' : (btn.dataset._label || 'Atualizar Base de Conhecimentos');
        }
        if (progressContainer) progressContainer.style.display = (state === 'idle') ? 'none' : 'block';
        if (progressText) progressText.textContent = msg || '';
        if (progressBar) {
            progressBar.style.width = (state === 'running') ? '50%' : (state === 'success' || state === 'error') ? '100%' : '0%';
        }
    };

    // início
    setUI('running', 'Iniciando...');

    try {
        const res = await fetch('/gerar_base_conhecimento', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        let data = {};
        try { data = await res.json(); } catch { }

        const statusStr = ((data.status || '') + '').toLowerCase();
        const ok = res.ok && (!statusStr || statusStr === 'sucesso');

        if (!ok) throw new Error(data.message || data.mensagem || `Falha (HTTP ${res.status})`);

        setUI('success', data.mensagem || data.message || 'Base de conhecimentos atualizada com sucesso.');
        if (resultContainer && resultContent) {
            resultContainer.style.display = 'block';
            resultContent.textContent = JSON.stringify(data, null, 2);
        } else {
            // fallback visual se você não tiver os contêineres no HTML
            alert(data.mensagem || data.message || 'Base de conhecimentos atualizada com sucesso.');
        }
    } catch (err) {
        console.error('Erro ao atualizar base:', err);
        setUI('error', 'Falha ao atualizar a base.');
        if (resultContainer && resultContent) {
            resultContainer.style.display = 'block';
            resultContent.textContent = (err && err.message) ? err.message : String(err);
        } else {
            alert('Falha ao atualizar a base: ' + ((err && err.message) ? err.message : String(err)));
        }
    } finally {
        setUI('idle', '');
    }
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
      // NOVO: Chamar a função de inicialização se a página carregada for pergunta_usuario

      if (opcao === 'opcao1') {
        await initializeGerarBase();
      }

      if (opcao === 'opcao3') {
        await initializePerguntaUsuario();
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
  //carregarConteudo('opcao1'); // Exemplo: Carrega a página de pergunta automaticamente ao abrir o menu


//Quick Replies- Açoes de mensagens rápidas
/**
 * Função para a ação 'Atualizar a base de conhecimentos'
 * Exibe um alert de confirmação (Sim/Não).
 */
function handleUpdateBase() {
    const confirmation = confirm('Atenção, essa ação irá remover a base de conhecimento atual. Confirma essa ação?(S/N)');

    if (confirmation) {
        alert('Ação de atualização confirmada. (Implemente a lógica Python/Backend aqui.)');
        // Futuramente, você adicionará a chamada AJAX para o backend Python aqui.
    } else {
        alert('Ação de atualização cancelada.');
    }
}

/**
 * Função para a ação 'Ferramentas'
 * Exibe um alert simples.
 */
function handleTools() {
    alert('Caixa de ferramentas');
}

/**
 * Função para a ação 'Sair'
 * Tenta fechar a janela do navegador.
 */
function handleExit() {
    if (window.opener) {
        // Se foi aberta por outra janela (ex: um popup), fecha
        window.close();
    } else {
        // Caso contrário, informa o usuário (navegadores modernos limitam o fechamento)
        alert('Para fechar, use o botão de fechar do seu navegador.');
    }
}
