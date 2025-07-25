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
      if (opcao === 'opcao3') {
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
