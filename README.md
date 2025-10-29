# FAQ_NFE
Projeto final do curso de agentes de IA.
Ministrado pelo professor Celso Azevedo do i2a2 academy.

O projeto se encontra sob a licença MIT.

O link de acesso para testes está no droplet da LDM Soluções, e será desativado logo após a avaliação do i2a2.

Para iniciar o sistema no ubuntu:
1. Coloque a sua chave no seu ambiente: export OPENAI_API_KEY='SUA CHAVE AQUI'
2. Crie uma pasta 'nfe', por exemplo e coloque todos os arquivos e sub-diretórios contidos na pasta 'src'
3. Crie o ambiente com base no arquivo 'requirements_atual.txt'
4. Ative o uvicorn: nohup uvicorn nfe.main:app --host 127.0.0.1 --port 8000 > log_agent.txt 2>&1 &
5. Rode a aplicação no seu browser: http://146.190.170.18/nfe

Caso haja erro ao baixar qualquer arquivo da pasta "Projeto Final - Artefatos", eles podem ser encontrados no link a seguir: https://drive.google.com/drive/folders/1Mub6sKzqRuhXzqmrKntFpxnnGF8dKk_w?usp=drive_link

A LDM Soluções se reserva no direito de desativar a url abaixo a qualquer momento.
