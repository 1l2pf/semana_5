# Sistema de Perguntas e Respostas com LangChain e Claude

Este projeto implementa um sistema de perguntas e respostas (QA) utilizando a biblioteca LangChain integrada com os modelos de linguagem Claude da Anthropic. O sistema carrega documentos de texto salvo na memória (para economizar a instalação de novas bibliotecas excedendo o tamanho do programa, o que inviabilizaria seu uso no github), ademais, cria embeddings e permite ao usuário fazer perguntas sobre o conteúdo dos documento inserido.

## Visão Geral

O sistema:
1. Carrega documentos de texto
2. Divide os documentos em chunks menores
3. Cria embeddings (representações vetoriais) dos chunks
4. Armazena os embeddings em um índice FAISS
5. Usa o modelo Claude da Anthropic para gerar respostas com base nas informações recuperadas

## Pré-requisitos

- Python 3.10 ou superior
- Pip (gerenciador de pacotes Python)
- Chave de API da Anthropic

## Instalação

### 1. Clone o repositório ou baixe os arquivos

```bash
git clone https://github.com/seu-usuario/qa-system.git
cd qa-system
```

### 2. Crie e ative um ambiente virtual

#### No Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### No macOS e Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## Configuração da API da Anthropic

1. Crie um arquivo `.env` na raiz do projeto
2. Adicione sua chave API ao arquivo:
   ```
   ANTHROPIC_API_KEY=sua_chave_api_aqui
   ```
## Como usar

### Executando o script

```bash
python main.py
```

Por padrão, o script utiliza o arquivo `dados.txt` como fonte de dados. Você pode especificar um arquivo diferente quando solicitado durante a execução.

### Fazendo perguntas

Após iniciar o script:

1. Digite sua pergunta quando solicitado
2. Receba a resposta baseada no conteúdo do documento
3. Continue fazendo mais perguntas
4. Digite 'sair' para encerrar o programa


## Dependências principais

- langchain - Framework para construção de aplicações com LLMs
- langchain-community - Componentes da comunidade para LangChain
- faiss-cpu - Biblioteca para busca de similaridade vetorial
- anthropic - Cliente Python oficial da Anthropic
- python-dotenv - Para carregar variáveis de ambiente do arquivo .env

