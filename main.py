import os
import requests
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA


try:
    from dotenv import load_dotenv
    load_dotenv()  
except ImportError:
    pass 

# Função para obter a chave da API de forma segura
def get_api_key():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Aviso: Variável de ambiente ANTHROPIC_API_KEY não encontrada.")
        api_key = input("Por favor, digite sua chave API da Anthropic: ")
    return api_key


def get_fake_embeddings():
    return FakeEmbeddings(size=384)

# Função para construir o sistema de perguntas e respostas
def build_qa_system(text_file, anthropic_api_key):
    # Carregando e dividindo o texto
    loader = TextLoader(text_file, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,  
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Usando FakeEmbeddings da biblioteca langchain_community
    embeddings = get_fake_embeddings()
    
    # Criando índice FAISS comprimido
    vectordb = FAISS.from_documents(chunks, embeddings)
    

    vectordb.save_local("faiss_index_compressed")
    
    # Configurando LLM e sistema de respostas usando a chave API Anthropic

    llm = ChatAnthropic(model="claude-3-5-sonnet-latest", api_key=anthropic_api_key)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2}) 
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False 
    )
    
    return qa_chain

# Função principal
def main():

    api_key = get_api_key()
    
    # Verificar se foi fornecida uma chave
    if not api_key:
        print("Erro: Não foi possível obter a chave API da Anthropic.")
        return
    
    # Solicitar o arquivo de dados se necessário
    text_file = input("Digite o caminho para o arquivo de texto (ou pressione Enter para usar 'dados.txt'): ") or "dados.txt"
    
    # Construção do sistema QA
    try:
        qa_system = build_qa_system(text_file, api_key)
        
        print(f"\nSistema de QA iniciado com dados de '{text_file}'. Digite 'sair' para encerrar.")
        while True:
            pergunta = input("\nSua pergunta: ")
            if pergunta.lower() == "sair":
                break
            try:
                resposta = qa_system.invoke({"query": pergunta})
                print("\nResposta:", resposta["result"])
            except Exception as e:
                print(f"Erro ao processar pergunta: {e}")
    
    except FileNotFoundError:
        print(f"Erro: O arquivo '{text_file}' não foi encontrado.")
    except Exception as e:
        print(f"Erro ao construir sistema QA: {e}")

if __name__ == "__main__":
    main()