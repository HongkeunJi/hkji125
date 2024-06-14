import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile
import requests

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 하드코딩된 LangChain API 키와 프로젝트 설정
langchain_api_key = "YOUR_LANGCHAIN_API_KEY"
langchain_project = "YOUR_LANGCHAIN_PROJECT"

def main():
    st.set_page_config(page_title="RAG Chat")

    st.title("국가연구과제 수행방법론 Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    repo_owner = 'HongkeunJi'
    repo_name = 'hkji125'
    file_path = '국가연구개발사업_연구개발비_사용_기준_개정안_본문_전문.pdf'
    branch = 'main'  # 브랜치 이름

    # GitHub API URL
    url = f'https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}'

    # 파일 내용 가져오기
    response = requests.get(url)
    content = response.content

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        process = st.button("Process")
    
    if process:
        # API 키를 환경변수로 설정
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = langchain_project
        
        # PDF 파일 로드. 파일의 경로 입력
        local_loader = PyPDFLoader(content)
        files_text2 = local_loader.load()
        text_chunks = get_text_chunks(files_text2)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, "gpt-4")
        st.session_state.processComplete = True
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 국가연구과제 수행관련 챗봇입니다."}]

    # Chat display
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

    # Chat input
    query = st.text_input("Message to chatbot")

    if st.button("Send"):
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            if st.session_state.conversation:
                with st.spinner("Thinking..."):
                    chain = st.session_state.conversation
                    result = chain({"question": query})
                    response = result['answer']
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    source_documents = result['source_documents']
                    st.session_state.chat_history = result['chat_history']

                    st.markdown(f"**Bot:** {response}")
                    with st.expander("참고 문서 확인"):
                        for doc in source_documents:
                            st.markdown(doc.metadata['source'], help=doc.page_content)

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_document(doc):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())

    try:
        if file_path.endswith('.pdf'):
            loaded_docs = PyPDFLoader(file_path).load_and_split()
        elif file_path.endswith('.docx'):
            loaded_docs = Docx2txtLoader(file_path).load_and_split()
        elif file_path.endswith('.pptx'):
            loaded_docs = UnstructuredPowerPointLoader(file_path).load_and_split()
        else:
            loaded_docs = []
    finally:
        os.remove(file_path)

    return loaded_docs

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()
