import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# Langsmith API 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 하드코딩된 LangChain API 키와 프로젝트 설정
langchain_api_key = "YOUR_LANGCHAIN_API_KEY"
langchain_project = "YOUR_LANGCHAIN_PROJECT"

def main():
    st.set_page_config(
        page_title="RAG Chat")

    st.title("국가연구과제 업무처리방법")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        process = st.button("Process")
    
    # 입력받은 환경변수로 설정
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

    if process:
        if not openai_api_key:
            st.info("Please add the OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, "gpt-4o")

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 관련법령 파일을 업로드하시고, 궁금한 점을 물어보세요."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("Message to chatbot"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
def tiktoken_len(text):
    """
    주어진 텍스트에 대한 토큰 길이를 계산합니다.

    Parameters:
    - text: str, 토큰 길이를 계산할 텍스트입니다.

    Returns:
    - int, 계산된 토큰 길이입니다.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def load_document(doc):

    # 임시 디렉토리에 파일 저장
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    # 파일 쓰기
    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())  # 파일 내용을 임시 파일에 쓴다

    # 파일 유형에 따라 적절한 로더를 사용하여 문서 로드 및 분할
    try:
        if file_path.endswith('.pdf'):
            loaded_docs = PyPDFLoader(file_path).load_and_split()
        elif file_path.endswith('.docx'):
            loaded_docs = Docx2txtLoader(file_path).load_and_split()
        elif file_path.endswith('.pptx'):
            loaded_docs = UnstructuredPowerPointLoader(file_path).load_and_split()
        else:
            loaded_docs = []  # 지원되지 않는 파일 유형
    finally:
        os.remove(file_path)  # 작업 완료 후 임시 파일 삭제

    return loaded_docs

def get_text(docs):
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list


def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
