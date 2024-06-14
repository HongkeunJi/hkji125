import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile
import git
import requests

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

# 하드코딩된 LangChain API 키와 프로젝트 설정
langchain_api_key = "YOUR_LANGCHAIN_API_KEY"
langchain_project = "YOUR_LANGCHAIN_PROJECT"

def main():
    st.set_page_config(
        page_title="RAG Chat")

    st.title("국가연구과제 수행방법론 Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

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
    content = response.text

    print(content)
    
    with st.sidebar:
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"),
            key="model_selection"
        )
        
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        process = st.button("Process")
    
    # API 키를 환경변수로 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project
    
    # PDF 파일 로드. 파일의 경로 입력
    local_loader = PyPDFLoader(url)
    files_text2 = local_loader.load()
    text_chunks = get_text_chunks(files_text2)
    vetorestore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)
    st.session_state.processComplete = True
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 국가연구과제 수행관련 챗봇입니다."}]

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
    """
    업로드된 문서 파일을 로드하고, 해당 포맷에 맞는 로더를 사용하여 문서를 분할합니다.

    지원되는 파일 유형에 따라 적절한 문서 로더(PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader)를 사용하여
    문서 내용을 로드하고 분할합니다. 지원되지 않는 파일 유형은 빈 리스트를 반환합니다.

    Parameters:
    - doc (UploadedFile): Streamlit을 통해 업로드된 파일 객체입니다.

    Returns:
    - List[Document]: 로드 및 분할된 문서 객체의 리스트입니다. 지원되지 않는 파일 유형의 경우 빈 리스트를 반환합니다.
    """
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

def get_text2(loader):
    docs = loader.load()
    text = ''
    for doc in docs:
        text += doc.text + '\n'
    return text

def get_text_chunks(text):
    """
    주어진 텍스트 목록을 특정 크기의 청크로 분할합니다.

    이 함수는 'RecursiveCharacterTextSplitter'를 사용하여 텍스트를 청크로 분할합니다. 각 청크의 크기는
    `chunk_size`에 의해 결정되며, 청크 간의 겹침은 `chunk_overlap`으로 조절됩니다. `length_function`은
    청크의 실제 길이를 계산하는 데 사용되는 함수입니다. 이 경우, `tiktoken_len` 함수가 사용되어 각 청크의
    토큰 길이를 계산합니다.

    Parameters:
    - text (List[str]): 분할할 텍스트 목록입니다.

    Returns:
    - List[str]: 분할된 텍스트 청크의 리스트입니다.

    사용 예시:
    텍스트 목록이 주어졌을 때, 이 함수를 호출하여 각 텍스트를 지정된 크기의 청크로 분할할 수 있습니다.
    이렇게 분할된 청크들은 텍스트 분석, 임베딩 생성, 또는 기계 학습 모델의 입력으로 사용될 수 있습니다.


    주의:
    `chunk_size`와 `chunk_overlap`은 분할의 세밀함과 처리할 텍스트의 양에 따라 조절할 수 있습니다.
    너무 작은 `chunk_size`는 처리할 청크의 수를 불필요하게 증가시킬 수 있고, 너무 큰 `chunk_size`는
    메모리 문제를 일으킬 수 있습니다. 적절한 값을 실험을 통해 결정하는 것이 좋습니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    주어진 텍스트 청크 리스트로부터 벡터 저장소를 생성합니다.

    이 함수는 Hugging Face의 'jhgan/ko-sroberta-multitask' 모델을 사용하여 각 텍스트 청크의 임베딩을 계산하고,
    이 임베딩들을 FAISS 인덱스에 저장하여 벡터 검색을 위한 저장소를 생성합니다. 이 저장소는 텍스트 청
