채팅 기록


ChatGPT
ChatGPT
GPT
Whimsical Diagrams
GPT
Write For Me
GPT
Photo Realistic GPT
GPT
어르신 대화 친구

3 더 보기

GPT 탐색
오늘
해석 및 제목화

Streamlit Chatbot Implementation
User 요청 및 해석
Read Git files Python
Text Processing with Langsmith
EDA 경찰서별 치안센터
DHEA 보충제 효과
어제
대용량 엑셀 데이터 처리
브랜치 변경 및 푸시
지난 7일
PTN UNI vs NNI.
주소에서 좌표 찾기
주소 데이터 가져오기
인건비 현물출자 정산 처리
국가 R&D 공고 API
RCMS API 요청 및 응답
과제 감사 준비 요령
Botpress vs ChatGPT-4 결과 비교
과제 신청 및 승인
연구비 관리 지침 요약
연구개발비 집행 가이드라인
RCMS 시스템 사용법
AI 연구과제 안내
스키마 오류 해결
API 신청 및 사용
법령 검색 API
법령 검색 스키마
과제 생성 절차 요약
웹 페이지 만들기
연구활동비 사용지침
GPTs: 무료 계정 제한
손흥민, 24-25 골 기록
도색 문제점 요약
필요한 서류 요약
등산 소요시간: 상원사 - 오대산
설악산 등산 코스
등산 코스 추천
지난 30일
선자령 등산 소요시간
통계 분석 소프트웨어 추천
켄싱턴호텔 & 용평리조트
발왕산 여행 코스
성향 잘 조절하기
표시: 을지로 65
RAG 모델과 Streamlit
RCMS 연구활동비 사용 방법
간접비 종류 및 사용기준
데이터베이스 스키마 사용목적
JSON 스키마 요청 및 응답
스키마 만들기 요청.
사주팔자 제공 가능!
RAG 모델 Streamlit 개발
Test: Help With Inquiry
Team 워크스페이스 추가
Team 플랜에서 함께 작업하세요
ChatGPT 4o


User
내 소스코드 해석해봐
ChatGPT
소스 코드를 제공해 주시면 해석해 드리겠습니다. 코드를 텍스트 형식으로 여기에 붙여 주세요.

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

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"



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
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"),
            key="model_selection"
        )
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        # 환경 변수 입력을 위한 UI 추가
        langchain_api_key = st.text_input("LangChain API Key", key="langchain_api_key", type="password")
        langchain_project = st.text_input("LangChain Project", key="langchain_project")
        
        process = st.button("Process")
    
    # 입력받은 환경변수로 설정
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

    if process:
        if not openai_api_key or not langchain_api_key or not langchain_project:
            st.info("Please add all necessary API keys and project information to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)

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

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key, model_selection):

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain



if __name__ == '__main__':
    main()
ChatGPT
이 코드는 Streamlit을 사용하여 웹 애플리케이션을 구축하는 예제입니다. 이 애플리케이션은 사용자가 PDF, DOCX, PPTX 파일을 업로드하고, OpenAI와 LangChain API를 사용하여 문서를 분석하고, 대화형 챗봇을 통해 질문에 답변하는 기능을 제공합니다. 코드의 주요 부분을 설명드리겠습니다.

주요 라이브러리 임포트
python
코드 복사
import streamlit as st
import tiktoken
from loguru import logger
import os
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
환경 변수 설정
python
코드 복사
# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
메인 함수
python
코드 복사
def main():
    st.set_page_config(page_title="RAG Chat")
    st.title("국가연구과제 업무처리방법")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        model_selection = st.selectbox("Choose the language model", ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"), key="model_selection")
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        langchain_api_key = st.text_input("LangChain API Key", key="langchain_api_key", type="password")
        langchain_project = st.text_input("LangChain Project", key="langchain_project")
        
        process = st.button("Process")
    
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

    if process:
        if not openai_api_key or not langchain_api_key or not langchain_project:
            st.info("Please add all necessary API keys and project information to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 관련법령 파일을 업로드하시고, 궁금한 점을 물어보세요."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

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

        st.session_state.messages.append({"role": "assistant", "content": response})
텍스트 및 문서 처리 함수
python
코드 복사
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
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain
실행 부분
python
코드 복사
if __name__ == '__main__':
    main()
요약
Streamlit 설정: Streamlit을 사용하여 웹 애플리케이션의 페이지 제목을 설정하고 사이드바에서 모델 선택, 파일 업로드, API 키 입력 등을 받습니다.
파일 처리: 업로드된 파일을 임시 디렉토리에 저장하고, 파일 유형에 맞는 로더를 사용하여 문서를 로드합니다.
텍스트 처리: 문서를 텍스트 청크로 분할하고, 이를 벡터스토어에 저장합니다.
대화 체인 생성: 선택된 모델과 API 키를 사용하여 대화형 체인을 생성하고, 사용자가 입력한 질문에 대해 답변을 제공합니다.
채팅 인터페이스: 사용자가 입력한 메시지와 봇의 응답을 화면에 표시하고, 채팅 내역을 관리합니다.
이 코드는 주로 문서의 내용을 분석하고, 사용자가 질문할 때 해당 문서에서 관련 내용을 찾아 답변하는 데 사용됩니다. API 키와 프로젝트 정보를 환경 변수로 설정하여 LangChain API와 OpenAI API를 사용하는 것이 특징입니다.

위 소스코드에서 langchain_api_key와 langchain_project 를 내가 미리 정하는 값으로 선택하고, 입력창은 없애고 싶어
ChatGPT
langchain_api_key와 langchain_project 값을 미리 설정하고 입력창을 없애기 위해서는 다음과 같이 코드를 수정하면 됩니다. 해당 변수들의 값을 코드 내에서 직접 할당하고, Streamlit 입력 부분을 주석 처리하거나 제거합니다.

python
코드 복사
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

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 미리 설정된 값
LANGCHAIN_API_KEY = "your_langchain_api_key"
LANGCHAIN_PROJECT = "your_langchain_project"

# 입력받은 환경변수로 설정
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

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
        model_selection = st.selectbox(
            "Choose the language model",
            ("gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o"),
            key="model_selection"
        )
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        
        # 환경 변수 입력을 위한 UI 제거
        # langchain_api_key = st.text_input("LangChain API Key", key="langchain_api_key", type="password")
        # langchain_project = st.text_input("LangChain Project", key="langchain_project")
        
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key, st.session_state.model_selection)

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
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_document(doc):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, doc.name)

    with open(file_path, "wb") as file:
        file.write(doc.getbuffer())  # 파일 내용을 임시 파일에 쓴다

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
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key, model_selection):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()
요약
LANGCHAIN_API_KEY와 LANGCHAIN_PROJECT 값을 미리 정의합니다.
Streamlit 사이드바에서 해당 값을 입력받는 UI 요소를 제거합니다.
환경 변수를 설정하는 부분은 그대로 유지합니다.







메시지 ChatGPT

ChatGPT는 실수를 할 수 있습니다. 중요한 정보를 확인하세요.
?
ChatGPT
