import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import os
import time

# LangSmith import
from langsmith import LangSmithClient

class Document:
    def __init__(self, text):
        self.text = text

class PDFLoader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def load(self):
        documents = []
        for pdf_path in self.pdf_paths:
            if pdf_path.endswith('.pdf'):
                text = self._extract_text_from_pdf(pdf_path)
                documents.append(Document(text))
        return documents

    def _extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text

def get_text(loader):
    docs = loader.load()
    text = ''
    for doc in docs:
        text += doc.text + '\n'
    return text

def main():
    st.title("Chat Application")

    # 미리 읽어들일 PDF 파일들
    pdf_paths = ["1.pdf", "2.pdf"]
    
    # PDF 파일들을 로드
    pdf_loader = PDFLoader(pdf_paths)
    context_text = get_text(pdf_loader)

    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    langsmith_api_key = "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"  # 직접 입력
    langsmith_project = "pt-bumpy-regard-71"  # 직접 입력

    model_selection = "gpt-4o"  # 기본적으로 gpt-4o를 사용

    if openai_api_key and langsmith_api_key and langsmith_project:
        if 'conversation' not in st.session_state:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection)
            memory = ConversationBufferMemory()
            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=memory
            )

            # LangSmith 클라이언트 설정
            langsmith_client = LangSmithClient(api_key=langsmith_api_key, project=langsmith_project)
            st.session_state.langsmith_client = langsmith_client

            # PDF 내용을 초기 컨텍스트로 설정
            st.session_state.conversation.memory.chat_memory.add_message(
                SystemMessage(content=context_text)
            )

        st.success("Conversation chain created successfully!")

        user_input = st.text_input("You:")
        if user_input:
            st.session_state.conversation.memory.chat_memory.add_message(
                HumanMessage(content=user_input)
            )
            response = None
            retries = 3  # 재시도 횟수
            for attempt in range(retries):
                try:
                    response = st.session_state.conversation.predict(input=user_input)
                    break
                except openai.error.RateLimitError:
                    st.warning("Rate limit exceeded. Retrying in 5 seconds...")
                    time.sleep(5)  # 5초 대기 후 재시도

            if response:
                st.session_state.conversation.memory.chat_memory.add_message(
                    AIMessage(content=response)
                )
                st.text_area("Response", response, height=400)

                # 대화 내용을 LangSmith에 기록
                st.session_state.langsmith_client.log_message(user_input, response)
            else:
                st.error("Failed to get a response from the OpenAI API after several attempts.")
    else:
        st.info("Please enter your OpenAI API key to proceed.")

if __name__ == '__main__':
    main()
