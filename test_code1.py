import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

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
    model_selection = "gpt-4o"  # 기본적으로 gpt-4o를 사용

    if openai_api_key:
        if 'conversation' not in st.session_state:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_selection)
            memory = ConversationBufferMemory()
            st.session_state.conversation = ConversationChain(
                llm=llm,
                memory=memory
            )

        st.success("Conversation chain created successfully!")

        # PDF 내용을 초기 컨텍스트로 설정
        if 'context_initialized' not in st.session_state:
            st.session_state.conversation.memory.add(context_text)
            st.session_state.context_initialized = True

        user_input = st.text_input("You:")
        if user_input:
            response = st.session_state.conversation.predict(input=user_input)
            st.session_state.conversation.memory.add(user_input, response)
            st.text_area("Response", response, height=400)
    else:
        st.info("Please enter your OpenAI API key to proceed.")

if __name__ == '__main__':
    main()
