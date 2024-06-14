import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI
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

def get_conversation_chain(api_key, model_name, context, temperature=0):
    try:
        llm = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)
        # Use the context to create an initial conversation
        response = llm({"prompt": context})
        return response['choices'][0]['text']
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

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
        st.session_state.conversation = get_conversation_chain(openai_api_key, model_selection, context_text)

        if st.session_state.conversation:
            st.success("Conversation chain created successfully!")
            st.text_area("Initial Response", st.session_state.conversation, height=400)

            user_input = st.text_input("You:")
            if user_input:
                context_text += "\n" + user_input
                response = get_conversation_chain(openai_api_key, model_selection, context_text)
                st.text_area("Response", response, height=400)
        else:
            st.error("Failed to create conversation chain.")
    else:
        st.info("Please enter your OpenAI API key to proceed.")

if __name__ == '__main__':
    main()
