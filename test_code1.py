import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI

class Document:
    def __init__(self, text):
        self.text = text

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load(self):
        documents = []
        if self.pdf_path.endswith('.pdf'):
            text = self._extract_text_from_pdf(self.pdf_path)
            documents.append(Document(text))
        return documents

    def _extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ''
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
        # Initialize the conversation chain with the given context.
        # Assuming the model takes the context as initial input.
        response = llm.complete(context)
        return response
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def main():
    st.title("Chat Application")

    # 사전에 정의된 PDF 파일 경로
    pdf_path = '1.pdf'
    pdf_loader = PDFLoader(pdf_path)
    context_text = get_text(pdf_loader)
    
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    model_selection = st.selectbox("Choose a model:", ["gpt-3.5-turbo", "gpt-4"])

    if openai_api_key and model_selection:
        st.session_state.conversation = get_conversation_chain(openai_api_key, model_selection, context_text)

        if st.session_state.conversation:
            st.success("Conversation chain created successfully!")
            st.text_area("Response", st.session_state.conversation, height=400)
        else:
            st.error("Failed to create conversation chain.")

if __name__ == '__main__':
    main()
