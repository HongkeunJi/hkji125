import PyPDF2
import openai
import streamlit as st
import os

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

def query_chatgpt(api_key, prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    st.title("PDF Reader and ChatGPT Query App")

    # Define the path to the local PDF file
    pdf_folder_path = "./"   # 폴더 경로를 지정하세요
    #pdf_folder_path = "./국가연구개발사업_연구개발비_사용_기준_개정안_본문_전문.pdf"  # 폴더 경로를 지정하세요
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    if pdf_files:
        selected_pdf = st.selectbox("Choose a PDF file", pdf_files)
        pdf_path = os.path.join(pdf_folder_path, selected_pdf)
        
        pdf_text = read_pdf(pdf_path)
        st.text_area("PDF Content", pdf_text, height=200)
        
        # Predefined API keys
        openai_api_key = "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"  # 여기에 OpenAI API 키를 입력하세요
        langsmith_api_key = "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"  # 여기에 LangSmith API 키를 입력하세요
        
        # Query input and button
        query = st.text_area("Enter your query")
        if st.button("Ask ChatGPT"):
            if query:
                response = query_chatgpt(openai_api_key, query)
                st.text_area("ChatGPT Response", response, height=200)
            else:
                st.error("Please provide a query.")
    else:
        st.error("No PDF files found in the specified folder.")

if __name__ == "__main__":
    main()
