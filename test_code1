import PyPDF2
import openai
import streamlit as st

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

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_text = read_pdf(uploaded_file)
        st.text_area("PDF Content", pdf_text, height=200)
        
        openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
        langsmith_api_key = st.text_input("Enter your LangSmith API Key", type="password")
        
        query = st.text_area("Enter your query")
        if st.button("Ask ChatGPT"):
            if openai_api_key and query:
                response = query_chatgpt(openai_api_key, query)
                st.text_area("ChatGPT Response", response, height=200)
            else:
                st.error("Please provide both OpenAI API key and a query.")

if __name__ == "__main__":
    main()
