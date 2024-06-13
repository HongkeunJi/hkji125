import openai
import os

# API 키를 환경변수로 설정
os.environ["OPENAI_API_KEY"] = "sk-proj-oh9VBlFkT5cckAvjPmgET3BlbkFJZ74YwLkbnMA76tsvQL2W"

# Langsmith api 환경변수 설정
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"]="pt-bumpy-regard-71"

import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from google.colab import drive
drive.mount('/content/drive')

from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드. 파일의 경로 입력
loader2 = PyPDFLoader("/content/drive/MyDrive/국가연구개발사업_연구개발비_사용_기준_개정안_본문_전문.pdf")
loader = PyPDFLoader("/content/drive/MyDrive/국가연구개발혁신법매뉴얼.pdf")

# 페이지 별 문서 로드
docs = loader.load()+loader2.load()
#docs2= loader2.load()

print(f"문서의 수: {len(docs)}")

# 50번째 페이지의 내용 출력
print(f"\n[페이지내용]\n{docs[50].page_content[:500]}")
print(f"\n[metadata]\n{docs[50].metadata}\n")

# text_splitter = TextSplitter(
#     chunk_size=1000, chunk_overlap=50)

# splits = text_splitter.split_documents(docs)
# len(splits)
# splits[0]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30)

splits = text_splitter.split_documents(docs)

len(splits)

# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())

# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever(k=8)

# template = """Use the following pieces of context to answer the question at the end. Please answer that you do not know anything that cannot be confirmed in the context below. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context : {context}

# Question : {question}
# Answer in Korean:"""

# from langchain_core.prompts import PromptTemplate

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=template,
# )

prompt = hub.pull("rlm/rag-prompt")
prompt

print(prompt.messages[0].prompt.template)

# llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)



def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


# 체인을 생성합니다.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("국가연구개발 과제 수행 시 연구활동비 정산 방법과 프로세스를 자세히 알려줘?")


