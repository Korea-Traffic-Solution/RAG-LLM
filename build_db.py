import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# PDF 파일 경로
PDF_PATH = "./pdfs/law.pdf"
DB_FAISS_PATH = "./vectorstore/db_faiss"

# PDF 로드 및 분할
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# 임베딩 및 벡터스토어 생성
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)
db.save_local(DB_FAISS_PATH)

print("✅ 벡터스토어 구축 완료")
