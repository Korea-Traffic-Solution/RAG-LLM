import os
from fastapi import FastAPI, Request, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from urllib.parse import urlencode
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# Firestore 초기화
import firebase_admin
from firebase_admin import firestore
import Firebase.firebase_config
firestore_db = firestore.client()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DB_FAISS_PATH = "./vectorstore/db_faiss"
embeddings = OpenAIEmbeddings()
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """
너는 전동킥보드 법률 안내 챗봇이야. 아래 '법률 정보'만 참고해.
답변은 반드시 아래 형식을 따라야 해.

🚨 법률 위반 감지
- 위반사항: [위반사항]
- 적용 법률: [법률명 및 조항]
- 벌금: [벌금]
- 비고: [초범/재범 여부, 추가 처벌 등]

법률 정보:
{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {
        "context": retriever | docs2str,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

@app.post("/load-user-data")
async def load_user_data(data: dict = Body(...)):
    uid = data.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="No uid")
    doc = firestore_db.collection("Conclusion").document(uid).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = doc.to_dict()
    return {"user_data": user_data}

class AskWithUid(BaseModel):
    uid: str
    question: str

@app.post("/ask_with_uid")
def ask_with_uid(query: AskWithUid):
    try:
        doc = firestore_db.collection("Conclusion").document(query.uid).get()
        user_data = doc.to_dict() if doc.exists else {}

        prompt_input = f"""
        유저 정보: {user_data}
        질문: {query.question}
"""
        answer = rag_chain.invoke(prompt_input)
        return {"result": answer.content if hasattr(answer, "content") else str(answer)}
    except Exception as e:
        return {"result": f"서버 오류: {e}"}

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request, result: str = None, violation: str = None):
    # GET 요청 시 result와 violation이 있으면 말풍선 표시, 없으면 빈 폼
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "violation": violation})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, violation: str = Form(...)):
    # POST 후 Redirect(쿼리파라미터로 답변과 질문 전달)
    answer = rag_chain.invoke(violation)
    result = answer.content if hasattr(answer, "content") else str(answer)
    query_params = urlencode({"result": result, "violation": violation})
    return RedirectResponse(url=f"/?{query_params}", status_code=303)

# 기존 /ask API도 유지
class Query(BaseModel):
    violation: str

@app.post("/ask")
def ask(query: Query):
    try:
        answer = rag_chain.invoke(query.violation)
        return {"result": answer.content if hasattr(answer, "content") else str(answer)}
    except Exception as e:
        return {"result": f"서버 오류: {e}"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)