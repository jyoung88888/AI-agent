from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.schema import HumanMessage, AIMessage , Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login
from decouple import config
import torch
import re , tiktoken
import numpy as np

import fitz  # PyMuPDF for PDF processing
from pptx import Presentation

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from accelerate import init_empty_weights




# FastAPI 인스턴스 생성
app = FastAPI(title="챗봇 API", description="챗봇 백엔드 API", version="1.0.0")

# Hugging Face 모델 ID 및 토큰 설정
model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"
huggingface_token = config('HUGGINGFACE_TOKEN')
login(token=huggingface_token, add_to_git_credential=True)

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunk_temp = text_splitter.split_text(text)
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cuda'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

# 모델 및 파이프라인 초기화
def initialize_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=huggingface_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        device=0
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


# RAG_챗봇 클래스 정의
class RaG_ChatBot:
    def __init__(self):
        # 대화 기록 초기화
        self.chat_history = InMemoryChatMessageHistory()
        # 언어 모델 파이프라인 초기화
        self.llm = initialize_pipeline()

    def get_conversation_chain(self , vetorestore, llm):
        conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                chain_type="stuff", 
                retriever=vetorestore.as_retriever(search_type = 'mmr', verbose = True), 
                memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
                get_chat_history=lambda h: h,
                return_source_documents=True,
                verbose = True
                )
        return conversation_chain

    def generate_response(self, user_input: str, vector_store : list ):
        # 사용자의 입력을 직접 프롬프트로 사용
        prompt_text = f"{user_input}\n\n"
        chain = self.get_conversation_chain(vector_store , self.llm)
        result = chain({"question": prompt_text})
        response = result['answer']
        result = response.split('\n')[-1]
        clean_response = re.sub(r'^Helpful Answer:', '', result)
                
        # 응답에서 불필요한 토큰 제거
        #clean_response = re.sub(r'</?div.*?>|</u>|</s>|</?[^>]+>|<pad>|<unk>|<mask>', '', response).strip()
        
        # 대화 기록 업데이트
        self.chat_history.add_message(HumanMessage(content=user_input))
        self.chat_history.add_message(AIMessage(content=clean_response))
        
        return clean_response

    def chat(self, user_input: str, vector_store : list ):
        # 응답 생성 및 대화 기록 유지
        return self.generate_response(user_input, vector_store)
    

# 챗봇 클래스 정의
class ChatBot:
    def __init__(self):
        # 대화 기록 초기화
        self.chat_history = InMemoryChatMessageHistory()
        # 언어 모델 파이프라인 초기화
        self.llm = initialize_pipeline()

    def generate_response(self, user_input: str) -> str:
        # 사용자의 입력을 직접 프롬프트로 사용
        prompt_text = f"{user_input}\n\n"

        # 모델 응답 생성
        response = self.llm.invoke(prompt_text)
        
        # 응답에서 불필요한 토큰 제거
        clean_response =  re.sub(r"^(.*?)\n\n", "", response, flags=re.DOTALL).strip()
        
        # 대화 기록 업데이트
        self.chat_history.add_message(HumanMessage(content=user_input))
        self.chat_history.add_message(AIMessage(content=clean_response))
        
        return clean_response

    def chat(self, user_input: str) -> str:
        # 응답 생성 및 대화 기록 유지
        return self.generate_response(user_input)



# 챗봇 인스턴스 초기화
rag_chatbot = RaG_ChatBot()
chatbot = ChatBot()

# API 요청 모델 정의
class ChatRequest(BaseModel):
    user_input: str
    


document_store = {"text": ""}

# 엔드포인트 1: API 엔드포인트 정의
@app.post("/chat", tags=["ChatBot"])
async def chat(request: ChatRequest):
    
    response_text = chatbot.chat(request.user_input)
    return {
        "user_input": request.user_input,
        "response": response_text
    }

    
# 엔드포인트 2: API 엔드포인트 정의
@app.post("/rag_chat", tags=["RaG_ChatBot"])
async def chat(request: ChatRequest):
        
    context_text = document_store.get("text", "") 
    text_chunks = get_text_chunks(context_text)
    vectordatabase = get_vectorstore(text_chunks)
    response_text = rag_chatbot.chat(request.user_input, vectordatabase)
    
    return {
        "user_input": request.user_input,
        "response": response_text
    }

# 엔드포인트 3: PDF 파일 업로드 및 처리 (POST)
@app.post("/upload_pdf",tags=["File Management"])
async def upload_pdf(file: UploadFile):
    
    # 업로드된 PDF 파일 처리
    file_content = await file.read()
    file_extension = file.filename.split('.')[-1]

    text = ""
    if file_extension == "pdf":
        # PDF 파일 처리
        document = fitz.open(stream=file_content, filetype="pdf")
        for page in document:
            text += page.get_text()
    else:
        return {"message": "지원되지 않는 파일 형식입니다."}
    
    document_store["text"] = text
    return {"message": "File received and processed successfully"}

