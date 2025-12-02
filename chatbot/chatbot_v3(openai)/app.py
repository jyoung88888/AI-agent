from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.schema import HumanMessage, AIMessage , Document
from langchain_core.chat_history import InMemoryChatMessageHistory
import re , tiktoken 
import numpy as np

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory
import os

# OpenAI API 연결
api_key = config('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

# FastAPI 인스턴스 생성
app = FastAPI(title="챗봇 API", description="챗봇 백엔드 API", version="1.0.0") 

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a AI assistant. You are
    currently having a conversation with a human. Answer the questions.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

# 모델 및 파이프라인 초기화
llm = ChatOpenAI(temperature=0,  # 창의성 0으로 설정 
                 model_name='gpt-3.5-turbo',  # 모델명
                )
#윈도우 크기 k를 지정하면 최근 k개의 대화만 기억하고 이전 대화는 삭제
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4) 

llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

# RAG_챗봇 클래스 정의

    

# 챗봇 클래스 정의
class ChatBot:
    def __init__(self):
        # 대화 기록 초기화
        self.chat_history = InMemoryChatMessageHistory()
        # 언어 모델 파이프라인 초기화
        self.llm = llm_chain()


    def chat(self, user_input: str) -> str:
        # 응답 생성 및 대화 기록 유지
        return self.llm_chain.predict(user_input)



# 챗봇 인스턴스 초기화
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
