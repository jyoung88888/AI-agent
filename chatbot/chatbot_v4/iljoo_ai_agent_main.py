import os
import uuid
import streamlit as st
from langchain_core.messages import trim_messages, HumanMessage 
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START
import requests
from langchain_core.output_parsers import JsonOutputParser
from huggingface_hub import login
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Langchain
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader # Wikipedia에서 문서를 로드합니다.
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import re
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch 
import os

# FastAPI 서버 URL
WEATHER_API = "http://127.0.0.1:8000//iljoo_ai_agent/weather"
CHAT_API = "http://127.0.0.1:8000/iljoo_ai_agent/chat"
UPLOADFILE_API = "http://127.0.0.1:8000/iljoo_ai_agent/upload_pdf"

# Huggingface token 설정 
hf_token = os.getenv("HF_TOKEN")

# ConversationBufferWindowMemory
def trim_conversation_history(messages, k=5):
    return trim_messages(
        messages,
        token_counter=len,
        max_tokens=k,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

# huggingface model 지정 
model_name = "iljoo/models-iljoodeephub-Bllossom-llama-3.2-Korean-Bllossom-3B_bf16_lr64_qlr4_test2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_buffers=True
)


def question_answer(messages):
    input_ids = tokenizer.apply_chat_template(
        [{"role": "human", "content": messages}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.1
    )

    result = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return result
    
# 대화 관리 설정
def call_model(state: MessagesState):
    # Trim the user's conversation history to manage token limits

    trimmed_messages = trim_conversation_history(state["messages"], k=5)
    response = question_answer(trimmed_messages[-1].content)
    
    return {"messages": [AIMessage(content=response)]}

# Set the workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# 메모리 지속성 구현
memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)

# 대화 스레드 관리
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# 대화 기록 관리 함수
def display_messages():
    for message in st.session_state["messages"]:
        if message["role"] == "human":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])

def main():
    st.title("Iljoo AI Agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 일주 AI Agent입니다. 무엇을 도와드릴까요?"}]

    if "uploaded_pdf" not in st.session_state:    # 업로드 파일 세션 유지 
        st.session_state['uploaded_pdf'] = [] 
         
    # 변경된 코드
    if st.button("전체 대화 삭제"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 일주 AI Agent입니다. 무엇을 도와드릴까요?"}
        ]
        st.query_params.clear()  # 대화 상태 재설정

    # 기존 메시지 표시
    display_messages()
    user_prompt = st.chat_input("질문을 입력하세요:")
    # 사용자 입력 처리
    if user_prompt:
        ai_response = ""
        # 사용자 메시지 기록
        st.session_state["messages"].append({"role": "human", "content": user_prompt})

        # 새로운 메시지 표시 (사용자 메시지)
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        input_message = HumanMessage(content=user_prompt)
        response_message = None
        for event in app.stream({"messages": input_message}, config, stream_mode="values"):
            response_message = event["messages"][-1]
            
        ai_response = response_message.content if response_message else "응답 생성 중 오류가 발생했습니다."
                
        # AI 응답 기록
        st.session_state["messages"].append({"role": "assistant", "content": ai_response})

        # 새로운 메시지 표시
        with st.chat_message("assistant"):
            st.markdown(ai_response)
            
# Run the Streamlit app
if __name__ == "__main__":
    main()
