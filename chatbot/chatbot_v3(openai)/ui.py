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

#openai 키 입력
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a AI assistant. You are
    currently having a conversation with a human. Answer the questions.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

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

st.title("ChatGPT AI Assistant")

# 세션에서 메시지를 확인하고 존재하지 않는 경우 생성
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요, 저는 AI Assistant입니다."}
    ]

# 모든 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_prompt = st.chat_input()

# 사용자 입력 보여주기
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

#마지막 메시지가 assistant로 부터 받은게 아니라면 새로운 답변 생성
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)