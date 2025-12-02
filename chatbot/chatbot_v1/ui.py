import streamlit as st
import requests
import time
import re 
import torch 


# FastAPI 백엔드의 API URL
API_URL  = "http://localhost:8000/chat"
API_URL_RAG = "http://localhost:8000/rag_chat"
API_URL_UPLOAD  = "http://localhost:8000/upload_pdf"

rag = False 
# 페이지 설정은 반드시 첫 줄에 있어야 함
st.set_page_config(
    page_title="Iljoo AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS 스타일 정의
st.markdown(
    """
    <style>
    /* 앱 전체 배경색 변경 */
    .stApp {
        background-color: white; /* 전체 배경색을 밝은 회색으로 변경 */
    }

    /* 제목 텍스트 스타일 */
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #333333;
        text-align: left;
        margin-bottom: 50px;
    }
    

    /* 대화 입력 박스 스타일 */
    div[data-testid="stChatInput"] > div {
        background-color : rgb(224, 255, 255);
        color : black;
        padding : 0px 0px ;
        border: 5px solid rgb(0, 139, 139); /* 기본 테두리 색상 설정*/
        border-radius: 10px ;
        font-size:16px;
        line-height : 2;
    }
    
    div[data-testid="stChatInput"] > div:focus-within {
        outline: none; /* 포커스 아웃라인 제거 */
        border: 5px solid rgb(0, 139, 139);
        border-color: #008080; /* 포커스 시에도 테두리 색상을 동일한 청록색으로 유지 */
    }

    /* 모든 input 요소에 대해 포커스 시 발생하는 빨간색 기본 아웃라인 제거 */
    input:focus {
        outline: none; /* 포커스 아웃라인 제거 */
        border: 5px solid rgb(0, 139, 139);/* 기본 테두리 색상 유지 (청록색) */
    }
    
    /* 입력 버튼 스타일 */
    .stButton button {
        background-color: rgb(224, 255, 255); /* 버튼 배경색 */
        color: black;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Streamlit UI 설정
st.markdown('<div class="title">🤖 Iljoo Mini Chat-bot </div>', unsafe_allow_html=True)


# 사이드바 설정
with st.sidebar:
    st.title("참고 문서 업로드")
    uploaded_files =  st.file_uploader("파일을 업로드하세요.",type=['pdf','docx'], accept_multiple_files=True)
    process = st.button("Process")
    
    
if process and uploaded_files:
    # 파일 전송
    files = {'file': [uploaded_files[0].name, uploaded_files[0].getvalue(), uploaded_files[0].type]}

    response = requests.post(API_URL_UPLOAD, files=files)
    
    # 업로드 결과 확인
    if response.status_code == 200:
        st.success("PDF 파일이 성공적으로 업로드되었습니다.")
    else:
        st.error(f"PDF 파일 업로드 실패: 상태 코드 {response.status_code}")
    
    rag = True
    
    
# 대화 기록 세션 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# 불필요한 부분을 제거하는 함수 정의
def clean_response(text):
    # <img>, <a> 태그 및 URL 패턴 제거
    text = re.sub(r'<img.*?>', '', text)  # <img> 태그 제거
    text = re.sub(r'<a.*?>.*?</a>', '', text)  # <a> 태그 제거
    text = re.sub(r'http[s]?://\S+', '', text)  # URL 제거

     # 첫 번째 줄까지만 추출 (필요 없는 추가 내용 제거)
    texts = re.split(r'\n\n|[.!?]\s', text) # 두 줄 사이 공백이나 마침표 이후 내용 제거
    result = ""
    for t in texts[:3]:
        if not t.endswith("."):
            text1 = t + ". " 
        result += text1

    # 텍스트 끝에 마침표가 없으면 마침표를 추가
    #if not text.endswith("."):
    #    text = text + "."
        
    return result.strip()

# 모든 대화 내용을 지우는 함수 정의
def clear_chat_history():
    st.session_state["chat_history"] = []
    


# '대화 내용 지우기' 버튼 추가
if st.button("내용 지우기"):
    clear_chat_history()
    # GPU 메모리 해제
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 기존 대화 기록을 먼저 표시
for message in st.session_state.chat_history:  # 모든 기록을 표시
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "bot":
        with st.chat_message("bot"):
            st.markdown(message["content"])
            

# 채팅 입력창
user_input = st.chat_input("질문을 입력하세요.")
if user_input :
    
    if rag == True:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = requests.post(API_URL_RAG, json={"user_input": user_input})
        response_text = response.json()["response"]
        cleaned_response_text = clean_response(response_text)

        # 봇의 응답을 대화 기록에 추가 (길이 제한 없이 전체 텍스트 추가)
        st.session_state.chat_history.append({"role": "bot", "content": cleaned_response_text})
        
        
    # 새롭게 추가된 질문과 답변 표시 (타이핑 애니메이션 포함)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("bot"):
            animated_text = st.empty()  # 애니메이션용 임시 출력 위치
            bot_text = ""
        
        # 한 글자씩 타이핑 애니메이션 구현
            for char in cleaned_response_text:
                bot_text += char
                animated_text.markdown(bot_text)
                time.sleep(0.05)  # 타이핑 속도 조절
                
    elif rag == False:
    # 사용자 입력을 대화 기록에 추가하고 즉시 표시
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
    # 사용자 입력을 FastAPI 백엔드에 전송하고 응답 받기
        response = requests.post(API_URL, json={"user_input": user_input})
        response_text = response.json()["response"]
    #st.markdown(response_text)

    # 질문 부분의 길이를 계산하고 답변에서 해당 부분을 제거
    #question_length = len(user_input)
    #trimmed_response_text = response_text[question_length:].strip()  # 질문 길이만큼 잘라내고 나머지 출력

    # 불필요한 부분을 제거
        cleaned_response_text = clean_response(response_text)

    # 봇의 응답을 대화 기록에 추가 (길이 제한 없이 전체 텍스트 추가)
        st.session_state.chat_history.append({"role": "bot", "content": cleaned_response_text})

    # 새롭게 추가된 질문과 답변 표시 (타이핑 애니메이션 포함)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("bot"):
            animated_text = st.empty()  # 애니메이션용 임시 출력 위치
            bot_text = ""
        
            # 한 글자씩 타이핑 애니메이션 구현
            for char in cleaned_response_text:
                bot_text += char
                animated_text.markdown(bot_text)
                time.sleep(0.05)  # 타이핑 속도 조절