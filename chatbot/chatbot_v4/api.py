from fastapi import FastAPI, Query, UploadFile, HTTPException
from tempfile import NamedTemporaryFile
import os
import json 
import requests
from datetime import datetime
from openai import OpenAI  # Import OpenAI for GPT API
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import  RetrievalQA
from openai import OpenAI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages
import re
import uuid
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# OpenAI API Initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI application
app = FastAPI(title="일주 AI 에이전트 API", description="통합 API", version="1.0.0")

# 현재 날짜 및 시간 가져오기
def get_current_date():
    return datetime.now().strftime("%Y%m%d")  # 예: 20241224

# 정각 시간 계산
def get_nearest_base_time():
    now = datetime.now()
    hour = now.hour
    return f"{hour:02}00"  # 정각 시간 (예: 1000)

# API URL
WEATHER_API_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

# 부산광역시와 구별 격자 좌표 정의
REGION_COORDS = {
    "부산광역시": {"nx": "98", "ny": "76"},
    "부산광역시 중구": {"nx": "97", "ny": "74"},
    "부산광역시 서구": {"nx": "96", "ny": "74"},
    "부산광역시 동구": {"nx": "98", "ny": "74"},
    "부산광역시 영도구": {"nx": "98", "ny": "73"},
    "부산광역시 진구": {"nx": "97", "ny": "75"},
    "부산광역시 동래구": {"nx": "98", "ny": "77"},
    "부산광역시 남구": {"nx": "97", "ny": "74"},
    "부산광역시 북구": {"nx": "96", "ny": "77"},
    "부산광역시 해운대구": {"nx": "100", "ny": "75"},
    "부산광역시 사하구": {"nx": "96", "ny": "73"},
    "부산광역시 금정구": {"nx": "99", "ny": "77"},
    "부산광역시 강서구": {"nx": "94", "ny": "77"},
    "부산광역시 연제구": {"nx": "98", "ny": "76"},
    "부산광역시 수영구": {"nx": "99", "ny": "75"},
    "부산광역시 사상구": {"nx": "96", "ny": "75"},
    "부산광역시 기장군": {"nx": "98", "ny": "78"},

    # 전국 특별시 및 광역시 좌표
    "서울특별시": {"nx": "60", "ny": "127"},
    "인천광역시": {"nx": "55", "ny": "124"},
    "대전광역시": {"nx": "67", "ny": "100"},
    "광주광역시": {"nx": "58", "ny": "74"},
    "대구광역시": {"nx": "89", "ny": "90"},
    "울산광역시": {"nx": "102", "ny": "84"},
    "세종특별자치시": {"nx": "66", "ny": "103"},

    # 전국 도 좌표
    "경기도": {"nx": "60", "ny": "120"},  # 수원 기준
    "강원도": {"nx": "73", "ny": "134"},  # 춘천 기준
    "충청북도": {"nx": "69", "ny": "107"},  # 청주 기준
    "충청남도": {"nx": "68", "ny": "100"},  # 홍성 기준
    "전라북도": {"nx": "63", "ny": "89"},  # 전주 기준
    "전라남도": {"nx": "51", "ny": "67"},  # 목포 기준
    "경상북도": {"nx": "91", "ny": "106"},  # 안동 기준
    "경상남도": {"nx": "91", "ny": "77"},  # 창원 기준
    "제주특별자치도": {"nx": "52", "ny": "38"},  # 제주 기준
}

# 기상청 API 호출 및 데이터 가져오기
def fetch_weather_data(nx, ny):
    current_date = get_current_date()
    base_time = get_nearest_base_time()

    params = {
        'serviceKey': os.getenv('WEATHER_API_KEY'),  # 환경 변수에서 서비스 키 가져오기
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': current_date,
        'base_time': base_time,
        'nx': nx,
        'ny': ny
    }
    
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        try:
            api_response = response.json()
            response_data = api_response.get('response', {})
            body = response_data.get('body', {})
            items = body.get('items', {}).get('item', [])
            
            # 필요한 데이터 추출
            temperature = next((item['obsrValue'] for item in items if item['category'] == 'T1H'), None)
            wind_speed = next((item['obsrValue'] for item in items if item['category'] == 'WSD'), None)
            humidity = next((item['obsrValue'] for item in items if item['category'] == 'REH'), None)
            
            return {
                "temperature": temperature,
                "wind_speed": wind_speed,
                "humidity": humidity
            }
        except Exception as e:
            return {"error": f"Data processing error: {e}"}
    else:
        return {"error": f"HTTP Error: {response.status_code}"}


def extract_location_from_question(question):
    """
    Extract the most specific region name from the user's question using GPT,
    prioritizing smaller regions (구 > 군 > 시) based on hierarchy.
    """
    prompt = f"""
    You are a helpful assistant. Analyze the user's question and identify the most specific region mentioned.
    Use the regions defined in the REGION_COORDS dictionary.

    Question: "{question}"
    REGION_COORDS: {list(REGION_COORDS.keys())}
    
    Your response should:
    1. Only include the name of the most specific region (e.g., 구 > 군 > 시) from the question.
    2. If a "동" is mentioned, determine the corresponding "구" if not directly listed in REGION_COORDS.
    3. If no region is found, return "None".
    """
    try:
        # GPT 요청
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.0
        )

        # GPT 응답에서 지역명 추출
        content = response.choices[0].message.content.strip()

        # Validate GPT response against REGION_COORDS
        if content in REGION_COORDS.keys():
            return content

        return None  # No valid region found
    except Exception as e:
        return {"error": f"Failed to extract location using GPT: {e}"}
    

@app.post("/iljoo_ai_agent/weather", tags=["Weather"])
def get_weather_response(question: str = Query(...)):
    """
    Extract location and generate weather-based recommendations from the user's question.
    """
    # OpenAI GPT를 이용하여 지역명 및 추천 메시지 추출
    region = extract_location_from_question(question)

    if not region or region == "None":
        return {"error": "No valid region found in the question."}

    # 지역 날씨 데이터 가져오기
    coords = REGION_COORDS.get(region)
    if not coords:
        return {"error": f"Coordinates for region '{region}' not found."}

    weather_data = fetch_weather_data(coords["nx"], coords["ny"])
    if "error" in weather_data or not weather_data.get("temperature"):
        return {"error": "Failed to fetch valid weather data for the region."}

    # 날씨 정보를 바탕으로 GPT에서 정확한 추천 생성
    try:
        prompt = f"""
        You are a helpful assistant. The user asked about the weather in '{region}'.
        Here is the weather data:
        - Temperature: {weather_data['temperature']}°C
        - Wind Speed: {weather_data['wind_speed']}m/s
        - Humidity: {weather_data['humidity']}%

        Based on this data, provide a recommendation for clothing and actions. Please make sure to write the response in Korean.
        Format your response as:
        Recommendation: [Weather Recommendation]
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3
        )

        # GPT 응답에서 추천 메시지 추출
        content = response.choices[0].message.content.strip()
        match_recommendation = re.search(r"Recommendation:\s*(.+)", content)
        recommendation = match_recommendation.group(1) if match_recommendation else "No specific recommendation available."
    except Exception as e:
        recommendation = f"Failed to generate recommendation: {e}"

    # 응답 메시지 생성
    return {
        "response": f"""
        {region}의 현재 날씨 정보:
        - 기온: {weather_data['temperature']}°C
        - 풍속: {weather_data['wind_speed']}m/s
        - 습도: {weather_data['humidity']}%
        
        추천 사항:
        {recommendation}
        """
    }

# RAG 
def RAG_pipeline(doc):
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    data = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(data, embeddings)
    return vector_store


# 챗봇 클래스 정의
task_prompt = {
    "summary" : """Please summarize the core contents of the uploaded business announcement. 
                    Write it down in detail, but divide the items and organize them. 
                    If possible, please organize the details structurally and utilize the nested structure by item.
                    You are an assistant that answers questions based on the following context: {context}
                    User's question: {question}\n""",        
   
    "qa" : """당신은 유능한 어시스턴트입니다. 아래 대화 기록과 문맥(context)을 기반으로 질문에 답하세요:
              You are an assistant that answers questions based on the following context: {context}
              User's question: {question}\n\n""",
    
    "json": """You are an assistant that processes business-related documents and provides structured outputs in JSON format. 
                You are expected to create a structured JSON output based on the following context: {context}. 
                Ensure that the JSON keys are written in Korean, and use a nested structure to organize the content. 
                If a user has a question, respond based on the context and the generated JSON.

                User's question: {question}
                Summarize the document with reference to the above example, and be sure to strictly adhere to the JSON format."""
}

class RAG_ChatBot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 50})
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_token=150)        
        #self.reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        
    def call_model(self, state: MessagesState, prompt ):
        Prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt
        )
        
        # Reranker / 상위 3개의 문서 선택
        #compressor = CrossEncoderReranker(model=self.reranker_model, top_n=3)

        # 문서 압축 검색기 초기화
        #compression_retriever = ContextualCompressionRetriever(
        #    base_compressor=compressor, base_retriever=self.retriever)
            
        # Chain 생성
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": Prompt},
            chain_type = "stuff"
        )
        
        selected_messages = trim_messages(
            state["messages"],
            token_counter=len,  # <-- len will simply count the number of messages rather than tokens
            max_tokens=5,  # <-- allow up to 5 messages.
            strategy="last",
            # Most chat models expect that chat history starts with either:
            # (1) a HumanMessage or
            # (2) a SystemMessage followed by a HumanMessage
            # start_on="human" makes sure we produce a valid chat history
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # if it's present in the original history.
            # The SystemMessage has special instructions for the model.
            include_system=True,
            allow_partial=False,
        )
        query = selected_messages[-1].content
        result = chain.invoke({'query': query})
        
        if prompt == task_prompt['json']:
            parsed_response = JsonOutputParser().parse(result['result'])
            json_response = '```json\n' +  json.dumps(parsed_response, ensure_ascii=False, indent=2) + '\n```'
            return {"messages": [AIMessage(content=json_response)]} 
        else:
            return {"messages": [AIMessage(content=result['result'])]} 
    
    def chat(self, call_model, user_input:str, task ):
        workflow = StateGraph(state_schema=MessagesState)
        
        workflow.add_edge(START, "model")
        if task == 'summary':
            workflow.add_node("model",  lambda state: self.call_model(state, task_prompt['summary']))
            
        elif task == 'qa':
            workflow.add_node("model",  lambda state: self.call_model(state, task_prompt['qa']))
        
        elif task == 'json':
            workflow.add_node("model",  lambda state: self.call_model(state, task_prompt['json']))
        
        memory = MemorySaver()
        
        app = workflow.compile(checkpointer=memory)
        
        thread_id = uuid.uuid4()
        config = {"configurable": {"thread_id": thread_id}}
        
        input_message = HumanMessage(content=user_input)
        for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
            answer = event["messages"][-1]
            
        return answer.content


# API 요청 모델 정의
class ChatRequest(BaseModel):
    user_input: str
    task : str



# 엔드포인트 1: API 엔드포인트 정의
@app.post("/iljoo_ai_agent/chat", tags=["Chatbot"])
async def chat(request: ChatRequest):
    
    vector_store = RAG_pipeline(app.state.document_store)
    chatbot = RAG_ChatBot(vector_store)
    print(request.task)
    response = chatbot.chat(call_model = chatbot.call_model , user_input= request.user_input, task = request.task)
    print(type(response))
    print(response)
    
    return {
        "user_input": request.user_input,
        "task": request.task,
        "response": response
    }

# 엔드포인트 2: PDF 파일 업로드 및 처리 (POST)
@app.post("/iljoo_ai_agent/upload_pdf",tags=["File Management"])
async def upload_pdf(file: UploadFile):
    
    app.state.document_store = []
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # 임시 파일에 저장
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_pdf_path = temp_file.name

        # Step 1: PDF 처리
        loader = PyMuPDFLoader(temp_pdf_path)
        docs = loader.load()
        app.state.document_store.extend(docs)
        print(len(app.state.document_store))
        
        return {"message": "PDF uploaded and processed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
            