from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

##########################################################################################################################################
# langsmith를 이용해 호출 로그를 실시간으로 기록해 봅니다

logging.langsmith("zeldatest")


##########################################################################################################################################
# langchain 완전 기본 설정으로 모델 호출을 시도해 봅니다

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
# question = "카카오의 오늘 주가는?"
# result = llm.predict("오늘 카카오 주가와 전망을 예측해봐 장난으로")
# print(f"[답변]: {llm.invoke(question)}")


##########################################################################################################################################
# langchain 기본 설정인 prompt template을 사용해 봅니다

# from langchain_core.prompts import PromptTemplate

# # 구성요소 1) prompt template

# template = """
# 당신은 신당동 골목에 있는 아주 용한 점쟁이입니다. 생년월일을 기반으로 사주팔자를 분석하고, MBTI를 첨가하여, 고객의 신년 운세를 점쳐 주세요.
# 참고로 싸가지 없는 말투일수록 용한 느낌이 강해집니다. 또 추상적이고 3문장 이내 정도로 짧은 말이어야 전문성이 강화됩니다.

# 고객 이름 : {이름}
# 고객 MBTI : {MBTI}
# 고객 생년월일 : {사주}
# """
# prompt = PromptTemplate.from_template(template)

# # 구성요소 2) model

# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     max_tokens=1000,
#     temperature=0.8,
# )

# # 구성요소 3) ouptut parser

# from langchain_core.output_parsers import StrOutputParser
# output_parser = StrOutputParser()

# # Langchain의 중요한 부분은 'chain 구성하기' > 세 구성요소를 서로 잇는 것
# chain = prompt | model | output_parser

# # 출력 요청
# input = {"이름":"박상아","MBTI": "INTJ", "사주": "1997년 11월 21일"}
# answer = chain.invoke(input)
# print(answer)


##########################################################################################################################################

# 이제 에이전트(Agent)가 필요한 것 같으니 만들어 봅니다

# from langchain.tools import tool
# from typing import List, Dict
# from langchain_teddynote.tools import GoogleNews
# import re
# import requests
# from bs4 import BeautifulSoup


# # 도구 생성

# @tool
# def search_news(query: str) -> List[Dict[str, str]]:
#     """Search Google News by input keyword"""
#     news_tool = GoogleNews()
#     return news_tool.search_by_keyword(query, k=5)

# @tool
# def naver_news_crawl(news_url: str) -> str:
#     """Crawls a 네이버 (naver.com) news article and returns the body content."""
#     # HTTP GET 요청 보내기
#     response = requests.get(news_url)

#     # 요청이 성공했는지 확인
#     if response.status_code == 200:
#         # BeautifulSoup을 사용하여 HTML 파싱
#         soup = BeautifulSoup(response.text, "html.parser")

#         # 원하는 정보 추출
#         title = soup.find("h2", id="title_area").get_text()
#         content = soup.find("div", id="contents").get_text()
#         cleaned_title = re.sub(r"\n{2,}", "\n", title)
#         cleaned_content = re.sub(r"\n{2,}", "\n", content)
#     else:
#         print(f"HTTP 요청 실패. 응답 코드: {response.status_code}")

#     return f"{cleaned_title}\n{cleaned_content}"


# tools = [search_news, naver_news_crawl]


# # 프롬프트 생성
# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "너는 기업의 주요 소식을 알려주는 에이전트야. 사용자가 관심 있는 기업을 입력하면 관련 기사를 검색해 소식을 요약해줘."
#             "Make sure to use the `search_news` tool or 'naver_news_crawl' tool for searching keyword related user questions.",
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

# # LLM 정의
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_tool_calling_agent
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # Agent 생성
# agent = create_tool_calling_agent(llm, tools, prompt)

# from langchain.agents import AgentExecutor

# ###################
# # AgentExecutor 생성
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     max_iterations=10,
#     max_execution_time=10,
#     handle_parsing_errors=True,
# )

# # AgentExecutor 실행
# result = agent_executor.invoke({"input": "소식을 알려줘"})

# print("Agent 실행 결과:")
# print(result["output"])



##########################################################################################################################################

# 드디어 LangGraph를 만들어 봅니다
# CRAG (Corrective-RAG) : (지식 정제) > 검색 > 유사도가 낮을 경우 검색 진행, 쿼리 재작성 > 답변 생성


##################################################################
# 1단계 : 우선 문서 로드/분할/임베딩/벡터 DB 생성 & 최종 검색기(retriever) 생성
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

loader = PyMuPDFLoader("data/domesticAItrends.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

pdf_retriever = retriever
pdf_chain = chain


##################################################################
# 2단계 : 검색된 문서의 관련성 평가 

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""

    # 문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# 채팅 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Retrieval 평가기 초기화
retrieval_grader = grade_prompt | structured_llm_grader


##################################################################
# 3단계 : 답변 생성 체인

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# LangChain Hub에서 RAG 프롬프트를 가져와 사용
prompt = hub.pull("teddynote/rag-prompt")

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata["source"]}</source><page>{doc.metadata["page"]+1}</page></document>'
            for doc in docs
        ]
    )


# 체인 생성
rag_chain = prompt | llm | StrOutputParser()


##################################################################
# 3단계 : 쿼리 재작성

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Query Rewrite 시스템 프롬프트
system = """You a question re-writer that converts an input question to a better version that is optimized 
for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

# 프롬프트 정의
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Question Re-writer 체인 초기화
question_rewriter = re_write_prompt | llm | StrOutputParser()


##################################################################
# 4단계 : 웹 검색 도구

from langchain_teddynote.tools.tavily import TavilySearch
# 최대 검색 결과를 3으로 설정
web_search_tool = TavilySearch(max_results=3)


##################################################################
##################################################################

#############
# 상태 정의
from typing import Annotated, List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: Annotated[str, "The question to answer"]
    generation: Annotated[str, "The generation from the LLM"]
    web_search: Annotated[str, "Whether to add search"] # 검색을 할지 말지 Y/N
    documents: Annotated[List[str], "The documents retrieved"]


#############
# 노드 정의
from langchain.schema import Document

# 문서 검색 노드
def retrieve(state: GraphState):
    print("\n==== RETRIEVE ====\n")
    question = state["question"]

    documents = pdf_retriever.invoke(question)
    return {"documents": documents}

# 답변 생성 노드
def generate(state: GraphState):
    print("\n==== GENERATE ====\n")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}

# 문서 평가 노드
def grade_documents(state: GraphState):
    print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    relevant_doc_count = 0

    for d in documents:
        # Question-Document 의 관련성 평가
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade == "yes":
            print("==== [GRADE: DOCUMENT RELEVANT] ====")
            # 관련 있는 문서를 filtered_docs 에 추가
            filtered_docs.append(d)
            relevant_doc_count += 1
        else:
            print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")
            continue

    # 관련 문서가 없으면 웹 검색 수행
    web_search = "Yes" if relevant_doc_count == 0 else "No"
    return {"documents": filtered_docs, "web_search": web_search}


# 쿼리 재작성 노드
def query_rewrite(state: GraphState):
    print("\n==== [REWRITE QUERY] ====\n")
    question = state["question"]

    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}


# 웹 검색 노드
def web_search(state: GraphState):
    print("\n==== [WEB SEARCH] ====\n")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents}


##################################################################
# 조건부 엣지 함수 정의
def decide_to_generate(state: GraphState):
    # 평가된 문서를 기반으로 다음 단계 결정
    print("==== [ASSESS GRADED DOCUMENTS] ====")
    # 웹 검색 필요 여부
    web_search = state["web_search"]

    if web_search == "Yes":
        # 웹 검색으로 정보 보강이 필요한 경우
        print(
            "==== [DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, QUERY REWRITE] ===="
        )
        # 쿼리 재작성 노드로 라우팅
        return "query_rewrite"
    else:
        # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
        print("==== [DECISION: GENERATE] ====")
        return "generate"

##################################################################
# 그래프 생성

from langgraph.graph import END, StateGraph, START

# 그래프 상태 초기화
workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("query_rewrite", query_rewrite)
workflow.add_node("web_search_node", web_search)

# 엣지 연결
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 문서 평가 노드에서 조건부 엣지 추가
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "query_rewrite": "query_rewrite",
        "generate": "generate",
    },
)

# 엣지 연결
workflow.add_edge("query_rewrite", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# 그래프 컴파일
app = workflow.compile()
from langchain_teddynote.graphs import visualize_graph

visualize_graph(app)



##################################################################
# 그래프 실행
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, invoke_graph, random_uuid

# config 설정(재귀 최대 횟수, thread_id)
config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

# 질문 입력
inputs = {
    "question": "카나나는 ai잠재력이 어때?",
}

# 그래프 실행
invoke_graph(app, inputs, config)
