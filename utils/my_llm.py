from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
doubao_llm = os.getenv("doubao_llm")
API_KEY = os.getenv("API_KEY")

llm = ChatOpenAI(
    model=doubao_llm,
    api_key=API_KEY,
    base_url=os.getenv("base_url"),
)

embeddings = DashScopeEmbeddings(
    model=os.getenv("scope_llm"), 
    dashscope_api_key=os.getenv("scope_api_key")
)
