from langchain_chroma import Chroma
from utils.my_llm import embeddings
from tools.vector_store import CHROMA_DB_PATH


print(f"正在连接数据库: {CHROMA_DB_PATH} ...")
vector_store = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=embeddings,
)

law_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
missing_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
async def search_law(query: str):
    res = await law_retriever.ainvoke(query)
    return res
    
async def search_missing(query: str):
    res = await missing_retriever.ainvoke(query)
    return res
