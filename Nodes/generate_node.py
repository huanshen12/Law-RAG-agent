from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.my_llm import llm
from class_typed.state import State
from langchain_core.prompts import MessagesPlaceholder



def generate_node(state: State):
    """
    根据信息，生成节点
    """
    print("当前节点:生成节点")
    evaluation = state.get("evaluation", "")
    user_question = state.get("question", "")
    documents = state.get("documents", "")
    context = state.get("context", "")
    caogao = state.get("caogao", "")
    final_context = "\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages([
        ("system","""你是一个回答助手，你的任务是根据用户的问题以及已有上下文和文档生成结果，
        在法律助手和平常回答之间切换，具体要看用户有无法律需求。若有评估结果内容，则根据评估结果内容改进生成内容
        请勿生成违规内容，如果无法根据已有信息提供准确答案，则回复抱歉，无法根据已有内容生成回答，请重试或更换问题"""),
        MessagesPlaceholder(variable_name="context"),
        ("human", """用户问题: {question}
        文档内容: {final_context}
        评估结果: {evaluation}
        上版草稿: {caogao}""")
        ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "question": user_question, "final_context": final_context, "evaluation": evaluation, "caogao": caogao})
    return {"caogao": result}
