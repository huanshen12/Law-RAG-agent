import json
from langchain_core.prompts import ChatPromptTemplate
from class_typed.state import State
from utils.my_llm import llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
def falvfenxi_node(state: State):
    """
    分析用户的法律咨询情况，返回检索的关键词或语句.
    """
    print("当前节点:法律分析")
    question = state["question"]
    context = state["context"]
    prompt = ChatPromptTemplate.from_messages([("system",
        """
        你是一名律师，请你根据用户描述的情况，返回法律条款检索的关键词或语句,，输出格式只有json字符串，包含reasoning（分析情况）和search_queries（检索的关键词或语句），不要生成违规敏感内容。
        示例如下：
        用户描述：我每天疯狂加班，昨天老板又让我免费熬夜赶项目，我实在受不了拒绝了，结果今天他直接在微信上发消息说把我开除了，而且还不给我结上个月的工资！我该怎么办？能去法院告他吗？
        你的回答：{{
    "reasoning": "用户描述的情境属于劳动争议。核心法律事实有三个：1. 用人单位安排免费加班（涉嫌违反工时与加班费规定）；2. 员工拒绝无偿加班后被微信辞退（涉嫌用人单位违法单方解除劳动合同）；3. 用人单位未支付上月工资（涉嫌拖欠劳动报酬）。需要检索劳动法中关于违法解除合同、加班费以及拖欠工资的法律责任。",
    "search_queries": [
    "用人单位 违法解除劳动合同 赔偿金",
    "拒绝加班 被辞退 法律责任",
    "用人单位 拖欠劳动报酬 经济补偿"
  ]
}}
    """),
    MessagesPlaceholder(variable_name="context"),
    ("human",
        f"""用户描述：{question}
        """
    )])
    chain = prompt | llm | StrOutputParser()
    result = json.loads(chain.invoke({"context": context, "question": question}))
    return {"keywords": result["search_queries"]}
