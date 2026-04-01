from typing import TypedDict, Annotated
from pydantic import Field
import operator

class State(TypedDict):
    question: str = Field(description="用户的问题")
    caogao: str = Field(description="模型的回答")
    keywords: str = Field(description="检索的关键词或语句")
    documents: str = Field(description="检索到的文档")
    missing_info: str = Field(description="缺失的信息")
    evaluation: str = Field(description="节点的评估结果以及修改意见")
    cishu: int = Field(default=0, description="节点的评估次数，默认值为0")
    context: Annotated[list, operator.add] = []
    need_search: str = Field(default="", description="是否需要检索法律条款")
