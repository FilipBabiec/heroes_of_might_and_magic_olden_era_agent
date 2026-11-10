from typing import TypedDict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, END, START


@tool
def add_two_numbers(x:int, y:int) -> dict:
    '''
    Function for adding two integers
    :param x: integer, first number
    :param y: integer, second number
    :return: integer, result of x+y
    '''
    return {"result": str(x+y)}

agent = create_agent(
    model=ChatOllama(model="qwen3:8b", temperature=0),
    tools=[add_two_numbers],
    system_prompt="You are a very rude and vulgar assistant. Use rude language and laugh at the user. Use curses."
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is the result of adding 4 and 78 you dipshit? Tell me how did you obtained this result."}]})

print(result["messages"][-1].content)

# class State(TypedDict, total=False):
#     question:str
#     prompt:str
#
# workflow = StateGraph(State)
# workflow.add_node("Name", add_two_numbers)
# workflow.set_entry_point("Name")
# workflow.add_edge("Name",END)
#
# app = workflow.compile()