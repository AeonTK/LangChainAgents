from langchain.agents import create_react_agent

from langchain.agents import load_tools
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub

from typing import List, Tuple

from langchain.agents import AgentExecutor

from tools.calculator import calculate
from tools.time import TimeTool

llm = OpenAI(temperature=0)
    
tool_names = ["wikipedia", "open-meteo-api"]
tools = load_tools(tool_names=tool_names, llm=llm)  # the community tools
tools.extend([calculate, TimeTool()])  # a custom tool

prompt = hub.pull("hwchase17/react")

class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )
    
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

react_agent_executor = AgentExecutor(agent=agent, tools=tools).with_types(
input_type=AgentInput
)
