from langchain.agents import create_openai_tools_agent, load_tools, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain import hub

from langsmith import traceable

from dotenv import load_dotenv

from tools.calculator import calculate
from tools.time import TimeTool


from typing import List, Tuple, Optional, Union

from langchain.agents import AgentExecutor

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    StrOutputParser,
    get_buffer_string,
)

from langchain.output_parsers.openai_tools import JsonOutputToolsParser


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

tool_names = ["wikipedia", "open-meteo-api"]
tools = load_tools(tool_names=tool_names, llm=llm)  # the community tools
tools.extend([calculate, TimeTool()])  # a custom tool

prompt = hub.pull("hwchase17/openai-tools-agent")

class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )
    
class AgentOutput(BaseModel):
    output: str
    
class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

# class ChainInput(BaseModel):
#     """Input for the chat bot."""

#     chat_history: Optional[List[BaseMessage]] = Field(
#         description="Previous chat messages."
#     )
#     text: str = Field(..., description="User's latest query.")
#     last_run_id: Optional[str] = Field("", description="Run ID of the last run.")


agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)

openai_tools_agent_executor = AgentExecutor(agent=agent, tools=tools).with_types(
input_type=AgentInput, output_type=AgentOutput
)

#output_parser = StrOutputParser()

#openai_tools_agent_executor = _openai_tools_agent_executor | output_parser



