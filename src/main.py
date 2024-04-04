from langchain.agents import initialize_agent, create_openai_tools_agent, load_tools, AgentExecutor
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from langchain import hub

from langsmith import traceable

from dotenv import load_dotenv

from tools.calculator import calculate
from tools.time import TimeTool

@traceable
def run(query: str, verbose: bool = False):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
    
    tool_names = ["wikipedia", "open-meteo-api", "dalle-image-generator"]
    tools = load_tools(tool_names=tool_names, llm=llm, verbose=verbose)  # the community tools
    tools.extend([calculate, TimeTool()])  # a custom tool
    
    # agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=verbose)
    # response = agent.run(query)
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": query})

    return response["output"]
    

if __name__ == "__main__":
    load_dotenv(override=True)
    output = run("How much is 10 plus current the number that the hour hand currently show, and then divided by the current temperature in Munich, Germany in Celsius?", verbose=True)
    print(output)