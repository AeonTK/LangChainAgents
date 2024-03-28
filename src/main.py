from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

from langsmith import traceable

from dotenv import load_dotenv

from tools.calculator import calculate
from tools.time import TimeTool

# from langchain.chains import APIChain
# from langchain.chains import APIChain
# from langchain.chains.api import open_meteo_docs


@traceable
def get_response(query: str, verbose: bool = False):
    llm = OpenAI(temperature=0)
    
    tool_names = ["wikipedia", "open-meteo-api", "dalle-image-generator"]
    tools = load_tools(tool_names=tool_names, llm=llm, verbose=verbose)  # the community tools
    tools.extend([calculate, TimeTool()])  # a custom tool
    
    agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=verbose)
    response = agent.run(query)
    return response
    

if __name__ == "__main__":
    load_dotenv(override=True)
    response = get_response("How much is 10 plus current the number that the hour hand currently show, and then divided by the current temperature in Munich, Germany in Celsius?", verbose=True)
    # response = get_response("What is the current time?", verbose=True)
    print(response)