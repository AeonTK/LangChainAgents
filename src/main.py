from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

from langsmith import traceable

from dotenv import load_dotenv

# from langchain.chains import APIChain
# from langchain.chains import APIChain
# from langchain.chains.api import open_meteo_docs


@traceable
def get_response(query: str, verbose: bool = False):
    llm = OpenAI(temperature=0)
    tool_names = ["wikipedia", "open-meteo-api", "dalle-image-generator"]
    tools = load_tools(tool_names=tool_names, llm=llm, verbose=verbose)
    agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=verbose)
    response = agent.run(query)
    return response
    

if __name__ == "__main__":
    load_dotenv(override=True)
    response = get_response("What is the weather like right now in Munich, Germany in Celsius?", verbose=True)
    print(response)