from langsmith import traceable

from dotenv import load_dotenv

from agents.openai_tools import openai_tools_agent_executor



@traceable
def run(query: str, verbose: bool = False):
    return openai_tools_agent_executor.invoke({"input": query})
    
    

if __name__ == "__main__":
    load_dotenv(override=True)
    response = run("How much is 10 plus current the number that the hour hand currently show, and then divided by the current temperature in Munich, Germany in Celsius?", verbose=True)
    print("Type: ",type(response))
    print(response)