from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from dotenv import load_dotenv

from agents.openai_tools import openai_tools_agent_executor
from agents.react import react_agent_executor

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(app, react_agent_executor, path="/react", playground_type="default")
add_routes(app, openai_tools_agent_executor, path="/openai-tools", playground_type="default")

if __name__ == "__main__":
    import uvicorn
    load_dotenv(override=True)

    uvicorn.run(app, host="0.0.0.0", port=8000)
