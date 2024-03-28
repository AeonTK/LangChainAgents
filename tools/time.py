from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from datetime import datetime
from typing import Optional, Type




class Input(BaseModel):
    query: str = Field(description="should be empty, as this tool does not require any input")



class TimeTool(BaseTool):
    name = "Time"
    description = "Useful for when you need to answer what time it is"
    args_schema: Type[BaseModel] = Input

    def _run(
        self, query, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        now = datetime.now()
        return now.strftime("%H:%M:%S")
