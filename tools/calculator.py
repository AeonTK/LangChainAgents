from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)
        