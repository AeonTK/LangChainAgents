from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression. The input should be interpretable by a python evaluator."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return str(e)
        