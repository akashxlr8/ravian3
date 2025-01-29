
# from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from typing import Any
from autogen_core.tools import FunctionTool
import pandas as pd
from pydantic import BaseModel, Field

import pandas as pd
from autogen_core.tools import FunctionTool

def csv_to_markdown(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Error: {str(e)}"


csv_markdown_tool = FunctionTool(
    csv_to_markdown, description="Converts a CSV file to markdown format. Args: file_path: str, the path to the CSV file. Returns: str, the markdown formatted string"
)


# def create_code_tool(df: pd.DataFrame):
#     """Creates a code execution tool with the provided DataFrame"""
#     return LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))

__all__ = [
    "csv_to_markdown",
    # "create_code_tool"
]