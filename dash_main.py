"""
Run the dashboard agent wihtout stream, only the final result

"""

import asyncio
from autogen import config_list_from_json
import pandas as pd
from autogen_agentchat.conditions import SourceMatchTermination, MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dash_agent import AnalystAgent, VisulizationSuggestionAgent, VisualizationCoderAgent
import os
from dotenv import load_dotenv
from logging_config import get_logger

code_list= []
logger = get_logger(__name__)

async def analyze_dataset(file_path: str) -> dict:
    """
    Analyze a dataset and return visualization code
    """
    load_dotenv()
    
    # Initialize the model client
    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment="bfsi-genai-demo-gpt-4o",
        azure_endpoint="https://bfsi-genai-demo.openai.azure.com/",
        model="gpt-4o-2024-05-13",
        api_version="2024-02-01",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY") or "",
    )
    
    # Read the data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df_5_rows = df.head(5)
    csv_string = df_5_rows.to_string(index=False)
    
    # Define agents
    analyst_agent = AnalystAgent(
        name="analyst_agent",
        model_client=model_client,
        df=df,
    )
    visualization_suggestion_agent = VisulizationSuggestionAgent(
        name="visulization_suggestion_agent",
        model_client=model_client,
        df=df,
        csv_string=csv_string
    )
    visualization_coder_agent = VisualizationCoderAgent(
        name="visualization_coder_agent",
        model_client=model_client,
        df=df,
        csv_string=csv_string
    )
    
    # Define termination condition
    termination = SourceMatchTermination(sources=["visualization_coder_agent"]) | MaxMessageTermination(25) | TextMentionTermination("TERMINATE")
    
    # Define a team with all agents
    agent_team = Swarm(
        [analyst_agent, visualization_suggestion_agent, visualization_coder_agent],
        termination_condition=termination,
    )
    
    # Run the analysis
    task = f"""Analyze the dataset in the "{file_path}" file and create 2-4 visualizations for the data insights."""
    task_result = await agent_team.run(task=task)
    
    logger.info(f"Task result: {task_result}")
    
    # Extract the final visualization code from the result
    messages = task_result.messages
    final_message = next((msg.content for msg in reversed(messages) 
                         if hasattr(msg, 'content') and isinstance(msg.content, str)), "No visualization code generated")
    
    logger.info(f"Final message: {final_message}")
    code_list = (final_message)
    return {"analysis": final_message, "code_list": code_list}

def run_analysis(file_path: str) -> dict:
    """
    Synchronous wrapper for analyze_dataset
    """
    logger.info(f"Running analysis for file: {file_path}")
    result = asyncio.run(analyze_dataset(file_path))
    return {"analysis": result, "code_list": code_list}

def run_analysis_structured(file_path: str) -> dict:
    """
    Run the analysis and return a dictionary with the analysis results
    """
    result = run_analysis(file_path)
    return {"analysis": result, "code_list": code_list}

if __name__ == "__main__":
    # Example usage
    result = run_analysis("mba_decision_dataset-mini.csv")
    logger.info(f"\n--Final Result--\n", result)
