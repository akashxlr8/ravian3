"""
This is a dashboard for uploading a CSV file and generating visualizations and analysis based on the data.
No user interaction is required. All the visualizations and analysis are generated automatically.
USing Autogen Agents and Azure OpenAI API.
Multiple Agents are used to generate the visualizations and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm 
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv
import asyncio
from logging_config import get_logger
from dataclasses import dataclass
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
import re
from dash_agent import __all__ as dash_agent_all
from dash_agent import AnalystAgent, VisulizationSuggestionAgent, VisualizationCoderAgent
from pydantic import ValidationError
import json
from dash_main import run_analysis, run_analysis_structured

logger = get_logger(__name__)
# Load environment variables
load_dotenv()

# Initialize the model client
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="bfsi-genai-demo-gpt-4o",
    azure_endpoint="https://bfsi-genai-demo.openai.azure.com/",
    model="gpt-4o-2024-05-13",
    api_version="2024-02-01",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY") or "",
)


# Initialize Assistant Agent with visualization capabilities and structured output
# Assistant = AssistantAgent(
#     "Assistant",
#     model_client=model_client,
#     tools=[],
#     description="Agent to analyze CSV data and create visualizations",
#     system_message=f"""You are a data visualization expert proficient in using Plotly Express. 
#     You have access to the already loaded dataframe 'df'. Your task is to generate Python code that creates insightful visualizations based on the provided dataset. 
#     Ensure that your response is a JSON object adhering to the following Pydantic model:
#             <Pydantic Model>
#             class CodeResponse:
#                 result: CodeBlock
#                     code: str  # The Python code to execute
#                     code_type: Literal['visualization', 'analysis']  # Type of code
#                     observation: Optional[str]  # trend/pattern/insight/summary of the data
#             </Pydantic Model>
#             - The generated code should be executable and free from errors, Use pd.concat() to combine multiple dataframes.
#             - Provide a brief explanation of observed trend/pattern/insight/summary of the data in the observation field.
#     """
# )

# Set up termination conditions
# termination = HandoffTermination(target="user") |TextMentionTermination("TERMINATE")
# team = Swarm([AnalysisAgent], termination_condition=termination, max_turns=20)

# async def get_visualization_code(df, query, csv_string):
    # """Get visualization code from the agent using structured output"""
    # task = f"""Given this dataframe with columns
    # <Columns>
    # {list(df.columns)}
    # </Columns>
    # <Query>
    # {query}
    # </Query>
    # Your response should be structured as specified.
    # IMPORTANT: The data is already loaded as 'df'. DO NOT include pd.read_csv().
    # First 5 rows of data: 
    # <Data>
    # {csv_string}
    # </Data>
    
    # Focus on the mentioned columns in the query to derive a correlation and plot the graph using Plotly.
    # Ensure the code is structured as a JSON object adhering to the following Pydantic model:
    # {{
    #     "result": {{
    #         "code": "import plotly.express as px\\n fig = px.scatter(df, x='Undergrad University Ranking', y='Expected Post-MBA Salary', title='Effect of Undergrad University Ranking on Expected Post-MBA Salary')", 
    #         "code_type": "visualization", 
    #         "observation": "This code creates a scatter plot showing the relationship between undergraduate university ranking and expected post-MBA salary. The plot shows a positive correlation between the two variables, indicating that higher ranked universities tend to have higher expected post-MBA salaries."
    #     }}
    # }}
    # Note:
    # - Exclude fig.show() at the end of the code
    # - Respond with a raw JSON object without any markdown formatting.
    # - Common visualization types:
    # -- px.bar → categorical comparisons (e.g., count of people in each major)
    # -- px.scatter → correlations (e.g., GPA vs. Salary)
    # -- px.line → trends over time (if applicable)
    # -- px.box → distributions (e.g., salary distribution by major)
    # """
    
    # logger.debug(f"Running task: {task}")
    # task_result = await Assistant.run(task=task)
    
    # messages = task_result.messages
    # response_content = next((msg.content for msg in reversed(messages) 
    #                        if hasattr(msg, 'content') and isinstance(msg.content, str)), 
    #                        None)
    
    # if response_content:
    #     try:
    #         # Clean up the response content by removing markdown formatting
    #         cleaned_content = response_content
    #         if "```" in cleaned_content:
    #             # Extract content between triple backticks
    #             cleaned_content = cleaned_content.split("```")[1]
    #             if cleaned_content.startswith("json"):
    #                 cleaned_content = cleaned_content[4:]  # Remove "json" prefix
    #             cleaned_content = cleaned_content.strip()
            
    #         # Parse the cleaned response into our Pydantic model
    #         structured_response = CodeResponse.model_validate_json(cleaned_content)
    #         logger.debug(f"Structured response: {structured_response}")
    #         return structured_response
    #     except Exception as e:
    #         logger.error(f"Error parsing structured response: {e}")
    #         # Fallback to basic parsing if structured parsing fails
    #         code = response_content.strip()
    #         if code.startswith("```python"):
    #             code = code.replace("```python", "").replace("```", "").strip()
    #         elif code.startswith("```"):
    #             code = code.replace("```", "").strip()
            
    #         # Create a CodeResponse object with the fallback parsing
    #         return CodeResponse(
    #             result=CodeResponse.CodeBlock(
    #                 code=code,
    #                 code_type='visualization' if 'fig' in code else 'analysis',
    #                 observation="Generated from fallback parsing"
    #             )
    #         )
    
    # return None

def extract_figure_names(code):
    """Extracts figure variable names from the given code."""
    # Regex to find variable names assigned to Plotly figures
    figure_names = re.findall(r'(\w+)\s*=\s*px\.', code)
    return figure_names

def extract_code_response(result_text: str) -> list:
    """Extract code response from the result text"""
    try:
        # Find the JSON content within the text
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            logger.error("No JSON content found in result")
            return []
            
        json_str = result_text[json_start:json_end]
        json_data = json.loads(json_str)
        
        # Get the code response list
        code_responses = json_data.get('code_response_list', [])
        logger.debug(f"Extracted code responses: {code_responses}")
        return code_responses
        
    except Exception as e:
        logger.error(f"Error parsing result: {e}")
        return []

def extract_code_response_list(response_content):
    match = re.search(r'({\s*"code_response_list".*})', response_content, re.DOTALL)
    if match:
        json_data = json.loads(match.group(1))  # Convert to dictionary
    code_response_list = json_data.get("code_response_list", [])

    # Print extracted codes
    for item in code_response_list:
        print(item["code"])
    else:
        print("No valid JSON found.")

def main():
    st.title("CSV Data Analysis & Visualization")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            with open("temp.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            df = pd.read_csv("temp.csv")
            # Basic data info
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: {df.shape[0]}")
            st.write(f"Number of columns: {df.shape[1]}")
            
            # Data preview
            st.subheader("Preview of Data")
            st.dataframe(df.head())
            
            # Run the analysis
            results = run_analysis("temp.csv")
            logger.info(f"Results after analysis: {results}")
            # Get the analysis content from the nested dictionary
            analysis_content = results.get("analysis", {}).get("analysis", "")
            
            # Extract code responses from the analysis content
            # code_responses = results.get("code_list", [])
            code_responses = extract_code_response(str(analysis_content))
            logger.info(f"Code responses after extraction: {code_responses}")
            
            # Display each visualization
            for i, response in enumerate(code_responses, 1):
                st.subheader(f"Visualization {i}")
                
                # Display code
                st.code(response['code'], language='python')
                
                # Display observation
                if response.get('observation'):
                    st.write("**Observation:**", response['observation'])
                
                # Execute and display visualization
                try:
                    local_vars = {'df': pd.read_csv("temp.csv"), 'pd': pd, 'px': px}
                    exec(response['code'], globals(), local_vars)
                    if 'fig' in local_vars:
                        st.plotly_chart(local_vars['fig'])
                except Exception as e:
                    st.error(f"Error executing visualization {i}: {e}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()