from typing import Any, Awaitable, Callable, List, Optional, Literal
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.tools import Tool
import pandas as pd
from pydantic import BaseModel, Field
from dash_tools import csv_to_markdown
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

# First, define your response format type
class ResponseFormat(BaseModel):
    type: Literal["json_object", "text"] = "json_object"
    schema: dict = Field(default_factory=dict)

# Then modify the client initialization
model_client = AzureOpenAIChatCompletionClient(
    response_format={"type": "json_object"},  # Remove schema parameter
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    model="gpt-4o-2024-05-13",
    api_version="2024-02-01",
    api_key=st.secrets["AZURE_OPENAI_API_KEY"] or ""
)

class AnalysisItem(BaseModel):
    """Single analysis with its details"""
    analysis_type: str = Field(..., description="Type of analysis: trend, pattern, insight, summary")
    analysis_title: str = Field(..., description="Title of the analysis")
    analysis_description: str = Field(..., description="Description of the analysis")
    observations: List[str] = Field(..., description="Observations about the data")
    data_source: str = Field(..., description="Source of the data")
    data_columns_to_use: List[str] = Field(..., description="Columns of the data to be used in the analysis")

class Analysis(BaseModel):
    """Container for multiple AnalysisItems"""
    analyses: List[AnalysisItem] = Field(..., 
        description="List of AnalysisItems generated from the data")

    @classmethod
    def from_json(cls, json_data: dict) -> "Analysis":
        """Create an Analysis instance from a JSON object."""
        return cls(**json_data)

class VisualizationSuggestionItem(BaseModel):
    """Single visualization suggestion with its details"""
    visualization_type: str = Field(..., description="Type of visualization: bar, scatter, line, box, pie, histogram, area, heatmap, line_polar, line_geo, line_mapbox, line_choropleth, line_mapbox")
    visualization_title: str = Field(..., description="Title of the visualization")
    visualization_description: str = Field(..., description="Description of the visualization")
    observations: List[str] = Field(..., description="Observations about the data, Keep the same as received from analyst_agent")
    data_source: str = Field(..., description="Source of the data, Keep the same as received from analyst_agent")
    data_columns_to_use: List[str] = Field(..., description="Columns of the data to be used in the visualization")

class VisualizationSuggestion(BaseModel):
    """Container for multiple visualization suggestions"""
    suggestions: List[VisualizationSuggestionItem] = Field(..., 
        description="List of visualization suggestions based on the analysis")

    @classmethod
    def from_json(cls, json_data: dict) -> "VisualizationSuggestion":
        """Create a VisualizationSuggestions instance from a JSON object."""
        return cls(**json_data)

class CodeResponseItem(BaseModel):
    """
    This is a Pydantic model for a single code response item.
    """
    code: str = Field(..., description="The Python code to be executed")
    code_type: str = Field(..., description="Type of code: visualization type")
    observation: Optional[str] = Field(None, description="Explanation of the trend/pattern/insight/summary of the data, Keep the same as received from analyst_agent")

class CodeResponse(BaseModel):
    """
    This is a Pydantic model for code response. To be used in the code generation agent and UI.
    """
    code_response_list: List[CodeResponseItem] = Field(..., description="List of code response items")

    @classmethod
    def from_json(cls, json_data: dict) -> "CodeResponse":
        """Create a CodeResponse instance from a JSON object."""
        return cls(**json_data)

# class DashboardAgent(AssistantAgent):
#     def __init__(
#         self,
#         name: str,
#         model_client: AzureOpenAIChatCompletionClient,
#         *,
#         df: pd.DataFrame = pd.DataFrame(),
#         tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
#         description: str = "An agent that takes csv file and passes markdown table to analyst_agent",
#         system_message: str | None = None
#     ):
#         if system_message is None:
#             system_message = """You are a helpful agent that is an expert at transforming csv file to markdown table.
#             Use the csv_to_markdown tool to transform the csv file to markdown table.
#             - Hand off to analyst_agent for data analysis.
#             """
        
#         super().__init__(
#             name=name,
#             model_client=model_client,
#             handoffs=["analyst_agent"],
#             tools=tools or [csv_to_markdown],
#             description=description,
#             # reflect_on_tool_use=True,
#             system_message=system_message,
#         )

class AnalystAgent(AssistantAgent):
    def __init__(
        self, 
        name: str, 
        model_client: AzureOpenAIChatCompletionClient = AzureOpenAIChatCompletionClient(response_format={"type": "json_object"}, azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                                                                                                    model="gpt-4o-2024-05-13",
                                                                                                    api_version="2024-02-01",
                                                                                                    temperature=0.9,
                                                                                                    api_key=st.secrets["AZURE_OPENAI_API_KEY"] or ""),
        df: pd.DataFrame = pd.DataFrame(), 
        *, 
        tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
        description: str = "An agent that takes dataframe and generates Insights, patterns and observations on the dataset",
        system_message: str | None = None
    ):
        self.df = df
        if system_message is None:
            system_message = f"""
            You are an expert at analyzing data and generating insights.
            Analyze the columns and see which columns are most relevant to the analysis, something like how a column is related to the other columns which might be missed by the analyst.
            Focus on numerical patterns, seasonal trends, and outliers, also provide the data source and the columns used to generate each insight.
            Try to be as curious as possible and generate insights that are not obvious and simple.
            You have access to the following data {df}.
            Output the analysis in the following json format:
            {Analysis.model_json_schema()}
            [
                {{
                    "analysis_type": "trend",
                    "analysis_title": "Gender Distribution",
                    "analysis_description": "There is a mix of male and female candidates, with males being slightly more represented.",
                    "observations": ["There is a mix of male and female candidates, with males being slightly more represented."],
                    "data_source": "mba_decision_dataset-mini.csv",
                    "data_columns_to_use": ["Gender"]
                }},
                {{
                    "analysis_type": "insight",
                    "analysis_title": "Preferred MBA Format",
                    "analysis_description": "On-campus MBAs are favored by individuals who seem to focus on entrepreneurship, while those opting for online programs focus more on networking or skill enhancement.",
                    "observations": ["On-campus MBAs are favored by individuals who seem to focus on entrepreneurship, while those opting for online programs focus more on networking or skill enhancement."],
                    "data_source": "mba_decision_dataset-mini.csv",
                    "data_columns_to_use": ["Online vs On-campus MBA", "Current Job Title", "Reason for MBA"]
                }},
                {{
                    "analysis_type": "pattern",
                    "analysis_title": "Undergraduate GPA vs. Desired Post-MBA Role",
                    "analysis_description": "This graph shows the relationship between undergraduate GPA and desired post-MBA role.",
                    "observations": ["Higher GPAs correlate with roles in Consulting or Marketing, while lower GPAs lean toward entrepreneurial or executive roles."],
                    "data_source": "mba_decision_dataset-mini.csv",
                    "data_columns_to_use": ["Undergraduate GPA", "Desired Post-MBA Role"]
                }}
           ]
        Hand off to visualization_suggestion_agent with the analysis details.
        """
        super().__init__(
            name="analyst_agent",
            model_client=model_client,
            handoffs=["visualization_suggestion_agent"],
            tools=tools,
            description=description,
            system_message=system_message
        )
        
class VisulizationSuggestionAgent(AssistantAgent):
    def __init__(self,
                 name: str,
                 model_client: AzureOpenAIChatCompletionClient = AzureOpenAIChatCompletionClient(response_format={"type":"json_object"},  azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                                                                                                    model="gpt-4o-2024-05-13",
                                                                                                    api_version="2024-02-01",
                                                                                                    api_key=st.secrets["AZURE_OPENAI_API_KEY"] or ""),
                 df: pd.DataFrame = pd.DataFrame(),
                 csv_string: str = "",
                 *,
                 tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
                 description: str = "An agent that takes the insights and generates visulization suggestions", 
                 system_message: str | None = None):
        
        self.df = df
        self.csv_string = csv_string
        
        if system_message is None:
            system_message = f"""You are an expert at generating visulization suggestions.
                 You are given a set of insights and you need to generate a visulization suggestion for each insight. Suggest what type of visulization to use for each insight. 
                 choose the best visulization type for each insight.
                 Do not suggest visulization for insights that are not provided. Pick the most interesting insights to suggest visulization for.
                 You have access to the following columns: {list(df.columns)}.
                 First 5 rows of data: {csv_string}
                 Use the following format for the response:
                 {VisualizationSuggestion.model_json_schema()}
                 

                 Example:
                 [
                    {{
                        "visulization_type": "pie_chart, px.pie",
                        "visulization_title": "Preferred MBA Format",
                        "visulization_description": "This pie chart shows the preferred MBA format of the candidates.",
                        "observations": ["On-campus MBAs are favored by individuals who seem to focus on entrepreneurship, while those opting for online programs focus more on networking or skill enhancement."],
                        "data_source": "mba_decision_dataset-mini.csv",
                        "data_columns_to_use": ["Online vs On-campus MBA", "Current Job Title", "Reason for MBA"]
                    }}
                 ]
                 Note: Use the following plotly express functions to generate the visulization:
                 -- px.bar → categorical comparisons (e.g., count of people in each major)
                 -- px.scatter → correlations (e.g., GPA vs. Salary)
                 -- px.line → trends over time (if applicable)
                 -- px.box → distributions (e.g., salary distribution by major)
                 -- px.pie → proportions (e.g., gender distribution)
                    -- px.histogram → frequency distribution (e.g., salary distribution by major)
                    -- px.area → trends over time (if applicable)
                    -- px.heatmap → correlations (e.g., GPA vs. Salary)
                    -- px.line_polar → trends over time (eg. salary distribution by gender)
                    -- px.line_geo → trends over time (eg. salary distribution by country)
                    -- px.line_mapbox → trends over time (eg. salary distribution by city)
                    -- px.line_choropleth → trends over time (eg. salary distribution by country)
                    -- px.line_mapbox → trends over time (eg. salary distribution by city)
                            1. Heatmaps for Correlation Analysis
                        Function: px.imshow(df.corr(), text_auto=True, color_continuous_scale='Viridis')
                        Use Case: Visualizing correlation matrices between numerical columns.
                        2. Histograms for Data Distribution
                        Function: px.histogram(df, x='Salary', nbins=20, title='Salary Distribution')
                        Use Case: Understanding distribution of numerical variables.
                        3. Treemaps for Hierarchical Data
                        Function: px.treemap(df, path=['Industry', 'Job Role'], values='Salary')
                        Use Case: Displaying nested categorical data like industry vs. salary.
                        4. Bubble Charts for Weighted Scatter Plots
                        Function: px.scatter(df, x='GPA', y='Salary', size='Work Experience', color='Industry')
                        Use Case: Showing relationships with an extra size dimension.
                        5. Sunburst Charts for Multi-Level Categories
                        Function: px.sunburst(df, path=['Country', 'State', 'City'], values='Population')
                        Use Case: Drill-down visualization for regional or hierarchical data
   
                 """
        
        super().__init__(name="visualization_suggestion_agent", 
                        model_client=model_client, 
                        tools=tools, 
                        handoffs=["visualization_coder_agent"],
                        description=description, 
                        system_message=system_message)

class VisualizationCoderAgent(AssistantAgent):
    def __init__(self, 
                 name: str,
                 model_client: AzureOpenAIChatCompletionClient = AzureOpenAIChatCompletionClient(response_format={"type":"json_object"},  azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
                                                                                                    model="gpt-4o-2024-05-13",
                                                                                                    api_version="2024-02-01",
                                                                                                    api_key=st.secrets["AZURE_OPENAI_API_KEY"] or ""), 
                 df: pd.DataFrame = pd.DataFrame(),
                 csv_string: str = "",
                 *,
                 tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
                 description: str = "An agent that takes visualization suggestions and generates visulization code for it",
                 system_message: str | None = None):
        
        self.df = df
        self.csv_string = csv_string
        
        if system_message is None:
            system_message = f"""You are a data visualization expert using plotly express.
            You have access to a DataFrame with these columns: {list(df.columns)}.
            First 5 rows of data: {csv_string}
            Return ONLY the Python code to create the visualizations.
            Use px.line, px.scatter, px.bar, or other appropriate plotly express charts.
            Do not include any explanations, just the code.
            Hand off to user one you have generated the code.
            Also respond with "TERMINATE" after the task is complete.
            Use the following format for the response:
            {CodeResponse.model_json_schema()}
            
            Example:
            [
                {{
                    "code": "import plotly.express as px\\nfig = px.scatter(df, x='Undergraduate GPA', y='Expected Post-MBA Salary', color='Desired Post-MBA Role', title='GPA vs Expected Salary by Role')",
                    "code_type": "visualization",
                    "observation": "The scatter plot shows a positive correlation between GPA and expected salary, particularly strong for consulting roles"
                }}
            ]
            Note:
            - Exclude fig.show() at the end of the code
            - Respond with a raw JSON object without any markdown formatting.
            - Common visualization types:
            -- px.bar → categorical comparisons (e.g., count of people in each major)
            -- px.scatter → correlations (e.g., GPA vs. Salary)
            -- px.line → trends over time (if applicable)
            -- px.box → distributions (e.g., salary distribution by major)
            """
        
        super().__init__(name="visualization_coder_agent", 
                        model_client=model_client, 
                        # handoffs,
                        tools=tools, 
                        description=description, 
                        system_message=system_message)

# class CodeExecutionAgent(AssistantAgent):
#     def __init__(self, name: str, model_client: AzureOpenAIChatCompletionClient, df: pd.DataFrame, *,
#                  tools: List[Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None,
#                  description: str = "An agent that executes visualization code",
#                  system_message: str | None = None):
        
#         self.df = df
#         code_tool = create_code_tool(df)  # Create the tool with the DataFrame
#         super().__init__(
#             name=name,
#             model_client=model_client,
#             tools=[code_tool],
#             reflect_on_tool_use=True,
#             description=description,
#             system_message=system_message
#         )
    
__all__ = ["AnalystAgent", "VisulizationSuggestionAgent", "VisualizationCoderAgent"]