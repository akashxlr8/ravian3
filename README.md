# CSV Data Analysis & Visualization

This application allows users to upload a CSV file, analyze its contents, and generate insightful visualizations using Plotly Express. The app is built with Streamlit and leverages an AI assistant to generate Python code for visualizations based on user queries.

## Features

- Upload and preview CSV files
- Generate visualizations based on the dataset
- Display generated Python code and visualizations
- Supports various types of visualizations including scatter plots, bar charts, histograms, and more

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/akashxlr8/ravian2.git
    cd ravian2
    ```

2. Create and activate a virtual environment:
    ```sh
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory and add the following variables:
        ```
        AZURE_OPENAI_DEPLOYMENT=<your-azure-openai-deployment>
        AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
        AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
        ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run ui_csv_upload_get_dash.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a CSV file to generate visualizations.

## Sample Datasets to test app

The repository includes sample datasets in the `sample_data_to_test` directory:

1. `mba_decision_dataset.csv`: A dataset containing information about students' decisions to pursue an MBA
    - Features: Age, Work Experience, Gender, Family Background, Academic Performance
    - Target Variable: MBA Decision (Yes/No)

Similarly, there are different datasets in the folder.

The datasets are clean and ready for analysis, making them ideal for testing the application's visualization capabilities.

## File Structure

- `ui_csv_upload_get_dash.py`: Main application file
- `dash_agent.py`: Contains the agent classes for data analysis, visualization suggestions, and code generation.
- `dash_tools.py`: Contains utility functions and tools used by the agents.
- `dash_main.py`: Main script to run the dashboard agent without Streamlit, only the final result.
- `logging_config.py`: Logging configuration
- `logs/`: Directory for log files
- `requirements.txt`: List of dependencies
- `.env`: Environment variables (not included in the repository)
- `.gitignore`: Git ignore file
- `secrets.toml`: Streamlit secrets configuration

## License

This project is licensed under the MIT License.