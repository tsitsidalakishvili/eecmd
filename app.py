import tempfile
import plotly.express as px
import openai
import streamlit as st
import pdfplumber
import numpy as np
from io import BytesIO
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFLoader, AssemblyAIAudioTranscriptLoader
import requests
import pandas as pd

import os
import fitz
from dotenv import load_dotenv
from langchain.chains import ConversationChain, GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import streamlit_scrollable_textbox as stx
import layoutparser as lp
from rapidocr_onnxruntime import RapidOCR
import re
import pyreadstat
from neo4j import GraphDatabase
from langchain.llms import OpenAI
from langchain.graphs import Neo4jGraph

# Initialize the session state for the API key if it's not already set
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Create an input field for the API key
api_key_input = st.text_input("Enter your OpenAI API Key:", type="password")

# Button to save the API key into the session state
if st.button('Save API Key'):
    st.session_state.api_key = api_key_input
    st.success("API Key saved!")

if st.session_state.api_key != "":
    # Initialize your OpenAI or other API-dependent services with st.session_state.api_key
    # For example, if using OpenAI:
    openai.api_key = st.session_state.api_key

    # Now, you can proceed with the rest of your app's functionality
    st.write("API Key is set. You can now use the app's features.")

    # Initialize OpenAI chat model
    chat_model = ChatOpenAI(api_key=st.session_state.api_key)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=chat_model, memory=memory)

# Initialize session_state.data for prompt templates if not already initialized
if 'data' not in st.session_state:
    st.session_state.data = {'prompt_templates': []}











class RapidOCRTextExtractor:
    def __init__(self, pdf_bytes):
        self.pdf_bytes = pdf_bytes
        self.engine = RapidOCR()  # Initialize the RapidOCR engine

    def extract_text(self):
        # Convert PDF bytes to images and then extract text using RapidOCR
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        text_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            image = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)
            result, _ = self.engine(image)  # Use RapidOCR for text extraction
            page_text = ' '.join([res[1] for res in result])
            text_content.append(page_text)

        return text_content

class UnstructuredPDFReader:
    def __init__(self, pdf_bytes, extract_images=True, mode="elements"):
        self.pdf_bytes = pdf_bytes
        self.extract_images = extract_images
        self.mode = mode

    def extract_text(self):
        # Save the bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(self.pdf_bytes)
            file_path = tmp.name

            loader = UnstructuredPDFLoader(file_path, extract_images=self.extract_images, mode=self.mode)
            data = loader.load()

            # Extract 'page_content' values and return them
            text_content = [document.page_content for document in data if not isinstance(document, str) and hasattr(document, 'page_content')]

        return text_content

class PDFPlumberTextExtractor:
    def __init__(self, pdf_bytes):
        self.pdf_bytes = pdf_bytes

    def extract_text(self):
        pdf_file = BytesIO(self.pdf_bytes)
        pdf = pdfplumber.open(pdf_file)
        text_content = []

        for page in pdf.pages:
            text = page.extract_text()
            text_content.append(text)

        return text_content

class PyMuPDFTextExtractor:
    def __init__(self, pdf_bytes):
        self.pdf_bytes = pdf_bytes

    def extract_text(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        text_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_content.append(text)

        return text_content

def extract_text(pdf_bytes, extraction_method):
    if extraction_method == "Unstructured PDFLoader":
        extractor = UnstructuredPDFReader(pdf_bytes)
    elif extraction_method == "PDFPlumber":
        extractor = PDFPlumberTextExtractor(pdf_bytes)
    elif extraction_method == "PyMuPDF":
        extractor = PyMuPDFTextExtractor(pdf_bytes)
    elif extraction_method == "RapidOCRTextExtractor":
        extractor = RapidOCRTextExtractor(pdf_bytes)
    else:
        raise ValueError("Invalid extraction method selected")

    return extractor.extract_text()


def extract_specific_part(extracted_text, prefix_or_regex):
    matching_text = []
    try:
        pattern = re.compile(prefix_or_regex, re.IGNORECASE)
    except re.error as e:
        return [f"Invalid regex pattern: {str(e)}"]

    for line in extracted_text:
        if re.search(pattern, line):
            matching_text.append(line)

    return matching_text

def initialize_default_prompt_templates():
    if 'prompt_templates' not in st.session_state.data:
        st.session_state.data['prompt_templates'] = []
    templates = [
        # Existing templates...
        
        # New Survey Analyzer Template
        {
            "name": "Survey Analyzer",
            "instructions": "Analyze the survey data to identify trends, patterns, and insights. Provide a summary of the key findings.",
            "example_input": "Given the survey data with questions as column names and responses from participants, identify key trends and insights.",
            "example_output": "The survey analysis indicates a strong preference for remote work among participants, with over 60% favoring it for its flexibility. A significant correlation was found between age and technology adoption, suggesting older participants are less comfortable with new tech. Actionable recommendation: Develop targeted tech training programs for older employees to enhance their comfort with new technologies.",
            "query_template": "Analyze the survey data focusing on {specific_aspect} and summarize the key findings.",
            "few_shot_count": 3
        },
        
        # More templates...
    ]
    if not st.session_state.data['prompt_templates']:
        st.session_state.data['prompt_templates'] = templates

initialize_default_prompt_templates()

# Construct Full Prompt with Dynamic Few-Shot Examples
def construct_full_prompt(template, actual_input):
    # Incorporate few-shot examples dynamically based on the template's 'few_shot_count'
    few_shot_examples = "\n".join([f"Example Input: {template['example_input']}\nExample Output: {template['example_output']}" for _ in range(template['few_shot_count'])])
    return f"{template['instructions']}\n{few_shot_examples}\n{template['query_template']}\n\n{actual_input}"

def execute_prompt(template):
    try:
        full_prompt = f"{template}\n\n{test_input}\n\n{filtered_data}"

        # Split the full_prompt into segments of appropriate length
        segments = [full_prompt[i:i+4096] for i in range(0, len(full_prompt), 4096)]

        responses = []

        for segment in segments:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": segment}]
            )
            responses.append(response.choices[0].message['content'])

        return " ".join(responses)
    except Exception as e:
        return f"Error: {e}"


    

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())  # Use .read() for BytesIO object
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving temporary file: {e}")
        return None

def create_textual_representation(df, columns):
    """Creates a simple textual representation of the DataFrame for the chatbot context."""
    text_representation = "You have selected data with the following columns: " + ", ".join(columns) + ".\n"
    text_representation += "Here's a preview of the data:\n"
    text_representation += df.head().to_string()  # Convert the DataFrame head to a string format
    return text_representation

def generate_data_summary(df):
    """Generates a summary of the dataframe for conversational context."""
    summary = "Data Summary:\n"
    # Adding general info
    summary += f"Total records: {len(df)}\n"
    # Summary statistics for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        summary += f"{col} - Mean: {df[col].mean():.2f}, Min: {df[col].min()}, Max: {df[col].max()}\n"
    # Frequency for categorical columns, limited to top 5 categories for brevity
    for col in df.select_dtypes(include=['category', 'object']).columns:
        summary += f"Top categories in {col}: \n{df[col].value_counts().head(5).to_string()}\n"
    return summary

def read_data(file_path, file_type, encoding='UTF-8'):
    try:
        if file_type == 'SPSS':
            df, meta = pyreadstat.read_sav(file_path, encoding=encoding, apply_value_formats=True, formats_as_category=True)
        elif file_type == 'Stata':
            df, meta = pyreadstat.read_dta(file_path, encoding=encoding, apply_value_formats=True, formats_as_category=True)
        else:
            st.error("Unsupported file type.")
            return None, None

        # Convert categorical data to string for compatibility with Streamlit
        for col in df.select_dtypes(['category']):
            df[col] = df[col].astype(str)
            
        return df, meta
    except Exception as e:
        st.error(f"Error reading {file_type} file: {e}")
        return None, None

def display_metadata(meta):
    if meta is not None:
        # Ensure value labels are displayed as a list of key-value pairs
        value_labels = {k: list(v.items()) for k, v in meta.variable_value_labels.items()}
        variable_info = pd.DataFrame({
            "Variable Name": meta.column_names,
            "Variable Label": [meta.column_labels[i] if i < len(meta.column_labels) else "" for i in range(len(meta.column_names))],
            "Value Labels": [value_labels.get(name, "No labels") for name in meta.column_names]
        })
        st.dataframe(variable_info)
    else:
        st.write("No metadata available.")

def plot_data(df, x_axis, y_axis=None, meta=None, x_label=None, y_label=None, plot_type="Bar Chart"):
    
    if plot_type == "Bar Chart":
        # Check if the y_axis is provided for a stacked bar chart
        if y_axis:
            # Group the data by the x_axis and stack by y_axis categories
            # This assumes that y_axis contains categorical data for stacking
            grouped = df.groupby([x_axis, y_axis]).size().reset_index(name='counts')
            fig = px.bar(grouped, x=x_axis, y='counts', color=y_axis, title=f" {x_label} / {y_label}")
            
            # Update the layout for a stacked bar chart
            fig.update_layout(barmode='stack')

            # If metadata is provided, use value labels for the x-axis tick labels
            if meta and x_axis in meta.variable_value_labels:
                value_labels = meta.variable_value_labels[x_axis]
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=list(value_labels.keys()),
                    ticktext=list(value_labels.values())
                )

            st.plotly_chart(fig)

        # If y_axis is not specified, show a message or handle accordingly
        else:
            st.write("Please specify a variable for the Y axis to create a stacked bar chart.")


def main():
    st.title("Survey Analysis Application")

    uploaded_file = st.file_uploader("Upload a data file", type=["pdf", "sav"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == 'sav':
            temp_file_path = save_uploaded_file(uploaded_file)
            df, meta = read_data(temp_file_path, "SPSS")
            if df is not None:
                # Adjusting layout for Data Preview and Metadata to be side by side
                col_data_preview, col_metadata = st.columns([7, 3])  # 70% for data preview, 30% for metadata
                
                with col_data_preview:
                    st.write("### Data Preview")
                    st.dataframe(df.head())

                with col_metadata:
                    st.write("### Metadata")
                    display_metadata(meta)

                # Interactive UI elements for data visualization moved to the main page
                st.write("## Plot Charts")
                variable_options = {f"{meta.column_labels[i]} ({meta.column_names[i]})": meta.column_names[i] for i in range(len(meta.column_names))}
                
                # Create UI for selecting variables for two plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Chart 1")
                    x_axis_label_1 = st.selectbox("Choose X axis for Chart 1", options=list(variable_options.keys()), key='x1')
                    y_axis_label_1 = st.selectbox("Choose Y axis for Chart 1", options=list(variable_options.keys()), key='y1')
                    plot_type_1 = st.selectbox("Select the type of plot for Chart 1", ["Line Chart", "Bar Chart", "Box Plot"], key='plot1')
                
                with col2:
                    st.write("### Chart 2")
                    x_axis_label_2 = st.selectbox("Choose X axis for Chart 2", options=list(variable_options.keys()), key='x2')
                    y_axis_label_2 = st.selectbox("Choose Y axis for Chart 2", options=list(variable_options.keys()), key='y2')
                    plot_type_2 = st.selectbox("Select the type of plot for Chart 2", ["Line Chart", "Bar Chart", "Box Plot"], key='plot2')
                
                # Prepare the data and plot the charts
                col3, col4 = st.columns(2)
                
                with col3:
                    plot_data(df, variable_options[x_axis_label_1], variable_options[y_axis_label_1], meta, x_axis_label_1, y_axis_label_1, plot_type_1)
                
                with col4:
                    plot_data(df, variable_options[x_axis_label_2], variable_options[y_axis_label_2], meta, x_axis_label_2, y_axis_label_2, plot_type_2)


                st.title("Use LLM")

                # Prepare options for multiselect with variable names and labels
                column_options = [f"{var} ({meta.column_labels[i]})" if meta.column_labels[i] else var for i, var in enumerate(df.columns)]
                selected_columns = st.multiselect("Select columns for analysis", options=column_options)

                # Extract original variable names from the selected options
                selected_vars = [option.split(' (')[0] for option in selected_columns]  # Assuming variable names don't contain ' ('
                selected_labels = [option[option.find("(")+1:option.find(")")] if "(" in option and ")" in option else option for option in selected_columns]

                if selected_vars:
                    # Select data based on variable names
                    selected_data = df[selected_vars].copy()

                    # Rename columns to their labels for display
                    selected_data.columns = selected_labels

                    st.write("Selected Columns Data Preview:")
                    st.dataframe(selected_data)

                    # Generate data summary for conversation context
                    data_summary = generate_data_summary(selected_data)
                    memory.save_context({"input": "Data Summary"}, {"output": data_summary})

                    # Chatbot interaction for SPSS data
                    user_question_data = st.text_input("Ask a question based on the selected data:")
                    if user_question_data:
                        chat_response_data = conversation.run(user_question_data)
                        st.text_area("Chatbot Response (Data)", chat_response_data, height=150)

        elif file_type == 'pdf':
            
            pdf_bytes = uploaded_file.read()

            # Selection of extraction method
            extraction_method = st.radio(
                "Select Text Extraction Method",
                ["Unstructured PDFLoader"]
            )

            # User choice for extraction options
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Text Extraction Options")
                extraction_option = st.radio("Select Text Extraction Option", ["Entire Text", "Specific Parts"])

            with col2:
                if extraction_option == "Specific Parts":
                    st.header("Specify a prefix or regex pattern to extract specific parts")
                    user_prefix = st.text_input("Enter a Prefix (e.g., LX or WG):")

            if extraction_option == "Entire Text":
                extracted_text = extract_text(pdf_bytes, extraction_method)
                st.header("Entire Extracted Text")
                entire_text = "\n".join(extracted_text)
                st.text_area("Text", entire_text, height=400)

            if extraction_option == "Specific Parts":
                if user_prefix:
                    extracted_text = extract_text(pdf_bytes, extraction_method)

                    matching_text = extract_specific_part(extracted_text, user_prefix)

                    if matching_text:
                        st.header(f"Extracted Text Matching the Prefix or Regex Pattern")
                        for match in matching_text:
                            st.write(match)
                    else:
                        st.write(f"No text found matching the provided prefix or regex pattern.")


            # Chatbot interaction
            user_question_pdf = st.text_input("Ask a question based on the PDF text:")
            if user_question_pdf:
                # Save PDF text to memory for chatbot reference
                memory.save_context({"input": "PDF Text"}, {"output": "\n".join(extracted_text)})

                chat_response_pdf = conversation.run(user_question_pdf)
                st.text_area("Chatbot Response (PDF)", chat_response_pdf, height=150)



    st.title("Query with Natural Language")
    
    openai_api_key = st.session_state.api_key

    if openai_api_key:
        langchain_llm = OpenAI(api_key=openai_api_key)
    else:
        st.error("Please enter your OpenAI API key to proceed.")
        st.stop()
    
    st.write("### Select a Prompt Template")
    # Ensure the templates are initialized
    initialize_default_prompt_templates()
    # Fetch the names of the templates for the dropdown
    template_names = [template["name"] for template in st.session_state.data['prompt_templates']]
    selected_template_name = st.selectbox("Choose a template", options=template_names)

    # Find the selected template
    selected_template = next((template for template in st.session_state.data['prompt_templates'] if template["name"] == selected_template_name), None)

    if selected_template:
        st.write(f"### {selected_template['name']}")
        st.write(f"**Instructions:** {selected_template['instructions']}")
        
        # Depending on the template, you might need different input fields. Here's an example for a generic input.
        specific_aspect = st.text_input("Specify the aspect for analysis (leave blank if not applicable):")
        
        if st.button("Run Analysis"):
            if not specific_aspect:
                st.error("Please specify the aspect for analysis.")
            else:
                # Construct the query based on the template and the user's input
                query = selected_template["query_template"].format(specific_aspect=specific_aspect)
                # Example of running a query - replace with actual logic to execute the query
                result = f"Running analysis on: {query}"
                st.text_area("Result:", value=result, height=200)
    else:
        st.error("No template selected or templates are not loaded correctly.")



if __name__ == "__main__":
    main()




