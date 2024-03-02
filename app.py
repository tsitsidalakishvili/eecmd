import streamlit as st
import pandas as pd
import pyreadstat
import tempfile
import os
import plotly.express as px
import openai

# Initialize session state for prompt templates and selected columns
if 'prompt_templates' not in st.session_state:
    st.session_state.prompt_templates = []
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

def initialize_default_prompt_templates():
    templates = [
        {
            "name": "General Analysis",
            "instructions": "Analyze the data to identify key trends and insights.",
            "query_template": "Analyze the data focusing on {columns}.",
        }
        # Add more templates as needed
    ]
    if not st.session_state.prompt_templates:
        st.session_state.prompt_templates = templates

initialize_default_prompt_templates()

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving temporary file: {e}")
        return None

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
        st.write("### Variable Information")
        st.dataframe(variable_info)
    else:
        st.write("No metadata available.")

def plot_data(df, x_axis, y_axis=None, meta=None, x_label=None, y_label=None):
    # Define the plot type outside of this function or pass it as an argument
    plot_type = st.selectbox("Select the type of plot", ["Line Chart", "Bar Chart", "Box Plot"])
    
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


def construct_prompt_with_data(template, filtered_df):
    columns_description = ", ".join(filtered_df.columns)
    full_prompt = template['query_template'].format(columns=columns_description)
    return full_prompt




def execute_prompt(template, filtered_df):
    try:
        # Set the OpenAI API key from the user input
        openai.api_key = st.session_state.api_key
        
        filtered_data = filtered_df.to_json(orient='records')
        message = {
            "role": "user",
            "content": f"{template['instructions']}\nData: {filtered_data}\n\nPrompt: {template['query_template']}"
        }
        chat_messages = [message]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=chat_messages
        )
        if response['choices'] and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            return "No response from model."
    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("Dataset Reader and Visualizer")
    
    # Initialize df as None to ensure it's always defined
    df = None
    
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=["sav", "dta"])
    
    if uploaded_file is not None:
        temp_file_path = save_uploaded_file(uploaded_file)
        if temp_file_path:
            file_type = 'SPSS' if uploaded_file.name.endswith('.sav') else 'Stata'
            df, meta = read_data(temp_file_path, file_type)
            os.unlink(temp_file_path)
            if df is not None:
                st.write(f"### {file_type} Dataset Information")
                st.write(df)
                display_metadata(meta)
                
                variable_options = {f"{meta.column_labels[i]} ({meta.column_names[i]})": meta.column_names[i] for i in range(len(meta.column_names))}
                x_axis_label = st.sidebar.selectbox("Choose X axis", options=list(variable_options.keys()))
                y_axis_label = st.sidebar.selectbox("Choose Y axis", options=list(variable_options.keys()))
                x_axis = variable_options[x_axis_label]
                y_axis = variable_options[y_axis_label]
                x_label = x_axis_label.split('(')[0].strip()
                y_label = y_axis_label.split('(')[0].strip()
                plot_data(df, x_axis, y_axis, meta, x_label, y_label)

    # Now 'df' is guaranteed to be defined (though it may be None)
    if df is not None:
        selected_columns = st.sidebar.multiselect("Select columns:", df.columns.tolist(), key="selected_columns")
        if selected_columns:
            filtered_df = df[selected_columns]
            
            st.sidebar.write("### Manage Prompt Templates")
            template_index = st.sidebar.selectbox("Choose a template", range(len(st.session_state.prompt_templates)), format_func=lambda x: st.session_state.prompt_templates[x]['name'])
            # API key input moved here to ensure it's always accessible
            api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", key="api_key")
            st.session_state.api_key = api_key if api_key else st.session_state.api_key
            
            instructions = st.text_area("Instructions:", value=st.session_state.prompt_templates[template_index]['instructions'], height=100)
            query_template = st.text_area("Query Template:", value=st.session_state.prompt_templates[template_index]['query_template'], height=100)
            
            st.session_state.prompt_templates[template_index]['instructions'] = instructions
            st.session_state.prompt_templates[template_index]['query_template'] = query_template
            
            selected_template = st.session_state.prompt_templates[template_index]

            if st.button("Execute Prompt with Selected Data"):
                response = execute_prompt(selected_template, filtered_df)
                st.write(response)


if __name__ == "__main__":
    main()
