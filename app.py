import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch API key from .env

HF_API_KEY= os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

def getllmresponse(input_text, no_words, blog_style):
    # Initialize Hugging Face Model
    llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    task='text-generation',
    huggingfacehub_api_token=HF_API_KEY)  # Pass the API token


    # Wrap it with ChatHuggingFace
    model = ChatHuggingFace(llm=llm)


    template = """
    Write a blog for a {blog_style} job profile on the topic "{input_text}"
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=['input_text', 'blog_style', 'no_words'],
        template=template
    )

    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = model.invoke(formatted_prompt)  # Corrected invocation
    return response.content


st.set_page_config(page_title='Generate Blog', layout='centered', initial_sidebar_state='collapsed')

st.header('Generate Blogs')
input_text = st.text_input('Enter the blog topic')

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of words', '100')  # Default value to avoid empty input

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button('Generate')

if submit:
    if input_text and no_words.isdigit():  # Ensuring valid input
        st.write(getllmresponse(input_text, int(no_words), blog_style))
    else:
        st.error("Please enter a valid topic and numeric word count.")
