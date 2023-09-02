import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from transformers import BertTokenizer
from langdetect import detect
from translate import Translator
import openai

# Detect user language
user_input = st.text_input("Please enter any question or comment:")
if user_input:
    language = detect(user_input)
    translator = Translator(to_lang=language)

    # Translate the title and interface texts to the detected language
    translated_title = translator.translate(
        "PDFMaster: Your PDF Document Assistant")
    st.title(translated_title)

# Try to load the API Key from st.secrets
API_KEY = st.secrets.get('API_KEY')

# If the API Key is not in st.secrets, ask the user for it
if not API_KEY:
    translated_text = translator.translate('OpenAI API Key')
    API_KEY = st.text_input(translated_text, type='password')

# If the API Key has not been provided, do not allow the user to do anything else
if not API_KEY:
    st.stop()

translated_text = translator.translate("Upload your document")
pdf_obj = st.file_uploader(translated_text, type="pdf")

# If a PDF has not been uploaded, do not allow the user to do anything else
if not pdf_obj:
    st.stop()

# Function to create text embeddings from a PDF


@st.cache_resource
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base, text

# Function to generate a summary of a text


def generate_summary(text):
    openai.api_key = API_KEY or st.secrets["API_KEY"]

    # Load a pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split the text into tokens
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    chunk_size = 4000
    segments = [tokens[i:i + chunk_size]
                for i in range(0, num_tokens, chunk_size)]
    summary = ""

    for segment in segments:
        segment_text = tokenizer.convert_tokens_to_string(segment)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please provide a summary of the following text:\n\n{segment_text}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )

        summary += response['choices'][0]['message']['content'].strip()

    return summary


if pdf_obj:
    translated_text = translator.translate('Options')
    st.sidebar.header(translated_text)
    options = [
        translator.translate('Ask questions'),
        translator.translate('Generate summary'),
    ]
    translated_text = translator.translate(
        "What do you want to do with the PDF?")
    selected_option = st.sidebar.selectbox(translated_text, options)

    if selected_option == options[0]:
        translated_text = translator.translate("Ask questions")
        st.header(translated_text)
        knowledge_base, _ = create_embeddings(pdf_obj)
        translated_text = translator.translate(
            "Ask a question about your PDF:")
        user_question = st.text_input(translated_text)

        if user_question:
            os.environ["OPENAI_API_KEY"] = API_KEY
            docs = knowledge_base.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)

            # Translate the answer to the detected language
            translated_answer = translator.translate(answer)
            st.write(translated_answer)

    elif selected_option == options[1]:
        translated_text = translator.translate("Generate summary")
        st.header(translated_text)
        _, text = create_embeddings(pdf_obj)
        summary = generate_summary(text)

        # Translate the summary to the detected language
        translated_summary = translator.translate(summary)
        st.write(translated_summary)
