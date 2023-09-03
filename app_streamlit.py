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

# Configuración de Streamlit
st.set_page_config('PDFMaster')
st.title("PDFMaster: Tu asistente de documentos PDF")

# Cargar API Key
# Intenta cargar la API Key desde st.secrets
API_KEY = st.secrets.get('API_KEY')

# Si la API Key no está en st.secrets, pídela al usuario
if not API_KEY:
    API_KEY = st.text_input('OpenAI API Key', type='password')

# Si no se ha proporcionado la API Key, no permitas que el usuario haga nada más
if not API_KEY:
    st.stop()

# Cargar PDF
pdf_obj = st.file_uploader(
    "Carga tu documento / Upload your document", type="pdf")

# Si no se ha cargado un PDF, no permitas que el usuario haga nada más
if not pdf_obj:
    st.stop()

# Función para crear embeddings


@st.cache_resource
def create_embeddings(pdf):
    # Leer PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Dividir texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base, text

# Función para generar resumen


def generate_summary(text, lang):
    import openai

    openai.api_key = API_KEY

    # Cargar un tokenizador pre-entrenado
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Dividir el texto en tokens
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
            {"role": "user", "content": f"Please provide a summary of the following text:\n\n{segment_text}" if lang ==
             'en' else f"Por favor, proporciona un resumen del siguiente texto:\n\n{segment_text}"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.5,
        )

        summary += response['choices'][0]['message']['content'].strip()

    return summary


# Principal
if pdf_obj:
    # Crear embeddings
    knowledge_base, text = create_embeddings(pdf_obj)

    # Detectar idioma
    lang = detect(text)
    lang = 'en' if lang != 'es' else 'es'  # Solo considera inglés y español

    # Opciones de usuario
    st.sidebar.header('Options' if lang == 'en' else 'Opciones')
    options = [
        'Ask questions',
        'Generate summary',
    ] if lang == 'en' else [
        'Realizar preguntas',
        'Generar resumen',
    ]
    selected_option = st.sidebar.selectbox(
        "What do you want to do with the PDF?" if lang == 'en' else "¿Qué deseas hacer con el PDF?", options)

    # Preguntar
    if selected_option == ('Ask questions' if lang == 'en' else 'Realizar preguntas'):
        st.header("Ask Questions" if lang == 'en' else "Realizar preguntas")
        user_question = st.text_input(
            "Ask a question about your PDF:" if lang == 'en' else "Haz una pregunta sobre tu PDF:")

        if user_question:
            os.environ["OPENAI_API_KEY"] = API_KEY
            docs = knowledge_base.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)
            st.write(respuesta)

    # Generar resumen
    elif selected_option == ('Generate summary' if lang == 'en' else 'Generar resumen'):
        st.header("Generate Summary" if lang == 'en' else "Generar resumen")
        resumen = generate_summary(text, lang)
        st.write(resumen)
