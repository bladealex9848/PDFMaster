import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
# from pdf2docx import Converter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import io
from transformers import BertTokenizer

st.set_page_config('PDFMaster')
st.title("PDFMaster: Tu asistente de documentos PDF")

# Cargar API_KEY desde una variable de entorno, un archivo o el input del usuario
API_KEY = os.environ.get('API_KEY')
if API_KEY is None:
    try:
        with open('api_key.txt', 'r') as f:
            API_KEY = f.read().strip()
    except FileNotFoundError:
        with open('api_key.txt', 'w') as f:
            f.write('tu_clave_de_api')
        API_KEY = st.text_input('OpenAI API Key', type='password')

pdf_obj = st.file_uploader("Carga tu documento", type="pdf")


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


def convert_pdf_to_word(pdf_obj):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(pdf_obj.read())
    tfile.close()  # Asegúrate de cerrar el archivo

    # Obtener el nombre del archivo sin la extensión
    filename = os.path.splitext(pdf_obj.name)[0]
    word_filename = f"{tempfile.gettempdir()}/{filename}.docx"

    cv = Converter(tfile.name)
    cv.convert(word_filename, start=0, end=None)
    cv.close()

    os.unlink(tfile.name)  # Elimina el archivo

    with open(word_filename, 'rb') as f:
        word_file = io.BytesIO(f.read())

    os.unlink(word_filename)  # Elimina el archivo

    return word_file


def generate_summary(text):
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
            {"role": "user", "content": f"Por favor, proporciona un resumen del siguiente texto:\n\n{segment_text}"}
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
    st.sidebar.header('Opciones')
    options = [
        'Realizar preguntas',
        # 'Convertir a Word',
        'Generar resumen',
        # 'Comprimir PDF',
        # 'Aplicar OCR',
    ]
    selected_option = st.sidebar.selectbox(
        "¿Qué deseas hacer con el PDF?", options)

    if selected_option == 'Realizar preguntas':
        st.header("Realizar preguntas")
        knowledge_base, _ = create_embeddings(pdf_obj)
        user_question = st.text_input("Haz una pregunta sobre tu PDF:")

        if user_question:
            os.environ["OPENAI_API_KEY"] = API_KEY
            docs = knowledge_base.similarity_search(user_question, 3)
            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)
            st.write(respuesta)

    elif selected_option == 'Convertir a Word':
        st.header("Convertir a Word")
        if st.button("Convertir"):
            word_file = convert_pdf_to_word(pdf_obj)
            # Obtener el nombre del archivo sin la extensión
            filename = os.path.splitext(pdf_obj.name)[0]
            st.download_button('Descargar archivo Word',
                               word_file, f'{filename}.docx')

    elif selected_option == 'Generar resumen':
        st.header("Generar resumen")
        _, text = create_embeddings(pdf_obj)
        resumen = generate_summary(text)
        st.write(resumen)
