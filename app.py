import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from pdf2docx import Converter
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
        'Convertir a Word',
        'Generar resumen',
        # 'Comprimir PDF',
        # 'Aplicar OCR',
    ]
    selected_option = st.sidebar.selectbox(
        "¿Qué deseas hacer con el PDF?", options)

    if selected_option == 'Realizar preguntas':
        st.header("Realizar preguntas")
        knowledge_base, _ = create_embeddings(pdf_obj)
        
        # Definir los prompts y sus descripciones
        prompts = [
            ("""Realiza un control de legalidad sobre una solicitud de vigilancia judicial administrativa recibida por el Consejo Seccional de la Judicatura de Sucre - Colombia. Evalúa la solicitud basándote en los criterios del ACUERDO No. PSAA11-8716, incluyendo competencia territorial, legitimidad del interesado, completitud y claridad de la solicitud, y su conformidad con los procedimientos y requisitos establecidos en el acuerdo. Presenta tu análisis en una tabla con las columnas 'Criterio', 'Descripción', 'Aplicación al Caso Hipotético', considerando si cada aspecto cumple o no con los requisitos, e incluye observaciones o recomendaciones pertinentes. Asegúrate de verificar si la solicitud pertenece a la jurisdicción territorial de Sucre, si el solicitante tiene un interés legítimo, y si la solicitud es completa y clara.

Extrae y resume la información relevante de una solicitud hipotética, incluyendo datos del solicitante (rol, identificación, dirección, contacto), detalles del proceso judicial (ubicación, tipo, número de radicado), motivos de la solicitud (incumplimientos, demoras), descripción breve de los hechos y anexos proporcionados. Presenta esta información de forma clara y concisa, asegurando que captures todos los elementos esenciales para entender la naturaleza y el propósito de la solicitud.

Importante que Posteriormente a la tabla de control de legalidad te Enfóques en extraer y resumir la información esencial: identifica y organiza los datos del solicitante, como su rol, identificación, dirección y contacto; detalla los aspectos críticos del proceso judicial mencionado, incluyendo la ubicación, tipo y número de radicado; resume los motivos de la solicitud, destacando las principales quejas o incidencias, como incumplimientos o demoras; y describe brevemente los hechos presentados y los anexos proporcionados. Asegura una presentación coherente y estructurada de esta información para facilitar una comprensión clara del caso.

Para terminar genera un parrafo titulado problema, donde ese parrafo puede seguir esta estructura: Para generar un resumen similar basado en los ejemplos proporcionados, puedes seguir esta estructura:

1. **Identificación del Solicitante y Rol**: Inicia con la identificación del solicitante, indicando su nombre y rol si está disponible (por ejemplo, "la doctora Diana Carolina Contreras Suárez").
2. **Detalles del Proceso Judicial**: Incluye el número de radicado del proceso y la entidad judicial responsable (por ejemplo, "proceso radicado No. 707084089001-2019-00123-00, de conocimiento del Juzgado Primero Promiscuo Municipal de San Marcos").
3. **Motivo Principal de la Solicitud**: Describe brevemente la razón principal de la solicitud, enfocándote en el problema específico que se denuncia (por ejemplo, "argumentando que el 16 de mayo de 2022 fue la última actuación emitida dentro del proceso y a la fecha no se ha registrado movimiento alguno").
4. **Acciones o Documentos Presentados**: Menciona las acciones o documentos presentados por el solicitante que son relevantes para la solicitud (por ejemplo, "presentó solicitud de medidas cautelares", "memorial aportando la liquidación del crédito").
5. **Estado Actual o Demora**: Destaca el estado actual de la solicitud o la demora específica que se está denunciando (por ejemplo, "el despacho no se ha pronunciado", "no se ha registrado movimiento alguno").

Este párrafo titulado problema permitirá resumir la información relevante de manera coherente, manteniendo un enfoque claro en el solicitante, el proceso judicial implicado, el problema denunciado y cualquier acción relevante tomada por el solicitante.""", 
         "Evaluar Legalidad", 
         "Realiza una evaluación de legalidad detallada según el ACUERDO No. PSAA11-8716"),
            ("Analiza el documento cargado y genera un resumen estructurado. Comienza con un título claro que refleje el tema principal del documento. En el desarrollo del resumen, inicia especificando el remitente, su institución, el número de oficio y la fecha del documento. Luego, proporciona un resumen conciso del contenido, destacando los puntos más relevantes de manera coherente. Asegúrate de presentar la información de manera accesible, adaptando el tono del resumen según sea necesario para facilitar su comprensión.", 
         "Generar Resumen Estructurado", 
         "Genera un resumen estructurado del documento, resaltando los elementos clave.")        
        ]

        # Crear botones para los prompts con descripciones emergentes
        for index, (prompt, label, description) in enumerate(prompts):
            # Decide en qué columna colocar el botón basándose en su índice
            col = st.columns(2)[index % 1]
            with col:
                if st.button(label, help=description):
                    st.session_state['selected_prompt'] = prompt

        # Mostrar el área de texto con el prompt seleccionado o el ingresado por el usuario
        user_question = st.text_area(
            "Haz una pregunta sobre tu PDF:",
            value=st.session_state.get('selected_prompt', ''),
            height=150
        )

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
