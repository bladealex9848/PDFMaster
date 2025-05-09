import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langdetect import detect
import anthropic
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from deep_translator import GoogleTranslator
from openai import OpenAI
from openai.error import APIError, RateLimitError, InvalidRequestError

# Configuración de Streamlit
st.set_page_config(
    page_title="PDFMaster: Tu asistente de documentos PDF",
    page_icon="📄",
    initial_sidebar_state='collapsed',
    menu_items={
        'Get Help': 'https://www.isabellaea.com',
        'Report a bug': None,
        'About': "PDFMaster es una herramienta completa para gestionar documentos PDF. Permite realizar diversas tareas como convertir PDF a Word, generar resúmenes, realizar preguntas y obtener respuestas específicas de un documento, y muchas otras funcionalidades que se están desarrollando."
    }
)

# Carga y muestra el logo de la aplicación
logo = Image.open('img/logo.png')
st.image(logo, width=250)

# Título y descripción de la aplicación
st.title("PDFMaster: Tu asistente de documentos PDF")
st.write("""
    Con PDFMaster, puedes convertir tus documentos PDF en conversaciones interactivas.
    No más lecturas aburridas o búsquedas tediosas. Haz preguntas directamente a tus documentos
    y obtén respuestas inmediatas gracias a la tecnología de chatGPT.
    """)

# Función para verificar si el archivo secrets.toml existe
def secrets_file_exists():
    secrets_path = os.path.join('.streamlit', 'secrets.toml')
    return os.path.isfile(secrets_path)

# Intentar obtener las claves de API desde las variables de entorno
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')

# Si las claves de API no están en las variables de entorno, intentar obtenerlas desde st.secrets si el archivo secrets.toml existe
if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
    if secrets_file_exists():
        if not OPENAI_API_KEY:
            try:
                OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
            except KeyError:
                OPENAI_API_KEY = None

        if not ANTHROPIC_API_KEY:
            try:
                ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']
            except KeyError:
                ANTHROPIC_API_KEY = None

# Si las claves de API aún no están disponibles, pedir al usuario que las introduzca
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')

if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = st.text_input('Anthropic API Key', type='password')

# Si no se proporcionan las claves de API, mostrar un error y detener la ejecución
if not OPENAI_API_KEY:
    st.error("Por favor, proporciona la clave de API de OpenAI.")
    st.stop()

if not ANTHROPIC_API_KEY:
    st.error("Por favor, proporciona la clave de API de Anthropic.")
    st.stop()

# Inicializar el cliente de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Cargar PDF
pdf_obj = st.file_uploader("Carga tu documento", type="pdf")

# Si no se ha cargado un PDF, no permitas que el usuario haga nada más
if not pdf_obj:
    st.warning('Por favor, carga un documento PDF para continuar.')
    st.stop()

# Función para crear embeddings
@st.cache_resource
def create_embeddings(pdf):
    try:
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

        return knowledge_base, text, pdf_reader

    except Exception as e:
        st.error(f"Error al crear los embeddings: {e}")
        st.stop()

# Función para evaluar la respuesta utilizando la API de Anthropic
def evaluate_response(question, document_content, response):
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        evaluation = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"Pregunta: {question}\nContenido del documento: {document_content}\nRespuesta: {response}\n\nPor favor, evalúa la calidad de la respuesta y proporciona comentarios sobre cómo mejorarla."}
            ]
        )

        # Procesar la evaluación recibida de Anthropic
        evaluation_content = ""
        for content_block in evaluation.content:
            if content_block.type == "text":
                evaluation_content += content_block.text
            elif content_block.type == "code":
                evaluation_content += f"\n```\n{content_block.code}\n```\n"

        return evaluation_content

    except Exception as e:
        st.error(f"Error al evaluar la respuesta con Anthropic: {e}")
        return "No se pudo evaluar la respuesta."

def generate_ideal_response(question, document_content):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(question, 10)

    try:
        response = client.chat.completions.create(
            model='gpt-4.1-nano',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )

        response_content = response.choices[0].message.content
        evaluation = evaluate_response(question, document_content, response_content)

        while "mejorar" in evaluation.lower() or "revisar" in evaluation.lower():
            st.info("La IA está trabajando para mejorar la respuesta. Por favor, espere...")
            prompt = f"Pregunta: {question}\nContenido del documento: {document_content}\nRespuesta: {response_content}\nEvaluación: {evaluation}\n\nGenera una respuesta mejorada basada en la evaluación proporcionada."

            response = client.chat.completions.create(
                model='gpt-4.1-nano',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            response_content = response.choices[0].message.content
            evaluation = evaluate_response(question, document_content, response_content)

    except Exception as e:
        st.error(f"Error al interactuar con la API de OpenAI: {e}")
        response_content = "Lo siento, no se pudo generar una respuesta en este momento."

    return response_content


def extract_text():
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    docs = [{"page_content": text}]
    llm = ChatOpenAI(model_name='gpt-4.1-nano')
    chain = load_qa_chain(llm, chain_type="stuff")

    question = "Por favor, genera un resumen conciso del texto proporcionado."
    summary = chain.run(input_documents=docs, question=question)
    return summary

def translate_text(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    translated_text = translator.translate(text)
    return translated_text

def generate_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph(text, styles['Normal'])]
    doc.build(elements)

    return buffer

# Principal
if pdf_obj:
    # Crear embeddings
    knowledge_base, text, pdf_reader = create_embeddings(pdf_obj)

    # Detectar idioma
    lang = detect(text)
    # Solo considera inglés y español
    lang = 'en' if lang != 'es' else 'es'

    # Seleccionar idioma
    st.sidebar.header('Language' if lang == 'en' else 'Idioma')
    lang = st.sidebar.radio(
        "", ['English', 'Español'], index=0 if lang == 'en' else 1)
    lang = 'en' if lang == 'English' else 'es'

    # Opciones de usuario
    st.sidebar.header('Options' if lang == 'en' else 'Opciones')
    options = [
        'Ask questions',
        'Extract text',
        'Summarize text',
        'Translate text',
    ] if lang == 'en' else [
        'Realizar preguntas',
        'Extraer texto',
        'Resumir texto',
        'Traducir texto',
    ]
    selected_option = st.sidebar.selectbox(
        "What do you want to do with the PDF?" if lang == 'en' else "¿Qué deseas hacer con el PDF?", options)

    # Selector de modelo de IA en el menú izquierdo
    st.sidebar.header("Modelo de IA")
    ia_model = st.sidebar.radio("Selecciona el modelo de IA", ("OpenAI", "Anthropic"))

    # Preguntar
    if selected_option == ('Ask questions' if lang == 'en' else 'Realizar preguntas'):
        st.header("Ask Questions" if lang == 'en' else "Realizar preguntas")

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
            st.info("Generando respuesta...")
            respuesta = generate_ideal_response(user_question, text)
            st.write(respuesta)

    # Extraer texto
    elif selected_option == ('Extract text' if lang == 'en' else 'Extraer texto'):
        st.header("Extract Text" if lang == 'en' else "Extraer texto")
        extracted_text = extract_text()
        st.write(extracted_text)

        # Descargar texto extraído como PDF
        if st.button("Download as PDF" if lang == 'en' else "Descargar como PDF"):
            pdf_buffer = generate_pdf(extracted_text)
            st.download_button(
                label="Download PDF" if lang == 'en' else "Descargar PDF",
                data=pdf_buffer,
                file_name="extracted_text.pdf",
                mime="application/pdf"
            )

    # Resumir texto
    elif selected_option == ('Summarize text' if lang == 'en' else 'Resumir texto'):
        st.header("Summarize Text" if lang == 'en' else "Resumir texto")
        text_to_summarize = extract_text()
        summary = summarize_text(text_to_summarize)
        st.write(summary)

    # Traducir texto
    elif selected_option == ('Translate text' if lang == 'en' else 'Traducir texto'):
        st.header("Translate Text" if lang == 'en' else "Traducir texto")
        text_to_translate = extract_text()

        target_lang = st.selectbox(
            "Select target language" if lang == 'en' else "Seleccionar idioma de destino",
            ('en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'hi', 'ar')
        )

        if st.button("Translate" if lang == 'en' else "Traducir"):
            translated_text = translate_text(text_to_translate, target_lang)
            st.write(translated_text)

            # Descargar texto traducido como PDF
            if st.button("Download as PDF" if lang == 'en' else "Descargar como PDF"):
                pdf_buffer = generate_pdf(translated_text)
                st.download_button(
                    label="Download PDF" if lang == 'en' else "Descargar PDF",
                    data=pdf_buffer,
                    file_name="translated_text.pdf",
                    mime="application/pdf"
                )

# Footer / Pie de página
    st.sidebar.markdown('---')
    st.sidebar.subheader('Created by' if lang == 'en' else 'Creado por:')
    st.sidebar.markdown('Alexander Oviedo Fadul')
    st.sidebar.markdown(
        "[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)"
    )