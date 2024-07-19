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

# Configuración de Streamlit / Streamlit Configuration
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

# Carga y muestra el logo de la aplicación / Load and show the application logo
logo = Image.open('img/logo.png')
st.image(logo, width=250)

# Título y descripción de la aplicación / Application title and description
st.title("PDFMaster: Tu asistente de documentos PDF")
st.write("""
    Con PDFMaster, puedes convertir tus documentos PDF en conversaciones interactivas.
    No más lecturas aburridas o búsquedas tediosas. Haz preguntas directamente a tus documentos
    y obtén respuestas inmediatas gracias a la tecnología de chatGPT.
    """)

# Cargar API Key / Load API Key
# Intenta cargar la API Key desde st.secrets / Try to load API Key from st.secrets
API_KEY = st.secrets.get('API_KEY')

# Si la API Key no está en st.secrets, pídela al usuario / If API Key is not in st.secrets, ask the user
if not API_KEY:
    API_KEY = st.text_input('OpenAI API Key', type='password')

# Si no se ha proporcionado la API Key, no permitas que el usuario haga nada más / If API Key is not provided, do not allow the user to do anything else
if not API_KEY:
    st.stop()

# Cargar PDF / Load PDF
pdf_obj = st.file_uploader(
    "Carga tu documento / Upload your document", type="pdf")

# Si no se ha cargado un PDF, no permitas que el usuario haga nada más / If a PDF has not been uploaded, do not allow the user to do anything else
if not pdf_obj:
    st.stop()

# Función para crear embeddings / Function to create embeddings


@st.cache_resource
def create_embeddings(pdf):
    # Leer PDF / Read PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Dividir texto en fragmentos / Split text into fragments
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Crear embeddings / Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base, text


# Principal / Main
if pdf_obj:
    # Crear embeddings / Create embeddings
    knowledge_base, text = create_embeddings(pdf_obj)

    # Detectar idioma / Detect language
    lang = detect(text)
    # Solo considera inglés y español / Only considers English and Spanish
    lang = 'en' if lang != 'es' else 'es'

    # Seleccionar idioma / Select language
    st.sidebar.header('Language' if lang == 'en' else 'Idioma')
    lang = st.sidebar.radio(
        "", ['English', 'Español'], index=0 if lang == 'en' else 1)
    lang = 'en' if lang == 'English' else 'es'

    # Opciones de usuario / User options
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

    # Preguntar / Ask questions
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
            os.environ["OPENAI_API_KEY"] = API_KEY
            docs = knowledge_base.similarity_search(user_question, 10)
            llm = ChatOpenAI(model_name='gpt-4o-mini')
            chain = load_qa_chain(llm, chain_type="stuff")
            respuesta = chain.run(input_documents=docs, question=user_question)
            st.write(respuesta)
    else:
        st.info("This option will be implemented soon" if lang ==
                'en' else "Esta opción se implementará próximamente")

    # Footer / Pie de página
    st.sidebar.markdown('---')
    st.sidebar.subheader('Created by' if lang == 'en' else 'Creado por:')
    st.sidebar.markdown('Alexander Oviedo Fadul')
    st.sidebar.markdown(
        "[GitHub](https://github.com/bladealex9848) | [Website](https://alexander.oviedo.isabellaea.com/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!%20)"
    )
