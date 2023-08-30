# PDFMaster: Tu asistente de documentos PDF

## Descripción

PDFMaster es una herramienta completa para gestionar documentos PDF. Permite realizar diversas tareas como convertir PDF a Word, generar resúmenes, realizar preguntas y obtener respuestas específicas de un documento, y muchas otras funcionalidades que se están desarrollando.

## Introducción

Con PDFMaster, en menos de 50 líneas de código, aprenderás a aprovechar todo el potencial de chatGPT y convertir tus documentos en conversaciones interactivas. No más lecturas aburridas o búsquedas tediosas, ahora podrás hacer preguntas directamente a tus documentos y obtener respuestas de chatGPT. Para desarrollar esta aplicación necesitaremos:

- OpenAI ChatGPT API
- Streamlit

## ¿Cómo funciona?

1. Divide documento en fragmentos (o chunks)
2. Crea los embeddings de los fragmentos de texto
3. Guarda los fragmentos y los embeddings en una base de conocimiento
4. Busca los fragmentos más similares a la pregunta del usuario gracias a los embeddings.
5. Pasa los fragmentos más similares junto a la pregunta a chatGPT que genera la respuesta

## Funcionalidades

- **Realizar preguntas**: Permite hacer preguntas específicas sobre el contenido de un documento PDF y obtener respuestas precisas.
- **Convertir a Word**: Convierte documentos PDF en documentos Word (.docx).
- **Generar resumen**: Genera un resumen del contenido de un documento PDF.

## Funcionalidades Futuras

- Comprimir PDF
- Aplicar OCR (Reconocimiento óptico de caracteres) a un PDF
- ... y más.

## Instalación

1. Asegúrate de tener Python 3.8 o superior instalado en tu máquina.
2. Clona este repositorio usando `git clone https://github.com/bladealex9848/PDFMaster.git`
3. Navega al directorio del proyecto `cd PDFMaster`
4. Instala las dependencias necesarias usando `pip install -r requirements.txt`
5. Ejecuta `streamlit run app.py` para iniciar la aplicación.
6. Obtén una clave API de OpenAI para usar su API ChatGPT.

## Uso

1. **Realizar preguntas**

- Carga un documento PDF.
- Selecciona la opción 'Realizar preguntas' en el menú de la izquierda.
- Escribe tu pregunta en el cuadro de texto.
- Presiona 'Enter' o haz clic fuera del cuadro de texto para obtener la respuesta.

2. **Convertir a Word**

- Carga un documento PDF.
- Selecciona la opción 'Convertir a Word' en el menú de la izquierda.
- Haz clic en 'Convertir' y espera a que la conversión se complete.
- Haz clic en 'Descargar archivo Word' para descargar el archivo convertido.

3. **Generar resumen**

- Carga un documento PDF.
- Selecciona la opción 'Generar resumen' en el menú de la izquierda.
- El resumen se generará automáticamente y se mostrará en pantalla.

## Contribuciones

Si deseas contribuir a este proyecto, por favor, realiza un fork de este repositorio, crea una nueva rama, realiza tus cambios y envía un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para obtener más detalles.
