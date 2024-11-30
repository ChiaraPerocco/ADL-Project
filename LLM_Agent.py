import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from textwrap import wrap
from diffusers import DiffusionPipeline

# 1. Emotionserkennung (dieser Teil hast du bereits implementiert)
# Angenommene Funktion, die die Emotion im Bild erkennt
def detect_emotion(image_path):
    # Beispielhafte Emotionserkennung: Hier kann deine eigene Logik oder ein Modell eingesetzt werden
    # Zum Beispiel könnte das Modell "Happy", "Angry", "Sad", "Neutral", "Surprised" zurückgeben.
    emotion = "Happy"  # Beispielhafte Emotion
    return emotion

# 2. LLM-Agent für die Artikelgenerierung
search = DuckDuckGoSearchRun()
llm = ChatOpenAI(api_key="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",temperature=0)

prompt_template = PromptTemplate(
    input_variables=["emotion"],
    template="Please provide a comprehensive article about the emotion {emotion}. The article should include an introduction, background, detailed analysis, and conclusion."
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

tools = [
    Tool(
        name="Search",
        func=search.invoke,
        description="Use this tool to search the web for relevant information."
    ),
    Tool(
        name="AnswerGenerator",
        func=llm_chain.run,
        description="Use this tool to generate detailed answers based on the emotion."
    )
]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# Funktion zur Generierung des Textes
def generate_article(emotion):
    specific_question = f"Write an article about the emotion {emotion}. Include an introduction, background, detailed analysis, and conclusion."
    result = agent.run(specific_question)
    return result

# 3. Bildunterschriftenerstellung
def generate_image_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# 4. Bildgenerierung mit Stable Diffusion oder DALL·E (Simuliert in diesem Beispiel)
def generate_image(emotion):
    
    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    
    # Dynamischer Prompt basierend auf der Emotion
    prompt = f"Create a picture of a person experiencing {emotion} emotion."

    # Bildgenerierung
    image = pipe(prompt).images[0]

    # Bildpfad festlegen und speichern
    image_path = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\generated_image.png'  # Ändere den Pfad nach Bedarf
    image.save(image_path)

    print(f"Image saved at {image_path}")

    return image_path  # Rückgabe des Bildpfads

# 5. PDF-Erstellung mit dem generierten Text und den Bildern
def draw_wrapped_text(c, text, x, y, max_chars, font_name="Helvetica", font_size=12, leading=14):
    wrapped_lines = wrap(text, width=max_chars)
    text_object = c.beginText(x, y)
    text_object.setFont(font_name, font_size)
    text_object.setLeading(leading)
    
    for line in wrapped_lines:
        text_object.textLine(line)
    c.drawText(text_object)
    
    return y - len(wrapped_lines) * leading

def create_article_pdf_with_images_and_captions(emotion, article_text, output_pdf_path):
    image_path = generate_image(emotion)  # Erstelle Bild basierend auf Emotion
    caption = generate_image_caption(image_path)  # Generiere passende Bildunterschrift

    # ReportLab Canvas für PDF
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    margin = 50
    max_chars = 90  # Max Zeichen pro Zeile
    current_y = height - margin  # Startposition für den Text

    def check_page_break(c, current_y, section_height):
        if current_y - section_height < margin:
            c.showPage()
            return height - margin
        return current_y

    # Titel
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, current_y, f"Generated Article about {emotion}")
    current_y -= 30

    # Bildunterschrift
    current_y = check_page_break(c, current_y, 40)
    c.setFont("Helvetica", 12)
    c.drawString(margin, current_y, f"Image Caption: {caption}")
    current_y -= 30

    # Bild hinzufügen
    if os.path.exists(image_path):
        max_image_height = 3 * 28.35  # Maximalhöhe des Bildes (3 cm in Punkten)
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        if img_height > max_image_height:
            scaling_factor = max_image_height / img_height
            img_width *= scaling_factor
            img_height = max_image_height

        if current_y - img_height < margin:
            c.showPage()
            current_y = height - margin

        c.drawImage(image_path, margin, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 20
    else:
        print(f"Image file not found at {image_path}")

    # Füge den Artikeltext hinzu
    current_y = draw_wrapped_text(c, article_text, margin, current_y, max_chars)
    current_y -= 30

    # PDF speichern
    c.showPage()
    c.save()
    print(f"Article PDF created at: {output_pdf_path}")

# Hauptaufruf
image_path = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\generated_image.png'  # Beispielhafte Bildpfad
emotion = detect_emotion(image_path)  # Emotion erkennen
article_text = generate_article(emotion)  # Artikel basierend auf der Emotion generieren
output_pdf_path = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\sad_text.pdf'  # PDF-Ausgabepfad
create_article_pdf_with_images_and_captions(emotion, article_text, output_pdf_path)
