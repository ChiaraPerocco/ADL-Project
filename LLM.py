###################################################################################################
#
# Image generation
# https://huggingface.co/ehristoforu/dalle-3-xl-v2
#
###################################################################################################
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

# get the path of current_dir
current_dir = os.path.dirname(__file__)

if False:
    from diffusers import DiffusionPipeline

    #pipe = DiffusionPipeline.from_pretrained("fluently/Fluently-XL-v2")
    #pipe.load_lora_weights("ehristoforu/dalle-3-xl-v2")

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")


    prompt = "Create picture of a sign language letter."
    image = pipe(prompt).images[0]

    # Save the image to a folder
    image_path = os.path.join(current_dir, "DiffusionModelOutput", 'generated_image.png')  # Change the path as needed
    image.save(image_path)

    print(f"Image saved at {image_path}")

image_path = os.path.join(current_dir, "DiffusionModelOutput", 'generated_image.png')


if False:
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    #pipe = pipe.to("cuda")

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    # Step 5: Save the image to a folder
    #image_path = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\ADL_team_project_master\Output images\generated_image.png'  # Change the path as needed
    #image.save(image_path)

    #print(f"Image saved at {image_path}")
    
###################################################################################################
#
# Image caption generation
#
# Hugging face
# source: https://huggingface.co/Salesforce/blip-image-captioning-base
#
###################################################################################################
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Bildunterschrift mit BLIP generieren
def generate_image_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Bild laden
    image = Image.open(image_path)
    
    # Bildunterschrift generieren
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

caption = generate_image_caption(image_path)
print("Generated Caption:", caption)


###################################################################################################
#
# Large Language Model: using Hugging Face's transformers
#
###################################################################################################

if False:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Load an open-source model and tokenizer, such as GPT-2 or GPT-Neo
    model_name = "EleutherAI/gpt-neo-1.3B"  # GPT-Neo is a good alternative to OpenAI's models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Generate a response to a prompt
    #prompt = "Five surprising names for a pet pelican"

    # Define the article structure with section prompts
    prompts = [
        "Write an introductory paragraph about the topic of climate change.",
        "Write a second paragraph explaining the causes of climate change.",
        "Write a third paragraph about the effects of climate change on the environment.",
        "Write a conclusion about actions to combat climate change."
    ]

    # Generate text for each section separately
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=250, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    # Combine responses into a single article
    article = "\n\n".join(responses)
    print(article)

        # Encode input and generate response
        #inputs = tokenizer(prompt, return_tensors="pt")
        #outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode and print the output text
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print(response)

###################################################################################################
#
# Text generation
#
# Hugging face, Large chain
# https://python.langchain.com/v0.2/docs/integrations/tools/ddg/
# To Do: Try - https://huggingface.co/tiiuae/falcon-40b
# To Do: Acceleration - https://huggingface.co/docs/transformers/accelerate
# To Do: Include Agent - https://huggingface.co/docs/transformers/agents_advanced
# To Do: Pubmed model - https://huggingface.co/microsoft/BioGPT-Large-PubMedQA
#
###################################################################################################
# Install necessary packages
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image
from textwrap import wrap
from transformers import pipeline
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# Load Hugging Face model for text generation
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_answer_for_section(question, aspect):
    # Create a specific prompt for each section
    specific_question = f"{question} ({aspect})"
    
    # Step 1: Use DuckDuckGo to get search results based on the specific question
    search_results = search.invoke(specific_question)
    
    # Step 2: Use search results as context and generate a focused response
    input_text = f"Question: {specific_question}\nContext: {search_results}"
    answer = qa_pipeline(input_text, max_length=200, min_length = 50, do_sample=False)[0]['generated_text']
    
    return answer

def draw_wrapped_text(c, text, x, y, max_chars, font_name="Helvetica", font_size=12, leading=14):
    """
    Draws wrapped text on the canvas `c`, with a max number of characters per line.
    Returns the updated vertical position after drawing.
    """
    wrapped_lines = wrap(text, width=max_chars)
    text_object = c.beginText(x, y)
    text_object.setFont(font_name, font_size)
    text_object.setLeading(leading)
    
    for line in wrapped_lines:
        text_object.textLine(line)
    c.drawText(text_object)
    
    # Return new Y position after drawing the text
    return y - len(wrapped_lines) * leading

# Artikel-PDF mit Bild und Bildunterschrift erstellen
def create_article_pdf(question, image_path, caption, output_pdf_path):
    # Define sections and aspects
    sections = [
        ("Introduction", "Provide a general overview of the topic"),
        ("Background", "Describe the historical or contextual background of the topic"),
        ("Detailed Analysis", "Give an in-depth exploration of specific details"),
        ("Conclusion", "Summarize the findings and discuss implications")
    ]
    
    
    # Generate content for each section
    section_texts = {}
    for section_title, aspect in sections:
        section_texts[section_title] = generate_answer_for_section(question, aspect)

    # Initialize ReportLab canvas
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    margin = 50
    max_chars = 90  # Approximate char width for wrapping
    current_y = height - margin  # Start from the top margin

    def check_page_break(c, current_y, section_height):
        if current_y - section_height < margin:
            c.showPage()
            return height - margin
        return current_y

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, current_y, "Generated Article: " + question)
    current_y -= 30

    # Caption
    current_y = check_page_break(c, current_y, 40)
    c.setFont("Helvetica", 12)
    c.drawString(margin, current_y, "Image Caption: " + caption)
    current_y -= 30

    # Add Image
    if os.path.exists(image_path):
        max_image_height = 3 * 28.35  # 3 cm in points
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
        
     # Draw each section
    for section_title, _ in sections:
        current_y = check_page_break(c, current_y, 80)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, current_y, section_title + ":")
        current_y -= 20
        section_content = section_texts[section_title]
        current_y = draw_wrapped_text(c, section_content, margin, current_y, max_chars)
        current_y -= 30  # Add space after each section

    # Save the PDF
    c.showPage()
    c.save()
    print(f"Article PDF created at: {output_pdf_path}")

# Main usage
question = "How has the use of sign language evolved over the years?"
pdf_path = os.path.join(current_dir, "Article", 'article.pdf') 
create_article_pdf(question, image_path, caption, pdf_path)
