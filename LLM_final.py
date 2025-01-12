from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain.agents import LLMSingleActionAgent
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor, AgentOutputParser
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
#from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from diffusers import AutoPipelineForText2Image
import time
from typing import List, Union
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import subprocess
from diffusers import DiffusionPipeline
import torch
import os
import json
import re
import sys
from langchain_core.exceptions import OutputParserException
from langchain.schema import AgentAction, AgentFinish, HumanMessage


 
#from diffusers import DiffusionPipeline
#import torch
#import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

# Get the path of current_dir
current_dir = os.path.dirname(__file__)

# Retry mechanism for DuckDuckGo tool
class DuckDuckGoToolWithRetry:
    def __init__(self, api_wrapper, max_retries=3, delay_between_retries=2):
        self.tool = DuckDuckGoSearchResults(api_wrapper=api_wrapper)
        self.max_retries = max_retries
        self.delay_between_retries = delay_between_retries

    def invoke(self, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                return self.tool.invoke(*args, **kwargs)
            except Exception as e:
                print(f"DuckDuckGo tool error: {e}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying DuckDuckGo tool... Attempt {retries + 1}")
                    time.sleep(self.delay_between_retries)
        print("DuckDuckGo tool failed after maximum retries. Skipping...")
        return "The DuckDuckGo tool encountered an issue and was skipped."

# Initialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
duckduckgo_tool_with_retry = DuckDuckGoToolWithRetry(api_wrapper=duckduckgo_wrapper)

tools = [
    Tool(name="Wikipedia", func=wikipedia_tool.run, description="Use this tool to search for information in Wikipedia."),
    Tool(name="DuckDuckGo", func=duckduckgo_tool_with_retry.invoke, description="Search for information using DuckDuckGo.")
]

# Output Parser
class CustomOutputParserOrg(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Check if there is a clear 'Final Answer:' part in the response
            if "Final Answer:" in llm_output:
                # If "Final Answer:" exists, extract the final output after it
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )

            # Fallback: If no clear 'Final Answer' section, treat the output as the final answer
            # We will assume that if no actionable item is detected, the entire text is the final output.
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )

        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            # Fallback response
            return AgentFinish(
                return_values={"output": "Unable to parse the output. Please refine your query."},
                log=llm_output,
            )



"""
# Funktion zur Generierung der Bildunterschrift
def generate_image_caption(image_paths):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Bild laden
    image = Image.open(image_paths)
    
    # Bildunterschrift generieren
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    print(caption)
    return caption
"""


def remove_non_printable(text):
    # Filtert alle Zeichen, die nicht druckbar sind (außer Tab, Newline und Wagenrücklauf)
    return ''.join(char for char in text if char.isprintable() or char in ['\t', '\n', '\r'])


# Funktion zur Bildgrößenanpassung (ohne Speichern)
def resize_image_in_memory(image_path, max_width, max_height):
    """Passt die Größe eines Bildes im Speicher an, damit es nicht größer als max_width x max_height ist."""
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))  # Größe proportional anpassen
        return img


# Extrahieren des "answer"-Werts ohne JSON und Dictionary, nur mit String-Methoden
def extract_answer_from_string(data):
    # Entfernen der ```json und ``` Markierungen
    #cleaned_data = data.replace('```json\n', '').replace('```\n', '')  # Entfernt JSON-Markierungen

    """ 
    # Extrahieren des "answer"-Werts
    start_answer = cleaned_data.find('"answer": `') + len('"answer": `')  # Finden der Startposition von 'answer'
    end_answer = cleaned_data.find('`', start_answer)  # Finden der Position des ersten ```, der das Ende markiert
    answer = cleaned_data[start_answer:end_answer]
    
    # Extrahieren des "source"-Werts
    start_source = cleaned_data.find('"source": "') + len('"source": "')
    end_source = cleaned_data.find('"', start_source)
    source = cleaned_data[start_source:end_source]
    """
    
    # Extrahieren des "answer"-Werts
    #start_answer = cleaned_data.find('"answer": "') + len('"answer": "')
    #end_answer = cleaned_data.find('"""\n', start_answer)  # Suche nach dem Ende des "answer"-Textes
    #answer = cleaned_data[start_answer:end_answer].strip()
    
    # Extrahieren des "source"-Werts
    #start_source = cleaned_data.find('"source": "') + len('"source": "')
    #end_source = cleaned_data.find('"', start_source)
    #source = cleaned_data[start_source:end_source]

    # Regulärer Ausdruck für "answer" mit Backticks (`` ` ``) oder dreifachen Anführungszeichen (""" """)
    answer_match = re.search(r'"answer":\s*(?P<quote>["`]{1,3})(?P<answer>.*?)\1', data, re.DOTALL)
    answer = answer_match.group("answer") if answer_match else None
    
    # Extrahieren des "source"-Werts (zwischen Anführungszeichen)
    source_match = re.search(r'"source":\s*"(.*?)"', data)
    source = source_match.group(1) if source_match else None

    # Extrahiere "answer"
    #answer_start = cleaned_data.find('"answer": "')
    #answer_end = cleaned_data.find('"', answer_start + len('"answer": "'))
    #answer = cleaned_data[answer_start + len('"answer": "'): answer_end]

    # Extrahiere "source"
    #source_start = cleaned_data.find('"source": "')
    #source_end = cleaned_data.find('"', source_start + len('"source": "'))
    #source = cleaned_data[source_start + len('"source": "'): source_end]

    # Ausgabe der extrahierten Werte

    return answer, source

    

def split_text(text):
    if text.startswith("###"):
        # Wenn der Text mit "###" beginnt, nach "###" splitten
        return re.split(r'###', text)
    elif text.startswith("**"):
        # Wenn der Text mit "**" beginnt, nach "**" splitten
        return re.split(r'\*\*', text)
    else:
        # Standardmäßig nach "###" splitten, falls keines der beiden zutrifft
        return re.split(r'###', text)



# Funktion: Artikel mit Bildern in Markdown erstellen
def generate_article_with_pandoc(answer, source, captions, image_paths, output_directory="output", output_file="article.pdf"):
    """
    Erstellt einen Artikel in Markdown und konvertiert ihn mithilfe von Pandoc in ein PDF.
    Bilder werden nach den entsprechenden Absätzen eingefügt.
    """
    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Temporäre Markdown-Datei
    temp_markdown_file = "temp_article.md"

    #data = json.loads(answer)

    # Extrahiere 'answer' und 'source'
    #a1 = data.get('answer')
    #s1 = data.get('source')
    #print("a1:", a1)

    #chapters = answer.split("###") # Split into chapters by level-3 headers
    
    #chapters = re.split(r'(\*\*|###)', answer)

    chapters = split_text(answer)

    markdown_content = chapters[0] + "\n"
    markdown_content += "\n"
    markdown_content += "\n"
    
    """
    # Füge Bilder nach jedem Kapitel ein
    for i, chapter in enumerate(chapters[1:], start=1):
        markdown_content += f"### {chapter.strip()}\n"
        # Bild einfügen
        markdown_content += f"![Image {i}](path/to/image{i}.png)\n"
        markdown_content += f"**Caption: Image {i} related to the chapter above**\n\n"
    """
    
    laenge = len(chapters)
    print(laenge)
    

    # Generiere Markdown-Inhalt mit Bildern
    for i, chapter in enumerate(chapters[1:], start=1):
        markdown_content += f"### {chapter.strip()}\n"
        
        #print(i)
        
        # Bild und Caption mittig einfügen
        if i <= len(image_paths): ## and os.path.exists(image_paths[i-1]):
            image_path = image_paths[i - 1]
            caption = captions[i - 1]
            #caption = captions[i]
            #print(caption)
            resized_image = resize_image_in_memory(image_path, max_width=200, max_height=200)
            image_inline_path = os.path.join(output_directory, f"inlined_image_{i}.png")
            resized_image.save(image_inline_path)  # Verkleinertes Bild temporär speichern
            
            markdown_content += f"\n"
            markdown_content += f"![Image {caption}]({image_inline_path})\n"
            markdown_content += f"\n"
            #markdown_content += f"**Caption: {caption}**\n\n"
            
        markdown_content += "\n"

    markdown_content += "\n"

    markdown_content += f"Source: {source}"

    print("MD1:", markdown_content)


    output_string = remove_non_printable(markdown_content)

    #markdown_content = re.sub(r'[\x00-\x1F\x7F]', '', output_string)

    markdown_content = output_string
  
    print("MD2:", markdown_content)

    # Speichere den Markdown-Inhalt in einer Datei
    with open(temp_markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Pfad zur Ausgabedatei
    output_path = os.path.join(output_directory, output_file)

    # Konvertiere Markdown zu PDF mit Pandoc
    try:
        subprocess.run(["pandoc", temp_markdown_file, "-o", output_path, "--pdf-engine=xelatex"], check=True)
        print(f"Article successfully generated as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while generating article with Pandoc: {e}")
    finally:
        # Entferne temporäre Markdown-Datei
        if os.path.exists(temp_markdown_file):
            os.remove(temp_markdown_file)
        





def generate_article(detected_letter):
    
    #detected_letter = ['B']

 
    question = f"""
    You must write a structured article in string format with the following requirements:

    ## The letter {detected_letter} in American Sign Language

    Divide the article into four sections:
    1. ### Introduction: What does the letter {detected_letter} symbolize and its meaning in different contexts?
    2. ### The letter in written language: The role and use of the letter {detected_letter} in the alphabet and in words.
    3. ### The letter in sign language: How is the {detected_letter} represented in American Sign Language (ASL)? Break down the steps with detailed instructions on how to sign the {detected_letter} in the American Sign Language Alphabet.
    4. ### Conclusion: Connect written language and sign language and reflect on the role of the letter {detected_letter}.

    The total word count for the article should exceed 2000 words, with each section containing at least 250 words.
    """

    #question = f"""
    #What does the letter {detected_letter} symbolize and its meaning in different contexts?
    #"""


    output_parser = CustomOutputParserOrg()

    # Response Schema and Structured Output Parser
    response_schemas = [
        ResponseSchema(name="answer",description="Answer to the user's question"),
        ResponseSchema(
            name="source",
            description="Source used to answer the user's question, should be a website." 
        ),
    ]


    output_parser2 = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser2.get_format_instructions()

    # Gibt Format-Anweisungen aus
    print(format_instructions)


    if False:    
        # Define the prompt with format instructions
        prompt = PromptTemplate(
            #template="Answer the user's question as best as possible using the following format:\n{format_instructions}\nQuestion: {question}",
            template="""
            You must generate a response in strict String format. Use the following schema:
            {format_instructions}
            
            Question: {question}
            """,
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions},
        )
        

    prompt = PromptTemplate(
        template="Answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )

    # LLM model and chain
    #model = OllamaLLM(model="llama3.1", temperature=0.7)


    model = OllamaLLM(model="llama3.1", temperature=0.7)

    # Überprüfen, ob das Modell korrekt geladen wurde
    if model is None:
        raise ValueError("Das Modell wurde nicht korrekt geladen.")

    # Überprüfen, ob der Prompt gesetzt ist
    if prompt is None or prompt == "":
        raise ValueError("Der Prompt darf nicht None oder leer sein.")

    # Überprüfen, ob output_parser korrekt definiert ist
    if output_parser is None:
        raise ValueError("Der Output Parser darf nicht None sein.")

    # Überprüfen der Tools
    if not tools or any(tool.name is None for tool in tools):
        raise ValueError("Die Tools sind entweder leer oder haben keinen Namen.")



    llm_chain = LLMChain(llm=model, prompt=prompt)


    # Überprüfen der Eingabewerte
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Output Parser: {output_parser}")
    print(f"Tools: {tools}")


        # React agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "\nResponse:"],
        allowed_tools=[tool.name for tool in tools]
    )

    # Agent executor
    chain = AgentExecutor(agent=agent, tools=tools, verbose=True)

    #output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


    # LLM
    #model = OllamaLLM(model="llama3.1", temperature=0.7)
    #model = OllamaLLM(model="llama3.1")

    #llm_chain = LLMChain(llm=model, prompt=prompt)
    #agent = create_react_agent(llm=model, tools=tools, prompt=prompt)
    #agent = create_react_agent(model, tools, prompt)

    # Agent executor
    #chain = AgentExecutor(agent=agent, tools=tools, verbose=True)
    #chain = AgentExecutor(agent=agent, tools=tools, verbose=True,
    #                            max_iterations=3, handle_parsing_errors=True)


    #model = OllamaLLM(model="llama3.1", temperature=0.7)
    #llm_chain = LLMChain(llm=model, prompt=prompt)

    # React agent
    #agent = LLMSingleActionAgent(
    #    llm_chain=llm_chain,
    #    output_parser=output_parser,
    #    stop=["\nObservation:"],
    #    allowed_tools=[tool.name for tool in tools]
    #)

    # Agent executor
    #chain = AgentExecutor(agent=agent, tools=tools, verbose=True)


    # Execute using the chain
    #chain = create_chain()
    #response = chain.invoke({"question": question})

    # Agent Executor erstellen
    #chain = create_agent_executor()

    # Eingabe für den Agenten
    #response = chain.invoke({"input": question})
    #print(response)


    #for s in chain.stream({"question": question}):
    #   print(s)


    # Beispiel für die Schleife, die den Stream verarbeitet und Daten extrahiert
    #for s in chain.stream({"question": question}):
    #    # Extrahiert die Antwort und Quelle
    #    answer, source = extract_answer_and_source(s['output'])
    # 
    #    # Ausgabe der extrahierten Daten
    #    print("Antwort:", answer)
    #    print("Quelle:", source)

    response = chain.invoke({"question": question})
    print(f"Chain Response: {response}")
    #print("response: " , response)

    final_answer = response['output']
    print(f"Chain FinalAnswer: {final_answer}")



    # Versuche den JSON-String in ein Python Dictionary zu konvertieren
    #response_dict = ""
    answer = ""

    output = response['output']

    print(type(output)) 
    print("answer:", output)


    # Antwort extrahieren
    answer, source = extract_answer_from_string(output)

    # Ausgabe der extrahierten Antwort und Quelle
    print("Answer:")
    print(answer)
    print("\nSource:")
    print(source)

    print("Answer vor Abfrage: ", answer)


    # Verarbeite die Antwort und generiere den Artikel
    if not answer:
        print("No answer found in the response.")
    else:
        
        # Auswahl treffen, ob neue Bilder generiert werden sollen
        user_choice = input("Do you want to generate new images using the diffusion model? (Yes/No): ").strip().lower()

        if user_choice == "yes":
            
            new_directory = "generated_images"
            os.makedirs(new_directory, exist_ok=True)
            
            # Diffusionsmodell laden und Bilder generieren
            #pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
            pipeline = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha")
            #pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
            pipeline.to("cuda") if torch.cuda.is_available() else torch.device('cpu')

            prompts = [
                f"an image of the {detected_letter}",
                "an image of the alphabet ABC",
                f"an image showing the sign of the {detected_letter} in american sign language",
                "An image about the American Sign Language Alphabet"
            ]
            
            image_paths = []
            for i, prompt in enumerate(prompts, 1):
                image = pipeline(prompt).images[0]
                image_path = os.path.join(new_directory, f"image_{i}.png")
                image.save(image_path)
                image_paths.append(image_path)
                
            #for i, prompt in enumerate(prompts, 1):
                # Negative Prompts (optional)
            #    negative_prompt = "background clutter, multiple letters, handwritten, artistic, overly abstract, distorted letters, artistic fonts, blurry, low contrast, overlapping letters, text outside the alphabet, incorrect hand pose, blurry hands, extra hands, unrealistic lighting, cartoonish, background objects"
                #negative_image_embeds = None   Set to None if not required
                # Negative Embeddings generieren
                #negative_image_embeds = pipeline.embed_text(negative_prompt)
            
                # Falls negative Prompts genutzt werden sollen:
                # negative_image_embeds = pipeline.embed_text(negative_prompt)
            
            #    image = pipeline(prompt, negative_image_embeds=negative_prompt, prior_guidance_scale =1.0, height=768, width=768).images[0]
            #    image_path = os.path.join(new_directory, f"image_{i}.png")
            #    image.save(image_path)
            #    image_paths.append(image_path)
                
            print(f"Generated images have been saved in the folder: {new_directory}")

        else:
            
            # Standardbilder verwenden
            print("Using default images.")
            current_dir = os.path.dirname(__file__)
            image_paths = [
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_1.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_2.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_3.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_4.png"),
            ]

        # Bildunterschriften generieren
        captions = []
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for image_path in image_paths:
            if os.path.exists(image_path):
                image = Image.open(image_path)
                inputs = processor(image, return_tensors="pt")
                outputs = blip_model.generate(**inputs, max_length=50)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                captions.append(caption)
            else:
                captions.append("No caption available (image missing).")

        output_directory = os.path.join(os.path.dirname(__file__), "Article")
        sys.stdout.flush()  # Leert den Puffer
        #output_file = input("Enter the file name for the article with extension (default: 'article.pdf'): ")
        output_file = input("Enter the file name for the article with extension (default: 'article.pdf'): ") or "article.pdf"
        
        # Generate the article as a PDF
        print("Generating article with Pandoc...")
        sys.stdout.flush()  # Leert den Puffer
        generate_article_with_pandoc(answer, source, captions,image_paths, output_directory=output_directory, output_file=output_file)
        print("Article saved.")



detected_letter = ['W']
generate_article(detected_letter)
    
   

