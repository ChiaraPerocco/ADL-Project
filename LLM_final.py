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
from langchain_core.exceptions import OutputParserException
from langchain.schema import AgentAction, AgentFinish, HumanMessage


if False:
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

    detected_letter = ['B']

    question = f"""
    You must write a structured article in string format with the following requirements:

    Divide the article into four sections:
    1. **Introduction**: What does the letter {detected_letter} symbolize and its meaning in different contexts?
    2. **The letter in written language**: The role and use of the letter {detected_letter} in the alphabet and in words.
    3. **The letter in sign language**: How is the {detected_letter} represented in American Sign Language (ASL)? Break down the steps with detailed instructions on how to sign the {detected_letter} in the American Sign Language Alphabet.
    4. **Conclusion**: Connect written language and sign language and reflect on the role of the letter {detected_letter}.

    The total word count for the article should exceed 2000 words, with each section containing at least 250 words.
    """

    #question = f"""
    #What does the letter {detected_letter} symbolize and its meaning in different contexts?
    #"""

    # Response Schema and Structured Output Parser
    response_schemas = [
        ResponseSchema(name="answer", description="Answer to the user's question", type="str"),
        ResponseSchema(
            name="source",
            description="Source used to answer the user's question, should be a website."
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Gibt Format-Anweisungen aus
    print(format_instructions)

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


    # LLM
    model = OllamaLLM(model="llama3.1", temperature=0.7)
    #llm_chain = LLMChain(llm=model, prompt=prompt)

    """
    # Create chain with prompt, model, and output parser
    def create_chain():
        return prompt | model | output_parser

    # Create the AgentExecutor

    def create_agent_executor():
        search_agent = create_react_agent(prompt, model, output_parser)
        return AgentExecutor(
            agent=search_agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True,
        )
    """

    """
    # React agent
    agent = LLMSingleActionAgent(
        llm_chain=prompt | model,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    """
    agent = create_react_agent(llm=model, tools=tools, prompt=prompt)

    # Agent executor
    chain = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Execute using the chain
    #chain = create_chain()
    #response = chain.invoke({"question": question})

    # Agent Executor erstellen
    #chain = create_agent_executor()

    # Eingabe für den Agenten
    #response = chain.invoke({"input": question})
    #print(response)

    response = chain.invoke({"question": question})
    #final_answer = response['output']
    #print(response['output'])

    # Versuche den JSON-String in ein Python Dictionary zu konvertieren
    response_dict = ""
    try:
        # Überprüfen des Typs von response
        print(type(response))  # Gibt den Typ von response aus

        # Wenn response ein String ist, versuche, es in ein Dictionary umzuwandeln
        if isinstance(response, str):
            try:
                response_dict = json.loads(response)  # String in ein Dictionary umwandeln
                print(type(response_dict))  # Gibt den Typ des umgewandelten Objekts aus
            except json.JSONDecodeError as e:
                print(f"Fehler beim Parsen von JSON: {e}")
        else:
            response_dict = response
            print("response ist kein String und kann nicht als JSON geparsed werden.")

        #response_dict = json.loads(response)  # String in ein Dictionary umwandeln
        print("Response type:", type(response_dict))  # Überprüfen, ob es nun ein Dictionary ist

        # Zugriff auf den "answer" und "source" Schlüssel
        answer = response_dict.get("answer", "")
        source = response_dict.get("source", "")

        #print("Answer:", answer)
        #print("Source:", source)
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen des JSON-Strings: {e}")

    print("Whole response:")
    print(response_dict)
    # Output results
    #print("Final Answer:")
    #print(response['answer'])
    #print("\nSources:")
    #print(response['source'])

    #final_answer = response['answer']
    #print(str(final_answer))

    #answer = response.get("answer", "")
    #print(answer)

    import subprocess
    import os

from diffusers import DiffusionPipeline
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

# Get the path of current_dir
current_dir = os.path.dirname(__file__)



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


# Funktion zur Bildgrößenanpassung (ohne Speichern)
def resize_image_in_memory(image_path, max_width, max_height):
    """Passt die Größe eines Bildes im Speicher an, damit es nicht größer als max_width x max_height ist."""
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))  # Größe proportional anpassen
        return img


# Funktion: Artikel mit Bildern in Markdown erstellen
def generate_article_with_pandoc(image_paths, output_directory="output", output_file="article.pdf"):
    """
    Erstellt einen Artikel in Markdown und konvertiert ihn mithilfe von Pandoc in ein PDF.
    Bilder werden nach den entsprechenden Absätzen eingefügt.
    """
    # Stelle sicher, dass das Ausgabeverzeichnis existiert
    os.makedirs(output_directory, exist_ok=True)

    # Temporäre Markdown-Datei
    temp_markdown_file = "temp_article.md"

    chapters = answer.split("###")  # Split into chapters by level-3 headers
    markdown_content = chapters[0] + "\n"
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
        # markdown_content += f"**Caption: {caption}**\n\n"
            
        markdown_content += f"\n"

            
    # Speichere den Markdown-Inhalt in einer Datei
    with open(temp_markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Pfad zur Ausgabedatei
    output_path = os.path.join(output_directory, output_file)

    # Konvertiere Markdown zu PDF mit Pandoc
    try:
        subprocess.run(["pandoc", temp_markdown_file, "-o", output_path], check=True)
        print(f"Article successfully generated as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while generating article with Pandoc: {e}")
    finally:
        # Entferne temporäre Markdown-Datei
        if os.path.exists(temp_markdown_file):
            os.remove(temp_markdown_file)
            
            
# Liste von Bildpfaden
#current_dir = os.path.dirname(__file__)




if True:
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


    detected_letter = ['B']

    question = f"""
    You must write a structured article in string format with the following requirements:

    Divide the article into four sections:
    1. **Introduction**: What does the letter {detected_letter} symbolize and its meaning in different contexts?
    2. **The letter in written language**: The role and use of the letter {detected_letter} in the alphabet and in words.
    3. **The letter in sign language**: How is the {detected_letter} represented in American Sign Language (ASL)? Break down the steps with detailed instructions on how to sign the {detected_letter} in the American Sign Language Alphabet.
    4. **Conclusion**: Connect written language and sign language and reflect on the role of the letter {detected_letter}.

    The total word count for the article should exceed 2000 words, with each section containing at least 250 words.
    """

    #question = f"""
    #What does the letter {detected_letter} symbolize and its meaning in different contexts?
    #"""

    # Response Schema and Structured Output Parser
    response_schemas = [
        ResponseSchema(name="answer", description="Answer to the user's question", type="str"),
        ResponseSchema(
            name="source",
            description="Source used to answer the user's question, should be a website."
        )
    ]

    # Output Parser
    class CustomOutputParser(AgentOutputParser):
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

    output_parser = CustomOutputParser()
    output_parser2 = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser2.get_format_instructions()

    # Gibt Format-Anweisungen aus
    print(format_instructions)

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


    # LLM model and chain
    model = OllamaLLM(model="llama3.1", temperature=0.7)
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # React agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
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

    response = chain.invoke({"question": question})
    #final_answer = response['output']
    #print(response['output'])

    # Versuche den JSON-String in ein Python Dictionary zu konvertieren
    response_dict = ""
    answer = ""
    if False:
        try:
            # Überprüfen des Typs von response
            print(type(response))  # Gibt den Typ von response aus

            # Wenn response ein String ist, versuche, es in ein Dictionary umzuwandeln
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)  # String in ein Dictionary umwandeln
                    print(type(response_dict))  # Gibt den Typ des umgewandelten Objekts aus
                except json.JSONDecodeError as e:
                    print(f"Fehler beim Parsen von JSON: {e}")
            else:
                response_dict = response
                print("response ist kein String und kann nicht als JSON geparsed werden.")

            #response_dict = json.loads(response)  # String in ein Dictionary umwandeln
            print("Response type:", type(response_dict))  # Überprüfen, ob es nun ein Dictionary ist

            # Zugriff auf den "answer" und "source" Schlüssel
            answer = response_dict.get("answer", "")
            source = response_dict.get("source", "")

            print("Answer:", answer)
            print("Source:", source)
        except json.JSONDecodeError as e:
            print(f"Fehler beim Parsen des JSON-Strings: {e}")


    output = response['output']
    print(type(output)) 
    print("answer:", output)
    print("Original Output:", repr(output)) 

    cleaned_data = output.strip("```json").strip("```")
    # Die Backticks im JSON-String (rund um `...`) durch Anführungszeichen ersetzen
    cleaned_data = cleaned_data.replace("`", '"')
    #print("cleaned_data:", cleaned_data)
    cleaned_data = cleaned_data.replace("\n", " ").replace("\t", " ")
    print("Bereinigte Daten:", repr(cleaned_data))
    if isinstance(cleaned_data, str):
        try:
            response_dict = json.loads(cleaned_data)  # String in ein Dictionary umwandeln
            print("umgewabdeltes Objekt:", response_dict)
            print(type(response_dict))  # Gibt den Typ des umgewandelten Objekts aus
            #print(response_dict['answer'])
            # Zugriff auf das `answer`-Attribut
            answer = response_dict.get("answer", "")
            print("Answer2:", answer)
        except json.JSONDecodeError as e:
            print(f"Fehler beim Parsen von JSON: {e}")
    else:
        response_dict = cleaned_data
        print("response ist kein String und kann nicht als JSON geparsed werden.")
        print(response_dict['answer'])

    #answer = response_dict.get("answer", "")
    #print(answer)
    #data = json.load(output)
    #print(data['answer'])  
     
    #response_dict = data['answer']
    #print(data['answer'])    
    
    """
    # Attempt to extract the "answer"
    try:
        output = response.get('output', '') # Extract 'output' key from the response
        #output = response.get('answer', '')
        if not output:
            raise ValueError("The 'output' key is empty or missing.")
        
        try:
            response_dict = json.loads(output)  # Parse the output as JSON
            answer = response_dict.get("answer", "").strip()  # Extract the answer
            source = response_dict.get("source", "").strip()  # Extract the source (if necessary)

            if answer:
                print("Answer:", answer)
                print("Source:", source)
            else:
                print("No 'answer' found in the JSON. Using fallback mechanism.")

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Fallback: Using raw output.")
            answer = output.strip()

        if answer:
            print("Extracted Answer:", answer)
        else:
            print("No answer found.")

    except Exception as e:
        print(f"Unexpected error extracting the answer: {e}")
    """        
    print("Whole response:")
    print(response_dict)
    # Output results
    #print("Final Answer:")
    #print(response['answer'])
    #print("\nSources:")
    #print(response['source'])

    #final_answer = response['answer']
    #print(str(final_answer))

    #answer = response.get("answer", "")
    #print(answer)
   

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
    output_file = input("Enter the file name for the article with extension (default: 'article.pdf'): ")
    
    # Generate the article as a PDF
    print("Generating article with Pandoc...")
    generate_article_with_pandoc(image_paths, output_directory=output_directory, output_file=output_file)
    print("Article saved.")





if False:
    # Prompt template
    template = '''Answer the following questions as best you can. You have access to the following tools:


    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: think about whether you can answer the question directly using your internal knowledge.
    If yes, provide the answer directly.
    If no, consider which tool to use to gather more information.

    Action: the action to take, should be one of [{tool_names}] if tools are necessary.
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

if False:
    # Define custom prompt template
    class CustomPromptTemplate(BaseChatPromptTemplate):
        template: str
        tools: List[Tool]

        def format_messages(self, **kwargs) -> List[HumanMessage]:
            intermediate_steps = kwargs.pop("intermediate_steps", [])
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            
            kwargs["agent_scratchpad"] = thoughts
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            
            return [HumanMessage(content=self.template.format(**kwargs))]

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )

    # Output Parser
    class CustomOutputParser(AgentOutputParser):
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

    output_parser = CustomOutputParser()

    """
    # LLM model and chain
    model = OllamaLLM(model="llama3.1", temperature=0.7)
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # React agent
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )

    # Agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    """

    # Question
    question = """
    Please provide a detailed text in string format about American Sign Language (ASL) with the following structure:
    1. **How the letter 'A' is signed in ASL**: Break down the steps with detailed instructions on how to sign the letter.
    2. **A general overview of ASL**: Include its history, structure, key features, and the communities that use it.
    3. **Interesting facts about ASL**: Cover its origins, cultural significance, and unique linguistic properties.
    4. Each section should have a thorough explanation of at least 250 words. Ensure the total word count exceeds 1000 words.

    Each section should be a clear and distinct paragraph. Please include specific examples where necessary.
    """


    # Execute
    response = agent_executor.invoke({"input": question})
    final_answer = response['output']
    print(response['output'])



 # Liste von Bildpfaden
#current_dir = os.path.dirname(__file__)



if False:

    from fpdf import FPDF  # pip install fpdf


    # Create the article in pdf format
    from fpdf import FPDF
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

    # Get the path of current_dir
    current_dir = os.path.dirname(__file__)

    # Initialisiere das PDF
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)
            self.add_page()
            self.set_font("Arial", size=12)

        # Add a paragraph with an image after it
        def add_paragraph_with_image(self, paragraph, image_path):
            # Text (Paragraph) hinzufügen
            self.multi_cell(0, 10, paragraph)
            self.ln()  # Abstand nach dem Text

            # Bild hinzufügen
            if image_path:
                self.image(image_path, x=None, y=None, w=100)  # x=None, y=None platziert das Bild an der aktuellen Position
                self.ln(10)  # Abstand nach dem Bild

        # Add two images side by side
        def add_two_images(self, image_path1, image_path2):
            # Place the first image on the left (x=10)
            self.image(image_path1, x=10, y=self.get_y(), w=85)
            
            # Move to the right for the second image
            self.image(image_path2, x=105, y=self.get_y(), w=85)
            
            # Move down after the images
            self.ln(90)  # Adjust the distance according to your preference

    # Liste von Bildpfaden
    image_paths = [
        os.path.join(current_dir, "DiffusionModelOutput", "article_image_1.png"),
        os.path.join(current_dir, "DiffusionModelOutput", "article_image_2.png"),
        os.path.join(current_dir, "DiffusionModelOutput", "article_image_3.png"),
        os.path.join(current_dir, "DiffusionModelOutput", "article_image_4.png"),
    ]

    # Create the PDF instance
    pdf = PDF()

    #pdf.add_headline("American Sign Language (ASL): An In-depth Overview")

    # Split the text into paragraphs
    paragraphs = final_answer.strip().split("\n\n")  # Split paragraphs based on double line breaks

    # Check if we have enough images for the paragraphs
    if len(paragraphs) > len(image_paths):
        raise ValueError("Not enough images for the paragraphs!")

    # Add paragraphs and images
    for i, paragraph in enumerate(paragraphs):
        # Add paragraph
        pdf.add_paragraph_with_image(paragraph.strip(), None)  # Add paragraph without image for now
        
        # Check if there's an image for the current paragraph
        if i < len(image_paths):
            image_path = image_paths[i]
            pdf.add_paragraph_with_image("", image_path)  # Add image after the paragraph

    # Add two images side by side, if necessary
    if len(paragraphs) % 2 == 0 and len(image_paths) >= 2:
        for i in range(0, len(image_paths)-1, 2):  # Step by 2
            pdf.add_two_images(image_paths[i], image_paths[i+1])

    # Save the PDF
    pdf_filename = input("Enter the desired PDF file name (without extension): ")
    pdf.output(os.path.join(current_dir, "Article", f"{pdf_filename}.pdf"))

    print("PDF was successfully created!")


