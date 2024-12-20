###################################################################################################
#
# Create Diffusion Model
#
###################################################################################################
"""
from diffusers import DiffusionPipeline
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

# Get the path of current_dir
current_dir = os.path.dirname(__file__)

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
#pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline.to("cuda") if torch.cuda.is_available() else torch.device('cpu')

# Prompt user for desired image name
#image_filename = input("Enter the desired base name for your images (e.g., article_image): ")


# Define the prompts for variety
prompts = [
    "An image about the history of sign language",
    "An image about the importance of sign language",
    "An image about education of sign language",
    "An image about the future of sign language"
]

# Generate and save 4 different images
for i, prompt in enumerate(prompts, 1):
    image = pipeline(prompt).images[0]  # Generate the image
    
    # Create a unique filename for each image
    #image_path = os.path.join(current_dir, "DiffusionModelOutput", f"{image_filename}_{i}.png")
    image_path = os.path.join(current_dir, "DiffusionModelOutput", f"article_image_{i}.png")

    # Save the image
    image.save(image_path)
    print(f"Image {i} saved as {image_path}")

"""
#from stable_diffusion import StableDiffusion

#model = StableDiffusion()
#image = model.generate("A beautiful landscape painting")
#image.show()

###################################################################################################
#
# Crate llm with agent
# https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain
#
###################################################################################################
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors


import re
from typing import List, Union
import torch

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
#from langchain import SerpAPIWrapper, LLMChain
from langchain.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# Hugging face imports
from huggingface_hub import login
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline

# Get the path of current_dir
current_dir = os.path.dirname(__file__)


# API Key Hugging Face
access_token = "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"

# Hugging Face API token for authentication
login(token=access_token)

# Initialize the Wikipedia tool manually
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Initialize Duckduckgo tool manually
#from langchain_community.tools import DuckDuckGoSearchResults
#from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

#duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
#duckduckgo_tool = DuckDuckGoSearchResults(api_wrapper=duckduckgo_wrapper, source="news")

from langchain_community.tools.pubmed.tool import PubmedQueryRun
pubmed_tool = PubmedQueryRun()
# Create the tools (only using the Wikipedia tool here for example)
tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information"),
         Tool(name="Pubmed", func=pubmed_tool.invoke, description="Search for information")]

print(wikipedia_tool.run("History of sign language"))
print(pubmed_tool.invoke("sign language"))

tool_names = [tool.name for tool in tools]

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Check for "Final Answer"
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )

            # Parse for Action and Action Input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)

            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")

            action = match.group(1).strip()
            action_input = match.group(2)
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            # Fallback response
            return AgentFinish(
                return_values={"output": "Unable to parse the output. Please refine your query."},
                log=llm_output,
            )

output_parser = CustomOutputParser()

# Initiate our LLM
# Load opensource llm and its tokenizer
model_id = "openai-community/gpt2-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Maximale Token-Länge abrufen
max_tokens = model.config.max_position_embeddings


# Set up the pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline=pipe)


# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose = True)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    # We use "Observation" as our stop sequence so it will stop when it receives Tool output
    # If you change your prompt template you'll need to adjust this as well
    stop=["\nObservation:"], 
    allowed_tools=tool_names,
    early_stopping_method = "generate"
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)



# ---------------------------
# Task Execution
# ---------------------------
# Initialize chat history as an empty list
chat_history = []

# Step 1: Wikipedia Query
wiki_query = f"History of sign language"
print("=== Wikipedia Lookup ===")
response_wiki = agent_executor.invoke({"input": wiki_query, "chat_history": chat_history})
wiki_output = response_wiki["output"] if "output" in response_wiki else "No Wikipedia result found."
print(wiki_output)

# Update chat history
chat_history.append({"role": "user", "content": wiki_query})
chat_history.append({"role": "assistant", "content": wiki_output})

# Step 2: Weather Query
print("\n=== Pubmed Lookup ===")
response_weather = agent.invoke({"input": f"History of sign language", "chat_history": chat_history})
weather_output = response_weather["output"] if "output" in response_weather else "No weather result found."
print(weather_output)

# Update chat history
chat_history.append({"role": "user", "content": f"weather in {random_city}"})
chat_history.append({"role": "assistant", "content": weather_output})

# Step 3: Story Generation
detected_letter = ["A"]

prompt = f"""
Write a funny fictive story about an office accident with everyday office objects in {random_city}.
The objects involved are: {detected_letter}. 
The accident should be exaggerated and unrealistic.
Use 2 or 3 key parts of the following current weather information: '{weather_output}' as part of the accident story.
Include one or two tourist attraction: '{wiki_output}' in the story.
Add funny warning labels at the end for what to be careful when dealing with the office objects. Limit the story to 300 words.
"""

print("\n=== Generating Story ===")
response_story = agent.invoke({"input": prompt, "chat_history": chat_history})
story_output = response_story["output"] if "output" in response_story else "Failed to generate a story."
print(story_output)

# Update chat history
chat_history.append({"role": "user", "content": prompt})
chat_history.append({"role": "assistant", "content": story_output})


# Initiate the agent that will respond to our queries
# Set verbose=True to share the CoT reasoning the LLM goes through

final_response = response["output"]  # Extract the final response
print("Final Response:", final_response)


from fpdf import FPDF  # pip install fpdf


# Create the article in pdf format
from fpdf import FPDF

# Initialisiere das PDF
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    # Textabschnitt mit Bild hinzufügen
    def add_paragraph_with_image(self, paragraph, image_path):
        # Text (Paragraph) hinzufügen
        self.multi_cell(0, 10, paragraph)
        self.ln()  # Abstand nach dem Text

        # Bild hinzufügen
        if image_path:
            self.image(image_path, x=None, y=None, w=100)  # x=None, y=None platziert das Bild an der aktuellen Position
            self.ln(10)  # Abstand nach dem Bild


# Liste von Bildpfaden
image_paths = [
    os.path.join(current_dir, "DiffusionModelOutput", "article_image_1.png"),
    os.path.join(current_dir, "DiffusionModelOutput", "article_image_2.png"),
    os.path.join(current_dir, "DiffusionModelOutput", "article_image_3.png"),
    os.path.join(current_dir, "DiffusionModelOutput", "article_image_4.png"),
]

# PDF erstellen
pdf = PDF()

# Text in Absätze aufteilen
paragraphs = final_response.strip().split("\n\n")  # Absätze erkennen anhand doppelter Zeilenumbrüche

# Prüfen, ob genug Bilder vorhanden sind
#if len(paragraphs) > len(image_paths):
#    raise ValueError("Nicht genügend Bilder für die Absätze vorhanden!")

# Absätze und Bilder hinzufügen
#for paragraph, image_path in zip(paragraphs, image_paths):
#    pdf.add_paragraph_with_image(paragraph.strip(), image_path)

# Bilder und Absätze hinzufügen
for i, paragraph in enumerate(paragraphs):
    # Hinzufügen des Textes
    pdf.add_paragraph(paragraph.strip())

    # Prüfen, ob noch Bilder vorhanden sind und wie viele
    if i < len(image_paths):  # Bild hinzufügen, wenn noch Bilder vorhanden sind
        image_path = image_paths[i]
        pdf.add_image(image_path)
    elif i < len(image_paths) + len(image_paths) // 2:  # Zwei Bilder nebeneinander
        if i + 1 < len(image_paths):  # Sicherstellen, dass es ein weiteres Bild gibt
            image_path1 = image_paths[i]
            image_path2 = image_paths[i + 1]
            pdf.add_two_images(image_path1, image_path2)
            i += 1  # Um den nächsten Absatz nach beiden Bildern zu überspringen

# PDF speichern
pdf.output("output.pdf")

"""
# Erstelle ein FPDF-Objekt
pdf = FPDF()

# Füge eine Seite hinzu
pdf.add_page()

# Setze die Schriftart
pdf.set_font("Arial", size=12)

# Angenommen, response ist ein Dictionary, das den Text unter einem bestimmten Schlüssel enthält
# Zum Beispiel: response = {"output": "Hier steht der generierte Text zum Thema Gebärdensprache..."}

# Schreibe den extrahierten Text in die PDF
pdf.multi_cell(0, 10, answer, align="c")

current_dir = os.path.dirname(__file__)
pdf_filename = input("Enter the desired PDF file name (without extension): ")
pdf.output(os.path.join(current_dir, "Article", f"{pdf_filename}.pdf"))

print("PDF wurde erfolgreich erstellt!")
"""