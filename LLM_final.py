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
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )

            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )

        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            return AgentFinish(
                return_values={"output": "Unable to parse the output. Please refine your query."},
                log=llm_output,
            )


def remove_non_printable(text):
    # filter non printable signs
    return ''.join(char for char in text if char.isprintable() or char in ['\t', '\n', '\r'])


# adjust image size
def resize_image_in_memory(image_path, max_width, max_height):
    """Passt die Größe eines Bildes im Speicher an, damit es nicht größer als max_width x max_height ist."""
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height))  
        return img


# extract answer and source
def extract_answer_from_string(data):
    
    answer_match = re.search(r'"answer":\s*(?P<quote>["`]{1,3})(?P<answer>.*?)\1', data, re.DOTALL)
    answer = answer_match.group("answer") if answer_match else None
   
    source_match = re.search(r'"source":\s*"(.*?)"', data)
    source = source_match.group(1) if source_match else None

    return answer, source

    

def split_text(text):
    if text.startswith("###"):
        return re.split(r'###', text)
    elif text.startswith("**"):
        return re.split(r'\*\*', text)
    else:
        return re.split(r'###', text)



# generate article with images in markdown
def generate_article_with_pandoc(answer, source, captions, image_paths, output_directory="output", output_file="article.pdf"):
   
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # temporary markdown file
    temp_markdown_file = "temp_article.md"

    chapters = split_text(answer)

    markdown_content = chapters[0] + "\n"
    markdown_content += "\n"
    markdown_content += "\n"
    
    laenge = len(chapters)
    #print(laenge)
    

    # generate markdown content
    for i, chapter in enumerate(chapters[1:], start=1):
        markdown_content += f"### {chapter.strip()}\n"
        
        #print(i)
        
        # add image and caption
        if i <= len(image_paths): 
            image_path = image_paths[i - 1]
            caption = captions[i - 1]
            resized_image = resize_image_in_memory(image_path, max_width=200, max_height=200)
            image_inline_path = os.path.join(output_directory, f"inlined_image_{i}.png")
            resized_image.save(image_inline_path)  
            
            markdown_content += f"\n"
            markdown_content += f"![Image {caption}]({image_inline_path})\n"
            markdown_content += f"\n"
            
        markdown_content += "\n"

    markdown_content += "\n"

    markdown_content += f"Source: {source}"

    print("MD1:", markdown_content)


    output_string = remove_non_printable(markdown_content)

    markdown_content = output_string
  
    print("MD2:", markdown_content)

    # save markdown content
    with open(temp_markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # path to output
    output_path = os.path.join(output_directory, output_file)

    # convert markdown to pdf using pandoc
    try:
        subprocess.run(["pandoc", temp_markdown_file, "-o", output_path, "--pdf-engine=xelatex"], check=True)
        print(f"Article successfully generated as {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while generating article with Pandoc: {e}")
    finally:
        #remove teemporary markdown file
        if os.path.exists(temp_markdown_file):
            os.remove(temp_markdown_file)


def generate_article(detected_letter):

    question = f"""
    You must write a structured article in string format with the following requirements:

    ## The letter {detected_letter} in American Sign Language

    Divide the article into four sections:
    1. ### Introduction: What does the letter {detected_letter} symbolize and its meaning in different contexts?
    2. ### The letter in written language: The role and use of the letter {detected_letter} in the alphabet and in words.
    3. ### The letter in sign language: Which hand shape does the {detected_letter} represent in the ASL alphabet?
    4. ### Conclusion: Connect written language and sign language and reflect on the role of the letter {detected_letter}.

    The total word count for the article should exceed 2000 words, with each section containing at least 250 words.
    """

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

    print(format_instructions)

    prompt = PromptTemplate(
        template="Answer the users question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )


    model = OllamaLLM(model="llama3.1", temperature=0.7)


    # check if model was loaded correct
    if model is None:
        raise ValueError("model failed to load.")

    # check prompt
    if prompt is None or prompt == "":
        raise ValueError("The prompt must not be None or empty.")

    # check if output parser is defined correct
    if output_parser is None:
        raise ValueError("The output parser must not be None.")

    # check the tools
    if not tools or any(tool.name is None for tool in tools):
        raise ValueError("The tools are either empty or have no name.")



    llm_chain = LLMChain(llm=model, prompt=prompt)


    # check input
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


    response = chain.invoke({"question": question})
    print(f"Chain Response: {response}")
    #print("response: " , response)

    final_answer = response['output']
    print(f"Chain FinalAnswer: {final_answer}")

    answer = ""

    output = response['output']

    print(type(output)) 
    print("answer:", output)


    # extract answer and source
    answer, source = extract_answer_from_string(output)

    # print extracted answer and source
    print("Answer:")
    print(answer)
    print("\nSource:")
    print(source)

    print("Answer vor Abfrage: ", answer)


    if not answer:
        print("No answer found in the response.")
    else:
        
        # generating new images
        user_choice = input("Do you want to generate new images using the diffusion model? (Yes/No): ").strip().lower()

        if user_choice == "yes":
            
            new_directory = "generated_images"
            os.makedirs(new_directory, exist_ok=True)
            
            # load diffuisonmodel and generate images
            pipeline = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha")
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
                
            
            print(f"Generated images have been saved in the folder: {new_directory}")

        else:
            
            # using default images
            print("Using default images.")
            current_dir = os.path.dirname(__file__)
            image_paths = [
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_1.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_2.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_3.png"),
                os.path.join(current_dir, "DiffusionModelOutput", "article_image_4.png"),
            ]

        # generate image captions
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
        sys.stdout.flush() 
        output_file = input("Enter the file name for the article with extension (default: 'article.pdf'): ") or "article.pdf"
        
        # Generate the article as a PDF
        print("Generating article with Pandoc...")
        sys.stdout.flush() 
        generate_article_with_pandoc(answer, source, captions,image_paths, output_directory=output_directory, output_file=output_file)
        print("Article saved.")


   

