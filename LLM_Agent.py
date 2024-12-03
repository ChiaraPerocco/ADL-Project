if False:
    import os
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from PIL import Image
    from langchain_community.tools import DuckDuckGoSearchRun
    #from langchain_openai import ChatOpenAI
    from transformers import pipeline
    from langchain.prompts import PromptTemplate
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.agents import AgentExecutor
    from langchain.chains import LLMChain
    from textwrap import wrap
    from diffusers import DiffusionPipeline
    from langchain_community.llms import HuggingFacePipeline
    #pip install -U langchain-community
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    #from langchain.utilities import DuckDuckGoSearchRun
    from langchain.llms import HuggingFacePipeline
    from transformers import pipeline

# https://medium.com/@aydinKerem/what-is-an-llm-agent-and-how-does-it-work-1d4d9e4381ca


from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

chat_model = ChatHuggingFace(llm=llm)


from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper

# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Who is the current holder of the speed skating world record on 500 meters? What is her current age raised to the 0.43 power?"
    }
)


#git clone https://github.com/ggerganov/llama.cpp.git
#cd llama.cpp
#pip install .
#https://medium.com/@aydinKerem/what-is-an-llm-agent-and-how-does-it-work-1d4d9e4381ca

# pip install ollama-python
#https://www.learndatasci.com/solutions/how-to-use-open-source-llms-locally-for-free-ollama-python/
# enable verbose to debug the LLM's operation
#verbose = False

#from langchain_community.chat_models import ChatOllama

#llm = model



# Tool integration
#tools = load_tools(['wikipedia'], llm=llm)

# Initialization of the agent
#agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
#print(agent)
"""
# Run the agent with a prompt
result = agent.run('What is the average age of a dog? Multiply the age by 3')

print(result)

"""
###################################################################################################
# simple
###################################################################################################
if False:
    from transformers import pipeline
    from langchain_community.tools import DuckDuckGoSearchRun
    #from langchain.chains.llm.LLMChain import LLMChain 
    from langchain_core.runnables import RunnableSequence
    from langchain.agents import LLMSingleActionAgent, AgentExecutor, AgentOutputParser, Tool



    # Use a Hugging Face Pipeline without authentication
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

    # Define a basic LLM wrapper (no login needed)
    class LocalLLM:
        def __call__(self, prompt):
            response = generator(prompt, max_new_tokens=80, do_sample=True, temperature=0.7) # max_length=114
            return response[0]["generated_text"]

    llm = LocalLLM()

    # Initialize search tool
    search = DuckDuckGoSearchRun()

    # Define tools
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Use this tool to search the web for relevant information."
        )
    ]

    # Convert tools to a string representation
    tools_str = "\n".join([f"Name: {tool.name}, Description: {tool.description}" for tool in tools])

    # ReAct-style prompt (simplified for article writing)
    prompt_template = (
        """
        You are an expert article writer. Write an article with 4 paragraphs
        and approximately 80 words about a given topic. Include its definition, 
        psychological effects, examples, and its role in human life.\n\n

        Tools available:
        {tools}
        
        Topic: {topic}\n\n
        
        Article:\n
        """
    )

    # Step 6: Create the RunnableSequence

    # Create the sequence of runnables (in this case, the prompt template and LLM)
    # The first step is to generate the article using the LLM
    article_generation = RunnableSequence(
        [prompt_template, llm]  # The prompt template will be followed by the LLM to generate text
    )
    # Step 6: Define the LLMChain to manage language model interaction
    #llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # Task description
    task_description = "Happiness"

    # Step 8: Create the agent (LLMSingleActionAgent)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=article_generation,
        output_parser=None,  # Assuming you don't need an output parser for simplicity
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # Step 9: Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )

    # Step 10: Test the agent with a task
    response = agent_executor.run("Generate a 4-paragraph article about the emotion 'happiness'.")
    print(response)

    # Generate the article prompt with tools
    #article_prompt = prompt_template.format(tools=tools_str, topic=task_description)
    #response = llm(article_prompt)

    # Output the article
    #print(response)

###################################################################################################
# best
###################################################################################################
if False:
    import re
    from typing import List, Union
    from langchain.agents import Tool
    from langchain import LLMChain, PromptTemplate
    from langchain.agents import LLMSingleActionAgent, AgentExecutor, AgentOutputParser
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_community.tools import DuckDuckGoSearchRun
    from transformers import pipeline
    from langchain.prompts import BaseChatPromptTemplate
    from langchain.schema import HumanMessage
    #from typing import List
    # Use an open-source Hugging Face model
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

    # Define a simple wrapper for the generator
    class LocalLLM:
        def __call__(self, prompt: str):
            response = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
            return response[0]["generated_text"]

    llm = LocalLLM()

    # Initialize search tool
    search = DuckDuckGoSearchRun()

    # Define tools
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Use this tool to search the web for relevant information."
        )
    ]

    # Define the custom prompt template
    template = """
    You are a helpful assistant capable of generating articles and providing detailed answers. 
    Use the tools provided to gather information when needed.

    Tools available:
    {tools}

    Task:
    {input}

    {agent_scratchpad}
    """

    class CustomPromptTemplate(BaseChatPromptTemplate):
        # The main template
        template: str
        # List of available tools
        tools: List[Tool]

        def format_messages(self, **kwargs) -> List[HumanMessage]:
            # Get intermediate steps (AgentAction, Observation tuples)
            intermediate_steps = kwargs.pop("intermediate_steps", [])
            # Construct agent_scratchpad by formatting intermediate steps
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            kwargs["agent_scratchpad"] = thoughts

            # Format the list of tools as a string
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

            # Format the template
            formatted = self.template.format(**kwargs)
            return [HumanMessage(content=formatted)]


    prompt = CustomPromptTemplate(template=template, tools=tools)

    # Define custom output parser
    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[Tool, str]:
            if "Final Answer:" in llm_output:
                return {
                    "output": llm_output.split("Final Answer:")[-1].strip(),
                    "log": llm_output
                }
            regex = r"Action: (.*?)\nAction Input: (.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2).strip(" ").strip('"')
            return {"tool": action, "input": action_input, "log": llm_output}

    output_parser = CustomOutputParser()

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Create the agent
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )

    # Setup conversational memory
    #memory = ConversationBufferWindowMemory(k=5)  # Retains the last 5 interactions

    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        #memory=memory,
        verbose=True
    )

    # Test the agent with a task
    response = agent_executor.run("Generate a 4-paragraph article about the emotion 'happiness'.")
    print(response)





if False:
    import os
    import pandas as pd
    #import pinecone
    import re
    from tqdm.auto import tqdm
    from typing import List, Union
    import zipfile

    from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
    from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
    from langchain import SerpAPIWrapper, LLMChain
    from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
    # LLM wrapper
    # Conversational memory
    from langchain.memory import ConversationBufferWindowMemory
    # Embeddings and vectorstore
    from langchain.embeddings.openai import OpenAIEmbeddings
    #from langchain.vectorstores import Pinecone

    # Initialize search tool
    search = DuckDuckGoSearchRun()

    # Set up Hugging Face LLM
    #hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
    #llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Define the prompt
    #prompt_template = PromptTemplate(
    #    input_variables=["emotion"],
    #    template=(
    #        "Please provide a comprehensive article about the emotion {emotion}. "
    #        "The article should include an introduction, background, detailed analysis, and conclusion."
    #    )
    #)

    # Create LLM chain
    #llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Define tools
    tools = [
        Tool(
            name="Search",
            func=search.invoke,
            description="Use this tool to search the web for relevant information."
        )#,
        #Tool(
        #    name="AnswerGenerator",
        #    func=llm_chain.run,
        #    description="Use this tool to generate detailed answers based on the emotion."
        #)
    ]

    # Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
    template = """
    Thought: I need to convert the time string into seconds.
    Action: { 
        "action": "convert_time",
        "action_input": {"time: "1:23:45
        }
    }
    Observation: {'seconds': '5035'}
    """

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
            
            # Check if agent should finish
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    # Return values is generally always a dictionary with a single `output` key
                    # It is not recommended to try anything else at the moment :)
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            
            # Parse out the action and action input
            regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            
            # If it can't parse the output it raises an error
            # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
            if not match:
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        
    output_parser = CustomOutputParser()


    #hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
    #llm = HuggingFacePipeline(pipeline=hf_pipeline)
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Using tools, the LLM chain and output_parser to make an agent
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )

    # Initiate the agent that will respond to our queries
    # Set verbose=True to share the CoT reasoning the LLM goes through
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    agent_executor.run("Explain the emotion happiness?")

    # Initialize the agent
    #agent = initialize_agent(
    #    tools=tools,
    #    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #    llm=llm,
    #    verbose=True
    #)

    # Test the agent
    #response = agent.invoke("Generate an article about happiness.")
    #print(response)

if False:

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
    #image_path = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\generated_image.png'  # Beispielhafte Bildpfad
    #emotion = detect_emotion(image_path)  # Emotion erkennen
    emotion = "Happy"
    article_text = generate_article(emotion)  # Artikel basierend auf der Emotion generieren

    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)
    # Ensure the output directory exists
    output_dir = os.path.join(current_dir, "generated_articles")
    os.makedirs(output_dir, exist_ok=True)
    # Define the output file path
    output_pdf_path = os.path.join(output_dir, "output.pdf")
    #output_pdf_path = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\sad_text.pdf'  # PDF-Ausgabepfad
    create_article_pdf_with_images_and_captions(emotion, article_text, output_pdf_path)
