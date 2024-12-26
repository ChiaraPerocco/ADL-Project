from langchain_ollama.llms import OllamaLLM
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from typing import List, Union
import time
import re

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

# Define the prompt template for the agent
template = '''Answer the following questions as best you can. You have access to the following tools:{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!

Question: {input}
Thought:{agent_scratchpad}'''

# Create the prompt from the template
prompt = PromptTemplate.from_template(template)

# LLM
model = OllamaLLM(model="llama3.1", temperature=0.7)


# Create the React agent with the prompt and tools
search_agent = create_react_agent(model, tools, prompt)

# Set up the AgentExecutor with the agent and tools
agent_executor = AgentExecutor(
    agent=search_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)


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

    # Define the prompt template
    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

    # Create the prompt
    prompt = PromptTemplate.from_template(template)

    # Initialize the LLM model (Ollama)
    model = OllamaLLM(model="llama3.1")

    # Initialize the Wikipedia tool
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    # Initialize the DuckDuckGo tool with retry logic
    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
    duckduckgo_tool_with_retry = DuckDuckGoToolWithRetry(api_wrapper=duckduckgo_wrapper)

    # Define tools
    tools = [
        Tool(name="Wikipedia", func=wikipedia_tool.run, description="Use this tool to search for information in Wikipedia."),
        Tool(name="DuckDuckGo", func=duckduckgo_tool_with_retry.invoke, description="Search for information using DuckDuckGo.")
    ]

    # Create the React agent
    agent = create_react_agent(model, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                max_iterations=3, handle_parsing_errors=True)

    # Define the question (prompt to generate the article)
    question = """
    Write a detailed text in string format about American Sign Language (ASL). The text should include the following:
    1. An explanation of how the letter 'A' is signed in ASL.
    2. A general overview of ASL.
    3. Interesting facts about ASL.
    Ensure that the text is structured into 4 paragraphs and contains more than 1000 words and the format is a string.
    """

    # Use the agent executor to answer the question
    response = agent_executor.invoke({"input": question})
    print(response)



if False:
    from langchain.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM
    from langchain.agents import AgentExecutor, create_react_agent, AgentOutputParser, Tool, AgentType
    from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
    #from langchain.agents import initialize_agent, 
    from langchain.agents import AgentExecutor
    from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchResults
    import re
    from typing import List, Union

    # Template für den Prompt
    #template = """Question: {question}

    #Answer: Let's think step by step."""

    # Template for the Prompt
    template = """
    Question: {question}

    Thought: Think step by step, but stop if the information is straightforward and sufficient to answer the question.
    Action: Provide the answer or take necessary action if more information is required.
    """

    # Prompt erstellen
    prompt = ChatPromptTemplate.from_template(template)

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

    # LLM Modell (Ollama) verwenden
    model = OllamaLLM(model="llama3.1")

    # Wikipedia API Wrapper für das Wikipedia Tool
    # Initialize the Wikipedia tool manually
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
    duckduckgo_tool = DuckDuckGoSearchResults(api_wrapper=duckduckgo_wrapper)


    # Ein Agent wird erstellt, der mit dem Wikipedia-Tool interagieren kann
    tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="Use this tool to search for information in Wikipedia."),
            Tool(name="DuckDuckgo", func=duckduckgo_tool.invoke, description="Search for information")]

    # Agent initialisieren
    agent = create_react_agent(
        tools=tools, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        #agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        llm=model,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations = 2,
        early_stopping_method = "generate"
    )

    # Die Frage stellen
    #question = "Tell me about American Sign Language and describe how the letter 'A' is signed. Include other facts as well."

    question = """
    Write a detailed text in string format about American Sign Language (ASL). The text should include the following:
    1. An explanation of how the letter 'A' is signed in ASL.
    2. A general overview of ASL.
    3. Interesting facts about ASL.
    Ensure that the text is structured into 4 paragraphs and contains more than 1000 words and the format is a string.
    """

    # Verwenden des Agenten zur Beantwortung der Frage
    response = agent.invoke(question)
    print(response)

if False: 
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="llama3.1")

    chain = prompt | model

    response = chain.invoke({"question": "Description of letter A in sign language."})
    print(response)
