from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor
from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
from langchain_community.tools import WikipediaQueryRun

# Template für den Prompt
template = """Question: {question}

Answer: Let's think step by step."""

#template = """Thought: {question}

#Action: Let's think step by step and generate the requested content."""

# Prompt erstellen
prompt = ChatPromptTemplate.from_template(template)

# LLM Modell (Ollama) verwenden
model = OllamaLLM(model="llama3.1")

# Wikipedia API Wrapper für das Wikipedia Tool
# Initialize the Wikipedia tool manually
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Ein Agent wird erstellt, der mit dem Wikipedia-Tool interagieren kann
tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="Use this tool to search for information in Wikipedia.")]

# Agent initialisieren
agent = initialize_agent(
    tools=tools, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    llm=model,
    verbose=True,
    handle_parsing_errors=True
)

# Die Frage stellen
#question = "Tell me about American Sign Language and describe how the letter 'A' is signed. Include other facts as well."

question = """
Write a detailed text about American Sign Language (ASL). The text should include the following:
1. An explanation of how the letter 'A' is signed in ASL.
2. A general overview of ASL.
3. Interesting facts about ASL.
Ensure that the text is structured into 4 paragraphs and contains more than 1000 words.
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
