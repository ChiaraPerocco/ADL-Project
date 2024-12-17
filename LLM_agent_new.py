if False: 
   # Import packages
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

    #from langchain.agents import AgentExecutor, load_tools
    from huggingface_hub import login
    #from langchain.agents import initialize_agent, AgentExecutor, Tool
    #from langchain.prompts import PromptTemplate
    #from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    #import requests

    # API Key Hugging Face
    access_token = "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"

    # Hugging Face API token for authentication
    login(token=access_token)

    #https://medium.com/@aqdasansari2024/level-up-your-llm-application-using-langchain-agents-688bc7fa7988
    #https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
    from langchain.agents import initialize_agent
    from langchain.tools import Tool
    from langchain.agents import AgentType

        # Configure logging
        #logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        #logger = logging.getLogger(__name__)

    # Use a more capable model for better results
    model_id = "EleutherAI/gpt-neo-2.7B"  # Use a larger model if feasible
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Set up the pipeline for text generation
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1500, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Initialize the Wikipedia tool manually
    wikipedia_tool = WikipediaAPIWrapper()

    # Create the list of tools (only using the Wikipedia tool here for example)
    tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="google search")]

    # Initialize the agent with appropriate configurations
    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iteration = 3
        # handle_parsing_errors=True  # Enable error handling for parsing errors
    )


    prompt = (
            "Please create a detailed essay on the topic of Sign Language. The essay should have at least 1000 words, "
            "divided into four paragraphs. Each paragraph should build upon the previous one and address the following aspects:\n\n"
            "1. An introduction to sign language and its importance in communication.\n"
            "2. The history and development of sign language.\n"
            "3. The role of sign language in society, particularly in education and inclusivity.\n"
            "4. The future of sign language, including technology's impact and global awareness.\n\n"
            "Ensure the essay is engaging, informative, and follows a logical flow."
    )

    # Invoke the agent to generate the essay
    result = agent_executor.invoke("Wo is the founder of google?")
    print(result)

if False:

    # Import packages
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

    #from langchain.agents import AgentExecutor, load_tools
    from huggingface_hub import login


    # API Key Hugging Face
    access_token = "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"

    # Hugging Face API token for authentication
    login(token=access_token)

        #https://medium.com/@aqdasansari2024/level-up-your-llm-application-using-langchain-agents-688bc7fa7988
        #https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
    from langchain.agents import initialize_agent
    from langchain.tools import Tool
    from langchain.agents import AgentType

            # Configure logging
            #logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
            #logger = logging.getLogger(__name__)

    # Use a more capable model for better results
    model_id = "gpt2"  # Use a larger model if feasible
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)


    # Create a pipeline for text generation
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Generate some text
    input_text = "Text about sign language"
    generated_text = generator(input_text, max_length=1500,min_length =1000, num_return_sequences=1)

    # Print the generated text
    print(generated_text[0]['generated_text'])

###################################################################################################
#
# Crate llm with agent from: https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain
#
###################################################################################################
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

import re
from typing import List, Union

# Langchain imports
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
#from langchain import SerpAPIWrapper, LLMChain
from langchain.chains import LLMChain
from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
from langchain_community.tools import WikipediaQueryRun
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# Hugging face imports
from huggingface_hub import login
from langchain_huggingface.llms import HuggingFacePipeline
#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline

# API Key Hugging Face
#access_token = "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"

# Hugging Face API token for authentication
#login(token=access_token)

# Initiate open-source LLM from Hugging Face
#llm_pipeline = pipeline("text-generation", model="facebook/opt-6.7b", device=-1)  # Adjust for your specific model
llm_pipeline = pipeline("text-generation", model="gpt2", device=-1)  # Adjust for your specific model

def llm_open_source(query: str):
    """Wrapper to interface with the open-source LLM pipeline."""
    result = llm_pipeline(query, max_length=500, do_sample=True)
    return result[0]['generated_text']


# Initialize the Wikipedia tool manually
wikipedia_tool = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Create the tools (only using the Wikipedia tool here for example)
tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information")]

# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

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

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

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


# Initiate our LLM
#model_id = "gpt2"  # Use a larger model if feasible
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id)

# Set up the pipeline for text generation
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1500, temperature=0.7)
#llm = HuggingFacePipeline(pipeline=pipe)


# LLM chain consisting of the LLM and a prompt
#llm_chain = LLMChain(llm=llm, prompt=prompt)


# Define LLM Chain
def llm_chain_fn(prompt: str):
    """Wrapper to interface with the open-source LLM for LangChain."""
    return llm_open_source(prompt)

llm_chain = LLMChain(llm=llm_chain_fn, prompt=prompt)


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

response = agent_executor.invoke("How many people live in canada as of 2023?")
print(response)