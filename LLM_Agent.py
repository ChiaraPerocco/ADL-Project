###################################################################################################
#
# LLM using agent
# https://medium.com/@aydinKerem/what-is-an-llm-agent-and-how-does-it-work-1d4d9e4381ca
# https://huggingface.co/blog/open-source-llms-as-agents
# pip install wikipedia
# pip install -U duckduckgo_search==5.3.1b1
# pip install --upgrade transformers langchain langchain_community
# pip install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate
# pip install transformers huggingface_hub

###################################################################################################
# Import packages
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

from langchain.agents import AgentExecutor, load_tools
from huggingface_hub import login
from langchain.agents import initialize_agent, AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import requests

# API Key Hugging Face
access_token = "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"

# Hugging Face API token for authentication
login(token=access_token)

if True:
    from typing import Dict, Any, Optional, Union
    from langchain.tools import BaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser

    from langchain_community.llms import HuggingFaceEndpoint
    from langchain_community.chat_models.huggingface import ChatHuggingFace

    llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

    chat_model = ChatHuggingFace(llm=llm)



    from langchain import hub
    from langchain.agents import AgentExecutor, load_tools
    from langchain.agents.format_scratchpad import format_log_to_str
    from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
    from langchain.tools.render import render_text_description
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_core.messages import AIMessage
    #from langchain_core.output_parsers import JsonOutputParser

    # Define external tools (Wikipedia and DuckDuckGo search as examples)
    #duckduckgo_search = DuckDuckGoSearchResults()
    import wikipedia
    tools = load_tools(["wikipedia"], llm = llm)
    #tools = []
    #search = DuckDuckGoSearchResults()
    #wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

    #search = DuckDuckGoSearchRun(api_wrapper=wrapper, source="news")
    #results = search.invoke("Sign Language")
    #print(results)
    #search.invoke("Obama")
    # Initialize DuckDuckGo search tool
    #search = DuckDuckGoSearchRun()

    # setup tools
    #tools = [
    #    #search  # DuckDuckGo tool for web search
        #wikipedia
    #]

    # Setup des ReAct-Style-Prompts
    prompt = hub.pull("hwchase17/react-json")
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )    

    # Definition des Agenten
    #chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        #| chat_model_with_stop
        | chat_model
        | ReActJsonSingleInputOutputParser()
    )

    # Instanziieren des AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True, handle_parsing_errors=True,
                                return_intermediate_steps = True)

    # Anfrage zum Thema "Gebärdensprache" und Generierung eines strukturierten Textes
    response = agent_executor.invoke(
        {
            "input": "Please create a detailed essay on the topic of Sign Language. The essay should have more than 1000 words and consist of 4 paragraphs, each building upon the previous one. The essay should cover the following aspects:\n"
                    "1. An introduction to sign language and its importance in communication.\n"
                    "2. The history and development of sign language.\n"
                    "3. The role of sign language in society, especially in education and inclusivity.\n"
                    "4. The future of sign language, technology's impact, and global awareness."
        }
    )

    print(response["intermediate step"])

if False:
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain.tools import BaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain.agents import AgentExecutor
    from langchain_core.runnables import RunnableLambda
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.agents.format_scratchpad import format_log_to_messages
    import wikipediaapi  # For Wikipedia API

    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Custom Wikipedia Tool
    class WikipediaTool(BaseTool):
        name = "wikipedia"
        description = "Fetch content from Wikipedia based on the query provided"

        def _run(self, query: str):
            wiki_wiki = wikipediaapi.Wikipedia('en')
            page = wiki_wiki.page(query)
            if page.exists():
                return page.text[:2000]  # Return the first 2000 characters
            else:
                return f"Page not found for query: {query}"

        async def _arun(self, query: str):
            raise NotImplementedError("Async is not implemented for this tool")
    tools = [WikipediaTool()]

    from typing import Dict, Any, Optional, Union
    from langchain.tools import BaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    from langchain_core.output_parsers import PydanticOutputParser

    class ToolParser(BaseModel):
        name: Optional[str] = Field(description="Tool name you are using")
        tool_args: Optional[Dict[str, Any]] = Field(description="Tool inputs args")
        
    class OutputParser(BaseModel):
        '''This is a output parser, Ensure you return only fields you use'''
        message: str = Field(description="answer to the user questions")
        tool_use: bool = Field(description="Return True if you are going to use tools, else return False")
        additional_kwargs: Union[Optional[ToolParser],None] = Field(description="Tool information that you want to invoke, Return None to this field if you don't need to use any tools")

    parser = PydanticOutputParser(pydantic_object=OutputParser)


    from langchain.tools.render import render_text_description_and_args
    from langchain_community.agent_toolkits import FileManagementToolkit
    # you can take any tools.

    #tools = FileManagementToolkit().get_tools()

    #tools_description = render_text_description_and_args(tools)

    from langchain.agents.agent import AgentFinish, AgentAction
    from langchain_core.runnables import RunnableLambda

    def agent_parser(parsed_data: OutputParser):
        if parsed_data.tool_use and parsed_data.additional_kwargs is not None:
            return AgentAction(
                tool=parsed_data.additional_kwargs.name,
                tool_input=parsed_data.additional_kwargs.tool_args,
                log="Agent action"
            )
        return AgentFinish(
            return_values = {
                "output": parsed_data.message
            },
            log= "Completed agent"
        )

    CustomAgentOutputParser = RunnableLambda(agent_parser)
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","You are good ai bot, you are provided with below tools: \n\n {tools_description} and \n\n {format_instruction}"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human","task: {input}"),
            MessagesPlaceholder("agent_scratchpad")
        ]
    )

    #prompt.partial_variables["tools_description"] = tools_description
    #prompt.partial_variables["format_instruction"] = parser.get_format_instructions()
    #prompt.input_variables= ["input","agent_scratchpad"]

    prompt.input_variables = ["input", "agent_scratchpad"]

    from langchain.agents.format_scratchpad import format_log_to_messages
    agent = (
        {
            "input": lambda x : x["input"],
            "agent_scratchpad": lambda x : format_log_to_messages(x["intermediate_steps"])
        } |
        prompt|
        llm |
        parser|
        CustomAgentOutputParser
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    response = executor.invoke({
        "input": "Fetch information about Sign Language and provide a summary."
        })

    print(response["output"])
#"""
import os
from fpdf import FPDF  # pip install fpdf

# Erstelle ein FPDF-Objekt
pdf = FPDF()

# Füge eine Seite hinzu
pdf.add_page()

# Setze die Schriftart
pdf.set_font("Arial", size=12)

# Angenommen, response ist ein Dictionary, das den Text unter einem bestimmten Schlüssel enthält
# Zum Beispiel: response = {"output": "Hier steht der generierte Text zum Thema Gebärdensprache..."}

# Schreibe den extrahierten Text in die PDF
pdf.multi_cell(0, 10, response["intermediate step"], align="c")

current_dir = os.path.dirname(__file__)
pdf_filename = input("Enter the desired PDF file name (without extension): ")
pdf.output(os.path.join(current_dir, "Article", f"{pdf_filename}.pdf"))

print("PDF wurde erfolgreich erstellt!")
#"""
