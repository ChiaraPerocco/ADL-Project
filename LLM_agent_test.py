import os
import re
from typing import List, Union
from fpdf import FPDF  # Für die PDF-Generierung
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.chains import LLMChain
from huggingface_hub import login
from langchain_ollama import OllamaLLM
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain_huggingface.llms import HuggingFacePipeline
import transformers
import torch

if True:
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' 
    # Get the path of current_dir
    current_dir = os.path.dirname(__file__)

    # API Key Hugging Face
    access_token = "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"

    # Hugging Face API token for authentication
    login(token=access_token)

    # Hugging Face Modell und Tokenizer laden
    #model_id = "openai-community/gpt2-large" # Beispielmodell, ggf. anpassen
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Text-Generierungspipeline erstellen
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    #pipe = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    #print(pipe("Test prompt"))

    # Langchain-Tools konfigurieren
    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
    duckduckgo_tool = DuckDuckGoSearchResults(api_wrapper=duckduckgo_wrapper)

    tools = [
        Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information"),
        Tool(name="DuckDuckgo", func=duckduckgo_tool.invoke, description="Search for information")
    ]


    tool_names = [tool.name for tool in tools]

    # Custom Prompt Template für LangChain Agenten
    class CustomPromptTemplate(BaseChatPromptTemplate):
        template: str
        tools: List[Tool]
        
        def format_messages(self, **kwargs) -> str:
            intermediate_steps = kwargs.pop("intermediate_steps")
            detected_letter = kwargs.pop("detected_letter", "")
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            
            kwargs["agent_scratchpad"] = thoughts
            kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
            #formatted = self.template.format(**kwargs)
            #return [HumanMessage(content=formatted)]
            
            # Template mit detected_letter korrekt formatieren
            formatted = self.template.format(detected_letter=detected_letter, **kwargs)
            return [HumanMessage(content=formatted)]

    # Custom Output Parser für Agenten-Ausgaben
    class CustomOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            try:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
            except Exception as e:
                print(f"Error parsing LLM output: {e}")
                return AgentFinish(
                    return_values={"output": "Unable to parse the output. Please refine your query."},
                    log=llm_output,
                )
    output_parser = CustomOutputParser()

    # LLM und LangChain Chain erstellen
    llm = HuggingFacePipeline(pipeline=pipe)
    
    template = """
    Schreibe einen prägnanten Artikel über die Bedeutung des Buchstabens "{detected_letter}" in der Gebärdensprache. Der Artikel soll in vier Abschnitte gegliedert sein:

    1. **Einleitung**: Was symbolisiert der Buchstabe "{detected_letter}"?
    2. **Der Buchstabe in der Schriftsprache**: Welche Rolle spielt der Buchstabe im Alphabet und in der Sprache?
    3. **Der Buchstabe in der Gebärdensprache**: Wie wird der Buchstabe in der Gebärdensprache dargestellt?
    4. **Fazit**: Was verbindet den Buchstaben mit Schriftsprache und Gebärdensprache?

    ### Antwortformat:
    Frage: {input}

    1. Verwende die Tools, um Informationen zu sammeln.
    2. Erstelle einen kurzen, informativen Artikel basierend auf den gesammelten Daten.

    {agent_scratchpad}
    """

    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    
    #print(llm("Test prompt"))
    #llm = OllamaLLM(model="llama3.1")

    # LLM chain consisting of the LLM and a prompt
    #llm_chain = LLMChain(llm=llm, prompt=prompt, verbose = True)

    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        early_stopping_method="generate"
    )

    # Agent-Executor erstellen
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # --------------------------- Beispiel: Artikel generieren ---------------------------

    # Chatverlauf initialisieren
    chat_history = []

    detected_letter = "A"  # Beispielbuchstabe

    # Artikel-Prompt
    article_prompt = f"""
    Schreibe einen prägnanten Artikel über den Buchstaben „{detected_letter}“ und seine Bedeutung in der Gebärdensprache. Der Artikel sollte diese vier Abschnitte beinhalten:

    1. **Einleitung**: Was symbolisiert der Buchstabe „{detected_letter}“ und seine Bedeutung in verschiedenen Kontexten?
    2. **Der Buchstabe in der Schriftsprache**: Rolle und Verwendung des Buchstabens im Alphabet und in Wörtern.
    3. **Der Buchstabe in der Gebärdensprache**: Wie wird der Buchstabe in der Deutschen Gebärdensprache (DGS) dargestellt? Beschreibe die Handform.
    4. **Fazit**: Verbinde Schriftsprache und Gebärdensprache und reflektiere über die Rolle des Buchstabens „{detected_letter}“.

    Der Artikel soll klar und prägnant sein, ohne unnötige Details.
    """


    # Artikel generieren
    response_article = agent_executor.invoke({"input": article_prompt, "chat_history": chat_history, "detected_letter": detected_letter, "max_new_tokens": 100})
    print(response_article)
    article_output = response_article["output"] if "output" in response_article else "Failed to generate the article."
    print(article_output)

    # Chatverlauf aktualisieren
    chat_history.append({"role": "user", "content": article_prompt})
    chat_history.append({"role": "assistant", "content": article_output})

    # Artikel als PDF speichern
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Artikel: Gebärdensprache', 0, 1, 'C')
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Seite {self.page_no()}', 0, 0, 'C')

    # PDF erstellen
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, article_output)

    # PDF speichern
    pdf_output_path = os.path.join(current_dir, "Article", "Artikel_Gebaerdensprache.pdf")
    pdf.output(pdf_output_path)

    print(f"\nPDF gespeichert unter: {pdf_output_path}")

    # Artikel anzeigen
    print("\nGenerated Article:")
    print(article_output)

if False:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import LLMResult
    from langchain.llms.base import LLM
    from typing import List
    from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor

    # Hugging Face GPT-2 Modell laden
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from langchain.llms.base import LLM
    from typing import List

    class GPT2LLM(LLM):
        def __init__(self, model_name="openai-community/gpt2-large"):
            # GPT-2 Modell und Tokenizer initialisieren
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        def _call(self, prompt: str, stop: List[str] = None) -> str:
            # Eingabe für das Modell vorbereiten
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            # Text generieren
            outputs = self.model.generate(**inputs, max_length=1024, num_return_sequences=1)
            
            # Ergebnis dekodieren
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Stop-Bedingungen anwenden
            if stop:
                for stop_token in stop:
                    result = result.split(stop_token)[0]
            
            return result.strip()

        @property
        def _identifying_params(self) -> dict:
            return {"model_name": "openai-community/gpt2-large"}

        @property
        def _llm_type(self) -> str:
            return "custom_gpt2"

    # GPT-2 Modell initialisieren
    llm = GPT2LLM()

    # Wikipedia Tool konfigurieren
    from langchain.tools import WikipediaQueryRun, WikipediaAPIWrapper

    wikipedia_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    tools = [
        Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information")
    ]

    tool_names = [tool.name for tool in tools]

    # Prompt für den Agenten
    template = """
    You are an intelligent assistant that can use tools to answer questions.
    The tools available are:
    {tools}

    When answering questions, decide which tool to use and think step-by-step.

    Question: {input}
    {agent_scratchpad}
    """

    # Agent erstellen
    from langchain.prompts import PromptTemplate
    from langchain.agents import LLMSingleActionAgent
    from langchain.prompts.chat import HumanMessage

    prompt = PromptTemplate(input_variables=["tools", "input", "agent_scratchpad"], template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    class CustomAgentOutputParser:
        def parse(self, text: str):
            from langchain.schema import AgentAction, AgentFinish
            if "Final Answer:" in text:
                return AgentFinish({"output": text.split("Final Answer:")[-1].strip()}, text)
            elif "Action:" in text:
                action = text.split("Action:")[1].split("\n")[0].strip()
                observation = text.split("Action Input:")[1].strip()
                return AgentAction(action, observation)
            else:
                raise ValueError(f"Could not parse: {text}")

    output_parser = CustomAgentOutputParser()

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
        early_stopping_method="generate"
    )

    # Agent Executor erstellen
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # Testeingabe
    response = agent_executor.run(input="What is the capital of France?")
    print(response)
