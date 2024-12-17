if True:
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
        from langchain.agents import initialize_agent, AgentType
        from langchain_huggingface.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper  
        from langchain.tools import Tool
        from langchain.agents import AgentExecutor, load_tools, create_react_agent
        from langchain.agents.format_scratchpad import format_log_to_str
        #from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
        #from langchain.agents.output_parsers import ReActTextOutputParser
        from langchain.agents.react.output_parser import ReActOutputParser
        from langchain_community.utilities import WikipediaAPIWrapper  # Explicitly importing the Wikipedia tool
        from langchain import hub
        import wikipedia
        from langchain.tools.render import render_text_description
        from langchain_community.llms import HuggingFaceEndpoint
        from langchain_community.chat_models.huggingface import ChatHuggingFace

        # Use a more capable model for better results
        #model_id = "EleutherAI/gpt-neo-2.7B"  # Use a larger model if feasible
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        #model = AutoModelForCausalLM.from_pretrained(model_id)

        # Set up the pipeline for text generation
        #pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1500, temperature=0.7)
        #llm = HuggingFacePipeline(pipeline=pipe)

        llm = HuggingFaceEndpoint(repo_id = "HuggingFaceH4/zephyr-7b-beta")
        chat_model = ChatHuggingFace(llm = llm)
        # Define external tool: Wikipedia
        #import wikipedia
        #tools = load_tools(["wikipedia"], llm = llm)

        # Initialize the Wikipedia tool manually
        wikipedia_tool = WikipediaAPIWrapper()
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        # Create the list of tools (only using the Wikipedia tool here for example)
        #tools = [Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search Wikipedia for information")]
        
        from langchain.tools import Wikipedia

        # Define the Wikipedia tool correctly
        wikipedia_tool = Tool(name="Wikipedia", func=Wikipedia().run, description="Search Wikipedia for information")

        tools = [wikipedia_tool]
        
        # Generate tool names from the tools list
        #tool_names = ", ".join([tool.name for tool in tools])

        #tools = load_tools(["wikipedia"], llm = llm)
        # Definition des Agenten
        #chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
        # Define agent
        #agent_executor = initialize_agent(
        #    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #    tools=tools,
        #    llm=llm,
        #    #verbose=True,
        #    #return_intermediate_steps=True,
        #    #max_iteration = 3
        #)

        #prompt = hub.pull("hwchase17/react-json")
        #prompt = prompt.partial(
        #    tools = render_text_description(tools),
        #    tool_names = ", ".join([t.name for t in tools])
        #)
        #prompt = (
        #    "Please create a detailed essay on the topic of Sign Language. The essay should have at least 1000 words, "
        #    "divided into four paragraphs. Each paragraph should build upon the previous one and address the following aspects:\n\n"
        #    "1. An introduction to sign language and its importance in communication.\n"
        #    "2. The history and development of sign language.\n"
        #    "3. The role of sign language in society, particularly in education and inclusivity.\n"
        #    "4. The future of sign language, including technology's impact and global awareness.\n\n"
        #    "Ensure the essay is engaging, informative, and follows a logical flow."
        #)

        # Bind the stop parameter to the chat model
        chat_model_with_stop = chat_model.bind(stop=["\nObservation"])

        # Custom prompt
        custom_prompt_template = """
            You are a helpful assistant equipped with the following tools:

            {tools}

            You can use these tools by invoking them as needed. Think step by step, and when you need to use a tool, do the following:
            1. Explain why the tool is needed.
            2. Call the tool.
            3. Observe the tool's response.
            4. Continue reasoning based on the results.

            When you decide to use a tool, format your response as follows:
            Action: Wikipedia
            Action Input: <input_for_the_tool>

            When you provide the final answer, format it as follows:
            Final Answer: <your_answer>

            Respond to the user based on the results of your reasoning.
            Tools: {tool_names}

            {agent_scratchpad}
        """

        # Create a prompt with required variables
        prompt = PromptTemplate(
            input_variables=["tools", "tool_names", "agent_scratchpad"],
            template=custom_prompt_template,
        )


        if True:
            #agent = (
            #    {
            #        "input": lambda x: x["input"],
            #        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            #    }
            #    | prompt
            #    | chat_model_with_stop
            #    | chat_model
            #    #| ReActJsonSingleInputOutputParser()
            #    | ReActOutputParser()  # Nutze hier den Text-basierten Parser
            #)

            # Create a ReAct Agent using the tools and a text-based parser
            agent = initialize_agent(
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                llm=chat_model,
                tools=tools,
                #prompt=prompt
                verbose = True
            )

            # Wrap the agent in an executor
            #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


            # Instanziieren des AgentExecutor
            #agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True, handle_parsing_errors=True)

            # Anfrage zum Thema "Gebärdensprache" und Generierung eines strukturierten Textes
            #response = agent_executor.invoke(
            #    {
            #         "input": "What is sign language?"
                    #"input": "Please create a detailed essay on the topic of Sign Language. The essay should have more than 1000 words and consist of 4 paragraphs, each building upon the previous one. The essay should cover the following aspects:\n"
                    #        "1. An introduction to sign language and its importance in communication.\n"
                    #        "2. The history and development of sign language.\n"
                    #        "3. The role of sign language in society, especially in education and inclusivity.\n"
                    #        "4. The future of sign language, technology's impact, and global awareness."
            #    }
            #)
        
        #response = agent_executor.invoke(
        #    {
        #        "input": "Please create a detailed essay on the topic of Sign Language."
        #    }
        #)

        # Print intermediate steps to debug
        #print(response["intermediate_steps"])

        # Print the essay
        #print(response["output"])
        #print(response)
    # Invoke the agent
    response = agent.invoke({"input": "What is the history of american sign language?",
                             "handle_parsing_errors": True})
    print(response)

    # Get the final output (this is the agent's answer)
    final_answer = response.get("output", "No answer received.")

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
    pdf.multi_cell(0, 10, final_answer, align="c")

    current_dir = os.path.dirname(__file__)
    pdf_filename = input("Enter the desired PDF file name (without extension): ")
    pdf.output(os.path.join(current_dir, "Article", f"{pdf_filename}.pdf"))

    print("PDF wurde erfolgreich erstellt!")
    #"""
