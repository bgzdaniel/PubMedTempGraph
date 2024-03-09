import ast
import torch
import regex as re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.embedding_utils import get_langchain_chroma
from utils.modelling_utils import init_llm
from utils.stdout_utils import StdoutSwitch
from utils.prompt_utils import get_research_prompt, get_overview_prompt, get_text2date_prompt
from spellchecker import SpellChecker

def main():
    params = {
        "top_k": 3,
        # ...
    }

    llm = init_llm()
    vectordb_path = "data/chroma_store"
    vectordb = get_langchain_chroma(device=device, persist_dir=vectordb_path)

    chat(llm, vectordb, params)

def chat(llm, vectordb, params):
    stdout = StdoutSwitch()
    
    text2date_prompt = PromptTemplate(
        template=get_text2date_prompt(),
        input_variables=["question"]
    )
    text2date_chain = LLMChain(prompt=text2date_prompt, llm=llm)

    chat_prompt = PromptTemplate(
        template=get_overview_prompt(),
        input_variables=["context", "question"]
    )
    overview_chain = LLMChain(prompt=chat_prompt, llm=llm)

    research_prompt = chat_prompt = PromptTemplate(
        template=get_research_prompt(),
        input_variables=["context", "question"]
    )
    research_chain = LLMChain(prompt=research_prompt, llm=llm)

    print("print user information") # TODO: add user information

    while True:
        question = input("Your question: ")
        if question == "exit":
            break
        # Treat highlighted words and qustion mark specifically
        pattern = r'\*(.*?)\*'  # Regular expression to match words enclosed in *
        highlighted_words = re.findall(pattern, question)
        question_mark = '?' if '?' in question else ''

        # Apply spell correction excluding asterisked words
        try:
            spell = SpellChecker()
            corrected_list = [spell.correction(token.strip('?').strip('*')) if token.strip('?').strip(
                '*') not in highlighted_words else token.strip('?').strip('*') for token in question.split()]
            question = ' '.join(corrected_list) + question_mark
        except:
            pass

        mode = input("Get an overview or get latest research? Type 'overview' or 'research': ")

        filters = []
        if mode == "overview":
            stdout.off()
            date_range_str = text2date_chain.invoke({"question": question})["text"].strip()
            stdout.on()
            date_range = ast.literal_eval(date_range_str)
            for year in date_range:
                filter = {
                    "year": {
                        "$eq": str(year) # TODO: change to int after rerunning vec2chroma
                    }
                }
                filters.append(filter)
        elif mode == "research":
            date_range = [2023, 2022, 2021, 2020, 2019] # last 5 years
            filter = {
                "year": {
                    "$in": date_range
                }
            }
            filters.append(filter)
                
        for filter in filters:
            docs = vectordb.max_marginal_relevance_search(question, k=params["top_k"], filter=filter)
            docs = [doc.page_content for doc in docs]
            context = "\n\n".join(docs)
            if mode == "overview":
                overview_chain.invoke({
                    "context": context,
                    "question": question,          
                })
            elif mode == "research":
                research_chain.invoke({
                    "context": context,
                    "question": question,
                })
    
if __name__ == '__main__':
    main()