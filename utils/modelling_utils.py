from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores.chroma import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.embedding_utils import get_langchain_chroma
from utils.prompt_utils import get_research_prompt, get_overview_prompt, get_text2date_prompt
from utils.stdout_utils import StdoutSwitch
import ast
import re
from typing import Any
from spellchecker import SpellChecker


def init_llm(
    temperature=0.3,
    max_tokens=1000,
    n_ctx=2048,
    top_p=1,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=True,
    path="data/mistral-7b-instruct-v0.2.Q6_K.gguf",
):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        verbose=verbose,  # Verbose is required to pass to the callback manager
    )
    llm.client.verbose = False
    print("\n\n\n")
    return llm


def init(device):
    llm = init_llm()
    vectordb_path = "data/chroma_store"
    vectordb = get_langchain_chroma(device=device, persist_dir=vectordb_path)

    text2date_prompt = PromptTemplate(
        template=get_text2date_prompt(),
        input_variables=["question"]
    )

    overview_prompt = PromptTemplate(
        template=get_overview_prompt(),
        input_variables=["context"]
    )

    research_prompt = PromptTemplate(
        template=get_research_prompt(),
        input_variables=["context", "question"]
    )

    chains = {
        "text2date_chain": LLMChain(prompt=text2date_prompt, llm=llm),
        "overview_chain": LLMChain(prompt=overview_prompt, llm=llm),
        "research_chain": LLMChain(prompt=research_prompt, llm=llm),
    }
    return vectordb, chains

def get_answer(
        question: str,
        mode: str,
        vectordb: Chroma,
        chains: dict[str, LLMChain],
        params: dict[str, Any]
    ) -> str:
    stdout = StdoutSwitch()

    text2date_chain = chains["text2date_chain"]
    overview_chain = chains["overview_chain"]
    research_chain = chains["research_chain"]

    filters = []
    if mode == "overview":
        stdout.off()
        format_incorrect = True
        while format_incorrect:
            try:
                date_range_str = text2date_chain.invoke({"question": question})["text"].strip()
                date_range = ast.literal_eval(date_range_str)
                format_incorrect = False
            except IndentationError:
                format_incorrect = True
        stdout.on()

        for year in date_range:
            filter = {
                "year": {
                    "$eq": year
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
    
    answer = []
    for filter in filters:
        docs = vectordb.similarity_search(question, k=params["top_k"], filter=filter)
        docs = [doc.page_content for doc in docs]
        context = "\n\n".join(docs)
        if mode == "overview":
            stdout.off()
            answer.append(overview_chain.invoke({
                "context": context,
                "question": question,
            })["text"].strip())
            stdout.on()
            print(answer[-1])
        elif mode == "research":
            stdout.off()
            answer.append(research_chain.invoke({
                "context": context,
                "question": question,
            })["text"].strip())
            stdout.on()
            print(answer[-1])
    return "\n".join(answer)

def spellcheck_question(question: str) -> str:
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
    return question