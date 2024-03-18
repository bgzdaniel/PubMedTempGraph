from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.embedding_utils import get_langchain_chroma, PubMedBert
from utils.prompt_utils import get_research_prompt, get_overview_prompt, get_text2date_prompt
from utils.stdout_utils import StdoutSwitch
import ast
import re
from typing import Any
from spellchecker import SpellChecker
import os


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


def init_chains():
    llm = init_llm()

    text2date_prompt = PromptTemplate(
        template=get_text2date_prompt(),
        input_variables=["question"]
    )

    overview_prompt = PromptTemplate(
        template=get_overview_prompt(),
        input_variables=["context", "question"]
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
    return chains

def get_answer(
        question: str,
        mode: str,
        chains: dict[str, LLMChain],
        embedding_model: PubMedBert,
        params: dict[str, Any],
    ) -> str:
    stdout = StdoutSwitch()
    text2date_chain = chains["text2date_chain"]
    overview_chain = chains["overview_chain"]
    research_chain = chains["research_chain"]

    vectordb_paths = []
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
            vectordb_paths.append(f"data/chroma_store_{year}")

    elif mode == "research":
        vectordb_paths.append("data/chroma_store")
    
    answer = []
    for vectordb_path in vectordb_paths:
        if os.path.isdir(vectordb_path):
            vectordb = get_langchain_chroma(model=embedding_model, persist_dir=vectordb_path)
        else:
            answer = "necessary chroma store for specified years is missing!"
            print(answer)
            return answer
        docs = vectordb.max_marginal_relevance_search(question, k=params["top_k"])
        docs = [doc.page_content for doc in docs]
        context = "\n\n".join(docs)
        if mode == "overview":
            stdout.off()
            answer.append(overview_chain.invoke({
                "context": context,
                "question": question,
            })["text"].strip())
            stdout.on()
            year = int(vectordb_path[-4:])
            print(f"{year}: {answer[-1]}")
        elif mode == "research":
            answer.append(research_chain.invoke({
                "context": context,
                "question": question,
            })["text"].strip())
    return "\n".join(answer)