from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def init_llm(
    temperature=0,
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
