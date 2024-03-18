import torch
from utils.modelling_utils import init_chains, spellcheck_question, get_answer
from utils.embedding_utils import PubMedBert

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "top_k": 3,
        # ...
    }
    chains = init_chains()
    embedding_model = PubMedBert(device=device)
    chat(chains, embedding_model, params)

def chat(chains, embedding_model, params):
    print("Welcome to the PubMed Chat application! Type 'exit' to close the program.")
    while True:
        question = input("Please enter your question: ")
        if question == "exit":
            break
        question = spellcheck_question(question)
        while True:
            mode = input("Get an overview or get latest research? Type 'overview' or 'research': ")
            if mode == "overview" or mode == "research":
                print("", end="\n\n")
                get_answer(question, mode, chains, embedding_model, params)
                print("", end="\n\n")
                break


if __name__ == '__main__':
    main()