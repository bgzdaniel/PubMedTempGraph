import torch
from utils.modelling_utils import init, spellcheck_question, get_answer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init(device)
    chat(vectordb, chains, params)

def chat(vectordb, chains, params):
    while True:
        question = input("Please enter your question: ")
        if question == "exit":
            break
        question = spellcheck_question(question)
        while True:
            mode = input("Get an overview or get latest research? Type 'overview' or 'research': ")
            if mode == "overview" or mode == "research":
                print("", end="\n\n")
                get_answer(question, mode, vectordb, chains, params)
                break


if __name__ == '__main__':
    main()