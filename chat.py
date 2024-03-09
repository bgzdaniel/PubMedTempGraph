from utils.modelling_utils import init, spellcheck_question, get_answer

def main():
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init()
    chat(vectordb, chains, params)

def chat(vectordb, chains, params):
    print("print user information") # TODO: add user information
    while True:
        question = input("Your question: ")
        if question == "exit":
            break
        question = spellcheck_question(question)
        mode = input("Get an overview or get latest research? Type 'overview' or 'research': ", end="\n\n")
        get_answer(question, mode, vectordb, chains, params)

if __name__ == '__main__':
    main()