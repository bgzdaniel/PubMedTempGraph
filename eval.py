from utils.modelling_utils import init, get_answer

def main():
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init()
    eval(vectordb, chains, params)

def eval(vectordb, chains, params):
    questions = None # TODO: get questions from csv
    answers = []
    for question in questions:
        mode = "research"
        answer = get_answer(question, mode, vectordb, chains, params)
        answers.append(answer)

    # TODO: evaluate on question types

if __name__ == '__main__':
    main()