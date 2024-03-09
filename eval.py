import torch
from utils.modelling_utils import init, get_answer
from utils.score_utils import Scorer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init(device)
    scorer = Scorer(device=device)
    eval(vectordb, chains, params, scorer)

def eval(vectordb, chains, params, scorer: Scorer):
    questions = None # TODO: get questions from csv
    answers = []
    for question in questions:
        mode = "research"
        answer = get_answer(question, mode, vectordb, chains, params)
        answers.append(answer)

    # TODO: evaluate on question types, see
    # https://github.com/KennyLoRI/pubMedNLP/blob/main/kedronlp/scripts/evaluation/valid_and_eval.py#L218-L261
    # for example

if __name__ == '__main__':
    main()