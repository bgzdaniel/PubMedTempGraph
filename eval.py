import torch
import numpy as np
from utils.modelling_utils import init, get_answer
from utils.eval_utils import Scorer, weighted_score
from langchain_community.vectorstores.chroma import Chroma

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init(device)
    scorer = Scorer(device=device)
    eval(vectordb, chains, params, scorer)

def eval(vectordb: Chroma, chains, params, scorer: Scorer):
    question_type_set = {}
    question_types_scores = {}
    for question_type, qa_list in question_type_set.items():
        for qa in qa_list:
            question = qa["generated_question"]
            gold_answer = qa["generated_answer"]
            predicted_answer = get_answer(question, "research", vectordb, chains, params)
            question_types_scores[question_type] = scorer.get_scores(
                predictions=[predicted_answer], references=[gold_answer]
            )

    ranked_question_types_scores = dict(sorted(question_types_scores.items(), key=lambda x: weighted_score(x[1]), reverse=True))


    overall_bleuscore = np.mean([scores["bleuscore"] for _, scores in ranked_question_types_scores.items()])
    overall_rougescore = np.mean([scores["rougescore"] for _, scores in ranked_question_types_scores.items()])
    overall_bertscore = np.mean([scores["bertscore"] for _, scores in ranked_question_types_scores.items()])
    overall_bleurtscore = np.mean([scores["bleurtscore"] for _, scores in ranked_question_types_scores.items()])
    overall_weighted_score = weighted_score(
        {
            "bleuscore": overall_bleuscore,
            "rougescore": overall_rougescore,
            "bertscore": overall_bertscore,
            "bleurtscore": overall_bleurtscore,
        })

    with open("data/eval_results.txt", "w") as file:
        for question_type, scores in ranked_question_types_scores.items():
            file.write(f"{question_type}:\n")
            for score_type, score in scores.items():
                file.write(f"\t{score_type}: {score:.4f}\n")
            file.write(f"weighted score: {weighted_score(scores):.4f}")
            file.write("\n\n")
        file.write(f"overall results:\n")
        file.write(f"\tbleuscore: {overall_bleuscore:.4f}\n")
        file.write(f"\trougescore: {overall_rougescore:.4f}\n")
        file.write(f"\tbertscore: {overall_bertscore:.4f}\n")
        file.write(f"\tbleurtscore: {overall_bleurtscore:.4f}\n")
        file.write(f"\tweighted score: {overall_weighted_score:.4f}\n")

if __name__ == '__main__':
    main()