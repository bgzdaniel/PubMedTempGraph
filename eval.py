import torch
from utils.modelling_utils import init, get_answer
from utils.score_utils import Scorer
from utils.eval_utils import weighted_score
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init(device)
    scorer = Scorer(device=device)
    eval(vectordb, chains, params, scorer)

def eval(vectordb, chains, params, test_set_df, scorer: Scorer):
    answers = []
    question_type_scores = {}
    #prepare test_set dictionar
    types = test_set_df["Question Type"].unique()
    test_set = {}
    for a_type in types:
        questions_answers = test_set_df.loc[test_set_df["Question Type"] == a_type][
            ["Question", "Answer", "Source"]
        ]
        test_set[a_type] = questions_answers.to_dict(orient="records")

    # Eval per question in question_type
    for question_type, question_answer_pairs in test_set.items():
        questions = [qa_pair['Question'] for qa_pair in question_answer_pairs]
        references = [qa_pair['Answer'] for qa_pair in question_answer_pairs]
        for question in questions:
            mode = "research"
            answer = get_answer(question, mode, vectordb, chains, params)
            answers.append(answer)
        question_type_scores[question_type] = scorer.get_scores(predictions=answers, references=references)

    # compute aggregates results
    ranked_question_types_scores = dict(sorted(question_type_scores.items(), key=lambda x: weighted_score(x[1]), reverse=True))
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

    #write results into texxt file
    with open (f"research_eval_scores.txt", "w") as file:
        file.write(f"combination:\n")
        for param, param_value in params.items():
            file.write(f"\t{param}: {param_value}\n")
        file.write("\n\n")
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