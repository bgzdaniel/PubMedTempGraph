import torch
from utils.modelling_utils import init, get_answer
from utils.eval_utils import Scorer, weighted_score, get_overall_scores, output_scores
from langchain_community.vectorstores.chroma import Chroma
import tensorflow as tf
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "top_k": 3,
        # ...
    }
    vectordb, chains = init(device)
    tf.config.set_visible_devices([], 'GPU') # run BleuRT on CPU, save GPU memory
    scorer = Scorer(device=device)
    eval(vectordb, chains, params, scorer)

def eval(vectordb: Chroma, chains, params, scorer: Scorer):
    df = pd.read_csv("data/eval_dataset.csv").fillna("")
    types = df["question_type"].unique()
    question_type_set = {}
    for a_type in types:
        questions_answers = df.loc[df["question_type"] == a_type][
            ["generated_question", "generated_answer", "gpt_answer_without_context"]
        ]
        question_type_set[a_type] = questions_answers.to_dict(orient="records")

    question_types_scores = {}
    question_types_scores_gpt = {}
    for question_type, qa_list in question_type_set.items():
        for qa in qa_list:
            question = qa["generated_question"]
            gold_answer = qa["generated_answer"]
            gpt_answer = qa["gpt_answer_without_context"]
            print(f"question: {question}")
            predicted_answer = get_answer(question, "research", vectordb, chains, params)

            question_types_scores[question_type] = scorer.get_scores(
                predictions=[predicted_answer], references=[gold_answer]
            )
            question_types_scores_gpt[question_type] = scorer.get_scores(
                predictions=[gpt_answer], references=[gold_answer]
            )

    ranked_question_types_scores = dict(sorted(question_types_scores.items(), key=lambda x: weighted_score(x[1]), reverse=True))
    ranked_question_types_scores_gpt = dict(sorted(question_types_scores_gpt.items(), key=lambda x: weighted_score(x[1]), reverse=True))
    
    overall_scores = get_overall_scores(ranked_question_types_scores)
    overall_scores["weighted_score"] = weighted_score(overall_scores)
    overall_scores_gpt = get_overall_scores(ranked_question_types_scores_gpt)
    overall_scores_gpt["weighted_score"] = weighted_score(overall_scores_gpt)

    output_scores(ranked_question_types_scores, overall_scores, "eval_results_dbks.txt")
    output_scores(ranked_question_types_scores_gpt, overall_scores_gpt, "eval_results_gpt.txt")

if __name__ == '__main__':
    main()