import evaluate
from bleurt import score
import numpy as np

class Scorer:
    def __init__(self, device):
        self.device = device
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        checkpoint = "data/BLEURT-20-D12"
        self.bleurt = score.BleurtScorer(checkpoint)

        # init bertscore with sample computation
        self.bertscore.compute(
            predictions=["predictions"],
            references=["references"],
            lang="en",
            model_type="distilbert-base-uncased",
            device=self.device,
        )

    def get_scores(self, predictions, references):
        bleuscore = self.bleu.compute(
            predictions=predictions,
            references=references
        )["bleu"]
        rougescore = np.mean(list(self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True,
        ).values()))
        bertscore = np.mean(
            self.bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                model_type="distilbert-base-uncased",
                device=self.device,
            )["f1"]
        )
        bleurtscore = np.mean(
            self.bleurt.score(references=references, candidates=predictions)
        )

        return {
            "bleuscore": bleuscore,
            "rougescore": rougescore,
            "bertscore": bertscore,
            "bleurtscore": bleurtscore,
        }
    
def weighted_score(scores):
    weighted_score = (
        scores["bleuscore"] * 0.1
        + scores["rougescore"] * 0.1
        + scores["bertscore"] * 0.4
        + scores["bleurtscore"] * 0.4
    )
    return weighted_score

def get_overall_scores(ranked_question_types_scores):
    overall_bleuscore = np.mean([scores["bleuscore"] for _, scores in ranked_question_types_scores.items()])
    overall_rougescore = np.mean([scores["rougescore"] for _, scores in ranked_question_types_scores.items()])
    overall_bertscore = np.mean([scores["bertscore"] for _, scores in ranked_question_types_scores.items()])
    overall_bleurtscore = np.mean([scores["bleurtscore"] for _, scores in ranked_question_types_scores.items()])
    return {
            "bleuscore": overall_bleuscore,
            "rougescore": overall_rougescore,
            "bertscore": overall_bertscore,
            "bleurtscore": overall_bleurtscore,
        }

def output_scores(ranked_question_types_scores, overall_scores, path):
    with open(f"data/{path}", "w") as file:
        for question_type, scores in ranked_question_types_scores.items():
            file.write(f"{question_type}:\n")
            for score_type, score in scores.items():
                file.write(f"\t{score_type}: {score:.4f}\n")
            file.write(f"weighted score: {weighted_score(scores):.4f}")
            file.write("\n\n")
        file.write(f"overall results:\n")
        file.write(f"\tbleuscore: {overall_scores['bleuscore']:.4f}\n")
        file.write(f"\trougescore: {overall_scores['rougescore']:.4f}\n")
        file.write(f"\tbertscore: {overall_scores['bertscore']:.4f}\n")
        file.write(f"\tbleurtscore: {overall_scores['bleurtscore']:.4f}\n")
        file.write(f"\tweighted score: {overall_scores['weighted_score']:.4f}\n")