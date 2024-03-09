
def weighted_score(scores):
    weighted_score = (
        scores["bleuscore"] * 0.1
        + scores["rougescore"] * 0.1
        + scores["bertscore"] * 0.4
        + scores["bleurtscore"] * 0.4
    )
    return weighted_score