from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def rouge(dec, ref):
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return scores, (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

def bleu(hypothesis, ref):
    """
    Calculate BLEU score between a reference and a hypothesis.

    :param reference: List of reference sentences (each a list of words)
    :param hypothesis: Hypothesis sentence (a list of words)
    :return: BLEU score
    """
    if hypothesis == '' or ref == '':
        return 0.0
    
    smoothing_function = SmoothingFunction().method4
    return sentence_bleu([ref.split()], hypothesis.split(), smoothing_function=smoothing_function)
