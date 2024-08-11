from rouge import Rouge

def rouge(dec, ref):
    if dec == '' or ref == '':
        return 0.0
    rouge = Rouge()
    scores = rouge.get_scores(dec, ref)
    return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3