from nltk import word_tokenize


# Following QMSum
def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens


# Following QMSum
def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text

def clean_data_ko(text):
    text = text.replace("[|endofturn|]", '')
    return text

def postprocess_text(pred, label):
    return pred.strip(), label.strip()


def postprocess_text_batch(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
