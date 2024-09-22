def mk_inst_for_summary_w_1shot(sent, prev_gold_sum=""):
    return ' '.join(f"너는 정말 똑똑한 친구이고 요약을 정말 잘 해. \
                전체 맥락에서 주요한 내용만 잘 찾아서 요약하는 걸 좋아하지. \
                다음 주어진 글에 대해서 요약해줘. 말투는 존댓말까지 안 써도 돼. \
                입력 문단 앞에 [원문]이 주어지고 [요약] 뒤에 너가 요약문을 작성하면 돼. 항상 이 포맷을 잘 지켜주고. \
                예제 요약을 줄게 이런 스타일로 요약해줘. 예제는 [예제 요약] 뒤에 넣어줄게. \
                [예제 요약]: {prev_gold_sum} \n \
                [원문]: {sent} \n [요약]: ".split())

def mk_inst_for_summary2(sent):
    return ' '.join(f"너는 정말 똑똑한 친구이고 요약을 정말 잘 해. \
                전체 맥락에서 주요한 내용만 잘 찾아서 요약하는 걸 좋아하지. \
                다음 주어진 글에 대해서 구두점으로 끊지말고 한 문장으로 요약하고, 100글자 이상으로 요약해줘. 말투는 존댓말까지 안 써도 돼. \
                입력 문단 앞에 [원문]이 주어지고 [요약] 뒤에 너가 요약문을 작성하면 돼. 항상 이 포맷을 잘 지켜주고. \
                [원문]: {sent} \n [요약]: ".split())


def mk_inst_for_summary(sent):
    return ' '.join(f"너는 정말 똑똑한 친구이고 요약을 정말 잘 해. 전체 맥락에서 주요한 내용만 잘 찾아서 요약해야 해. \
                주어진 글에 대해서 다음 조건을 지켜줘. \
                1) delimiter (구두점 및 구분자 .)로 끊지말고, 2) [원문] 길이에 비례하여 무조건 한 문장으로 요약해줘 길이는 30~200 글자 사이로. 3) 시간 정보(날짜, 요일 등)는 꼭 포함해서, 4) 말투는 존댓말까지 안 써도 돼. \
                입력 문단 앞에 [원문]이 주어지고 [요약] 뒤에 너가 요약문을 작성하면 돼. 항상 이 포맷을 잘 지켜주고. \
                [원문]: {sent} \n [요약]: ".split())

def mk_inst_for_summary_w_cda(sent, cda):
    return ' '.join(f"너는 정말 똑똑한 친구이고 요약을 정말 잘 해. 전체 맥락에서 주요한 내용만 잘 찾아서 요약해야 해. \
                주어진 글에 대해서 다음 조건을 지켜줘. \
                1) delimiter (구두점 및 구분자 .)로 끊지말고, 2) [원문] 길이에 비례하여 무조건 한 문장으로 요약해줘 길이는 30~150 글자 사이로. 3) 시간 정보(날짜, 요일 등)는 꼭 포함해서, 4) 말투는 존댓말까지 >안 써도 돼. \
                주어진 [원문]과 잘못된 요약인 [Negative Sample]을 확인하여 올바른 요약문을 만들어줘. [요약] 뒤에 너가 요약문을 작성하면 돼. \
                [Negative Sample]과 요약에 대한 설명은 생성하지마. \n\
                [원문]: {sent} \n \
                [Negative Sample]: {cda}\n \
                [요약]: ".split())

def mk_inst_for_summary_w_cda_en(sent, cda):
    return ' '.join(f"You are a very smart friend and really good at summarizing. You need to find and summarize only the key points from the overall context. For the given text, please follow these conditions \
        Do not break the summary with delimiters (punctuation marks or separators such as periods). \
        The summary should be in one sentence, proportionate to the length of the original text, and between 30 to 150 characters. \
        Include time information (dates, days, etc.). \
        You don't need to use polite language. \
        Check the original text and the incorrect summary Negative Sample and create the correct summary. After Summary, write your summary. Do not generate explanations for the Negative Sample and summary. \
        [Input Text]: {sent} \n \
        [Negative Sample]: {cda} \n \
        [Summary]: ".split())

def mk_inst_for_counterfactual_summary(sent):
    return ' '.join(f"주어진 [원문]에 대하여 잘못된 요약(counterfactual)을 만들어줘. [에러 요약] 뒤에 요약문을 생성해주면 돼.\
                요약은 한 문장으로 만들고, 개체명에 해당하는 단어는 같은 원문에서 찾아서 같은 개체명에 해당하는 단어들 중 하나로 바꿔주고. 1) 전체 생성은 100 글자 이하로 한정하고 2) 원문, 설명은 생성하지마. \n \
                [원문]: {sent} \n \
                [에러 요약]: ".split())

def mk_inst_for_counterfactual_summary_en(sent):
    return ' '.join(f"Create counterfactual summary for the given [input text]. After the [error summary], generate a summary. \
        You should write one sentence of the summary and replace named entities with other entities from the original text. 1) Limit the entire generation to under 100 characters, and 2) do not include the input text and any explanation. \n \
        [Input Text]: {sent} \n \
        [Error Summary]: ".split())

# def mk_inst_for_counterfactual_summary(sent):
#     return ' '.join(f"[원문]이 주어지면 개체명(named entity)를 기반으로 counterfactual summary를 [Counterfactual Summary] 뒤에 한 문장으로 요약해줘. 항상 이 포맷을 잘 지켜주고. \
#                 counterfactual summary를 만들때 개체명에 해당하는 단어는 같은 개체명 레이블에 해당하는 단어들 중 하나로 바꿔주고. \
#                 꼭 한문장으로 요약해주고. \n \
#                 [원문]: {sent} \n \
#                 [Counterfactual Summary]: ".split())

def mk_inst_etri_augmentation(sent):
    return f"""주어진 입력 문장을 아래 22가지 변형 유형(augmentation type)에 따라 변형해줘.
<order>:
1. 의미가 크게 벗어나선 안됩니다.
2. [변형 유형] 이 주워집니다.
3. [변형 유형] <생성한 문장> 과 같이 공백이나 추가 줄바꿈 없이 대괄호 이어서 바로 작성해야 합니다.
4. 입력 문장은 다시 말할 필요 없고, 순서 번호 안 써도 됩니다.

<format>:
입력 문장: {sent}

<augmentation type>
[동의어 가능 모든 단수 명사 문장 안에 표시] <s>단어</s>
[문장 구조 변경]
[구어체 변형]
[어순 변형]
[감정 강조]
[부정 표현 추가]
[질문형으로 변형]
[감정 추가]
[디테일 추가]
[상반된 상황 표현]
[피동형 사용]
[무작위성 도입]
[비유적 표현 추가]
[반어법 사용]
[주어를 강조]
[상황 설명 추가]
[시제 변경]
[복합문으로 변형]
[간결한 표현으로 축약]
[강조 표현 사용]
[유머 추가]
[청중에게 질문하는 방식]"""