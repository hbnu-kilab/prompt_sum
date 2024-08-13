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
                1) delimiter (구두점 및 구분자 .)로 끊지말고, 2) [원문] 길이에 비례하여 무조건 한 문장으로 요약해줘 길이는 30~200 글자 사이로. 3) 시간 정보(날짜, 요일 등)는 꼭 포함해서, 4) 말투는 존댓말까지 안 써도 돼. \
                입력 문단 앞에 [원문]이 주어지고 [요약] 뒤에 너가 요약문을 작성하면 돼. 항상 이 포맷을 잘 지켜주고. \
                negative sample인 counterfactual summary을 줄게 같이 보면서 틀리지 않게 정확하고 올바를 요약을 만들어줘. negative sapmle은 [negative sample] 뒤에 넣어줄게, 이건 생성하지마. \n\
                [원문]: {sent} \n \
                [Negative Sample]: {cda}\n \
                [요약]: ".split())

def mk_inst_for_counterfactual_summary(sent):
    return ' '.join(f"[원문]이 주어지면 개체명(named entity)를 기반으로 counterfactual summary를 [Counterfactual Summary] 뒤에 한 문장으로 요약해줘. 항상 이 포맷을 잘 지켜주고. \
                counterfactual summary를 만들때 개체명에 해당하는 단어는 같은 개체명 레이블에 해당하는 단어들 중 하나로 바꿔주고. \
                꼭 한문장으로 요약해주고. \n \
                [원문]: {sent} \n \
                [Counterfactual Summary]: ".split())