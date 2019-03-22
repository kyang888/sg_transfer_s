import jieba


def LiangciOk(question, answer):
    if "几" in question:
        index = question.rfind("几")
        if index == len(question) - 1:
            return True
        liangci = question[index + 1]
        if liangci not in answer:
            return False
    return True


def RankOK(question, answer):
    if ("第几" in question or "第多少" in question) and ("第" not in answer):
        return 0.5
    return 1.0


def CharOK(question, answer):
    if ("1个字" in question or "一个字" in question or "1字" in question or "一字" in question) and len(answer) != 1:
        # print(question)
        return False
    return True


def Head(anslen, answer):
    ret = 1
    if answer[:2] in question:
        ret *= 0.5
    if answer[-2:] in question:
        ret *= 2
    elif answer[-1:] in question:
        ret *= 1.5
    return ret


def LengthType(answer, anslen, ans_length):
    lentp = ans_length.GetType(len(answer))
    return anslen[lentp]


def Satisfy(question, answer):
    ret = 1
    if not LiangciOk(question, answer):
        ret *= 0.2
    if not CharOK(question, answer):
        ret = 0
    return ret * RankOK(question, answer)  # *Head(question,answer)


def QuestionContextOverlap(question, contextList):
    ret = 0
    questionTokens = jieba.lcut(question)
    for contextTokens in contextList:
        context = "".join(contextTokens)
        fz = 0.0
        for term in questionTokens:
            if term in contextTokens:
                fz += 1.0
        # ret=max(ret,fz/(len(questionTokens)+len(contextTokens)-fz))
        ret += fz / (len(questionTokens) + len(contextTokens) - fz)
    return ret


def QuestionAnswerOverlap(question, answer):
    fz = 0.0
    for term in answer:
        if term in question:
            fz += 1.0
    # ret=max(ret,fz/(len(questionTokens)+len(contextTokens)-fz))
    ret = fz / (len(question) + len(answer) - fz)
    return ret


def GetRuleFeature(question, answer, contextList, anslen=None, ans_length=None):
    ret = []
    ret.append(int(LiangciOk(question, answer)))
    ret.append(int(CharOK(question, answer)))
    ret.append(RankOK(question, answer))
    ret.append(LengthType(answer, anslen, ans_length))
    ret.append(QuestionContextOverlap(question, contextList))
    ret.append(QuestionAnswerOverlap(question, answer))
    return ret
