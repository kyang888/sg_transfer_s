import pickle, random, json


def json_file_to_dict(fileName):
    res = []
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dict = json.loads(s=line)
            res.append(dict)
    return res


def pickleWriter(fileName, obj):
    print('start writing file %s ...' % fileName)
    output = open(fileName, 'wb')
    pickle.dump(obj, output)
    output.close()
    print('finished writing file')

def pickleLoader(fileName):
    print('loading file %s' % fileName)
    pkl_file = open(fileName, 'rb')
    data = pickle.load(pkl_file)
    print('finished loading %s' % fileName)
    return data


def toDouList(doubleDict, threshold=0):
    transList = [[i, doubleDict[i]] for i in doubleDict if doubleDict[i] >= threshold]
    transList.sort(key=takeSecond, reverse=True)
    return transList

def takeSecond(elem):
    return elem[1]

def toSinList(doubleDict, threshold=0):
    transList = [i for i in doubleDict if doubleDict[i] >= threshold]
    return transList

def toDict(doubleDict, threshold=0):
    transDic = {k: v for k, v in doubleDict.items() if v >= threshold}
    return transDic


def randomPick(list, weights, num):
    picks = set()
    plist = [i for i in list if i in weights.keys()]
    modiWeights = [weights[i] for i in plist]
    table = [z for x,y in zip(plist, modiWeights) for z in [x] * y]
    while len(picks) < min(len(plist), num):
        picks.add(random.choice(table))
    print('picks: ', picks)
    return picks

def loadAndInvertId2X(fileName):
    return {v: k for k, v in enumerate(pickleLoader(fileName))}


def matchSubstr(u, v):
    u, v = u[-50:], v[-50:]
    lu, lv = len(u), len(v)
    g = getMatchingMatrix(u, v)
    rr, x, y = [], lu, lv
    while x > 0 and y > 0:
        gg = g[x][y]
        if gg == 3: rr.append(u[x-1])
        x -= gg & 1
        y -= (gg & 2) // 2
    return ''.join(reversed(rr))

def getMatchingMatrix(u, v):
    u, v = u[-50:], v[-50:]
    lu, lv = len(u), len(v)
    f, g = [[0] * 51 for i in range(51)], [[0] * 51 for i in range(51)]
    f[0][0] = 0
    for i in range(1, lu + 1):
        for j in range(1, lv + 1):
            f[i][j] = max(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1] + (1 if u[i - 1] == v[j - 1] else 0))
            if f[i - 1][j] == f[i][j]:
                g[i][j] = 1
            elif f[i][j - 1] == f[i][j]:
                g[i][j] = 2
            else:
                g[i][j] = 3
    return g


def strFormatCleaner(oriStr, mode='default'):
    if mode == 'default':
        oriStr = oriStr.rstrip('?？')
        replace = [['，', ','], ['、', ','], ['：', ':'], ['“', '"'], ['”', '"'], ['；', ';'], ['％', '%'],['\xa0', ''],
                   ['？', '?'], ['’', "'"], ['‘', "'"], ['（', '('], ['）', ')'], [' ', ''], ['！', '!'], ['.', '。']]
        for i in replace:
            oriStr = oriStr.replace(i[0], i[1])
    return oriStr.lower()

def isPureEng(str, mode='default'):
    if mode == 'default':
        ch = ['.', '_', '-', '/', '*', '(', ')', '+', ':', ';', '×', '／', '～', '=', '￥', '$', '!',
          '"', "'", ',', '<', '>', '《', '》', '·', '%', '°', '′', '－', '?', '…', '。']
    else:
        ch = ['。', '_', '-', '/', '*', ':', ';', '%', '×', '=', '.', '\'', '?', ',', '(', ')', '+', '－']
    for item in str.lower():
        if 'a' <= item <= 'z' or item in ch or '0' <= item <= '9':
            continue
        return False
    else:
        return True
