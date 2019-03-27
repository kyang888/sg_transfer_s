import os, random, h5py
import numpy as np
import utils
from settings import Settings
setting = Settings()

CLEAN_DIR = os.path.join('data', 'clean')
HELPER_DIR = os.path.join('data', 'helper')
DATASET_DIR = os.path.join('data', 'dataset')
if not os.path.isdir(DATASET_DIR):
    os.makedirs(DATASET_DIR)

c2id = {}
w2id = {}

xps = setting.xps

posi_exp_cnt = 0
neg_exp_cnt = 0


def tokenize_dataset(dataset):
    tokenized_dataset = []
    for data in dataset:
        passage_toks = data['passage']['passage_toks']
        query_toks = data['query']['query_toks']
        y0, y1 = get_answer_arr(data)
        query_c_ids = get_c_ids(query_toks)
        query_w_ids = [w2id.get(tok, 1) for tok in query_toks]
        passage_c_ids = get_c_ids(passage_toks)
        passage_w_ids = [w2id.get(tok, 1) for tok in passage_toks]
        query_c_feature = data['query']['query_char_features']
        query_w_feature = data['query']['query_context_features']
        passage_c_feature = data['passage']['passage_char_features']
        passage_w_features = data['passage']['passage_context_features']

        rec = [[y0, y1],
               query_c_ids, query_w_ids,
               passage_c_ids, passage_w_ids,
               query_c_feature, query_w_feature,
               passage_c_feature, passage_w_features,
               data
               ]
        tokenized_dataset.append(rec)
    return tokenized_dataset


def ComputeAnswerIndex(tokens, answer, answers=None):
    ret_start = np.zeros(setting.passage_len)
    ret_end = np.zeros(setting.passage_len)
    for i in range(len(tokens)):
        if i >= setting.passage_len: break
        tans = ""
        for j in range(i, len(tokens)):
            if j >= setting.passage_len: break
            tans += tokens[j]
            if tans == answer:
                ret_start[i] = 1
                ret_end[j] = 1
            if (answers is not None) and (tans in answers):
                ret_start[i] = 1
                ret_end[j] = 1
        # if len(tans) >= len(answer) * 2: break
    # return ret_start, ret_end
    if sum(ret_start) == 0:
        achrs = [set(answer)] + [] if answers is None else [set(a) for a in answers]
        for i in range(len(tokens)):
            if i >= setting.passage_len: break
            tans = ""
            for j in range(i, len(tokens)):
                if j >= setting.passage_len: break
                tans += tokens[j]
                chrs = set(tans)
                vratio = 0
                for aa in achrs:
                    vcomm = len(chrs.intersection(aa))
                    vratio = max(vratio, vcomm / (len(chrs) + len(aa) - vcomm + 1e-10))
                if vratio > 0.5 and vratio > ret_start[i] and vratio > ret_end[j]:
                    ret_start[i] = ret_end[j] = vratio
            # if len(tans)>=len(answer) * 3: break
    return ret_start, ret_end

qidAnswers = {}

def read_question_answers():
    fnlist = ['data/qid_answer_expand', 'data/qid_answer_expand.train',  'data/qid_answer_expand.valid']
    for fn in fnlist:
        for tokens in utils.LoadCSV(fn):
            if len(tokens) != 3: continue
            qid = tokens[0]
            answers = tokens[2].split('|')
            qidAnswers[qid] = set(answers)


def get_answer_arr(train_data):
    answers = qidAnswers.get(str(train_data['query']['query_id']), None)
    answer = set(train_data['answer'])
    tokens = train_data['passage']['passage_toks']
    y0, y1 = ComputeAnswerIndex(tokens, answer, answers)
    return y0, y1


def get_c_ids(tokens):
    global setting
    tokens = tokens[:setting.char_feature_len]  # to be modified!!!
    ret = []
    for tok in tokens:
        c_l = [c2id.get(char, 1) for char in tok]
        ret.append(c_l[:10])  # change to settings.length!!!
    return ret


def wrapper(tdata, vdata, dataset_dir=os.path.join(DATASET_DIR, 'dataset_plen%i.h5'%setting.passage_len)):
    te, tr = tdata, vdata
    utils.pickleWriter(os.path.join(DATASET_DIR, 'validation_raw_plen%i.pkl'%setting.passage_len), te)
    utils.pickleWriter(os.path.join(DATASET_DIR, 'train_raw_plen%i.pkl'%setting.passage_len), tr)
    random.shuffle(tr)
    tXs, tY = MakeMatrix(te, 0, len(te))
    XXs, YY = MakeMatrix(tr, 0, len(tr))
    with h5py.File(dataset_dir, 'w') as dfile:
        dfile.create_dataset('YY', data=YY)
        dfile.create_dataset('tY', data=tY)
        for i in range(len(xps)):
            dfile.create_dataset('XXs%d' % i, data=XXs[i])
            dfile.create_dataset('tXs%d' % i, data=tXs[i])


def MakeMatrix(tdata, low, high):
    num = high - low
    YY = np.zeros((2, num, setting.passage_len))
    XXs = []
    for xp in xps:
        if len(xp) == 1:
            XXs.append(np.zeros((num, xp[0]), dtype='int32'))
        elif len(xp) == 2:
            XXs.append(np.zeros((num, xp[0], xp[1]), dtype='float64'))
        else:
            XXs.append(np.zeros((num, xp[0], xp[1], xp[2]), dtype='int32'))

    for k, lln in enumerate(tdata[low:high]):
        YY[0][k] = lln[0][0]
        YY[1][k] = lln[0][1]
        for i in range(len(xps)):
            if len(lln[i+1]) == 0: continue
            llns = lln[i+1]
            le = xps[i][0]
            if len(xps[i]) == 1:
                llns = llns[:le]
                for kk, vv in enumerate(llns):
                    XXs[i][k][le-len(llns)+kk] = vv
            elif len(xps[i]) == 2:
                llns = llns[:le]
                for kk, vv in enumerate(llns):
                    for k2, v2 in enumerate(vv):
                        XXs[i][k][le-len(llns)+kk][xps[i][1] - len(vv) + k2] = v2
            else:
                XXs[i][k] = llns
    return XXs, YY


if __name__ == '__main__':
    read_question_answers()
    training_set = utils.pickleLoader(os.path.join(CLEAN_DIR, 'teared_train_factoid_plen%i.pkl'%setting.passage_len))
    validation_set = utils.pickleLoader(os.path.join(CLEAN_DIR, 'teared_validation_factoid_plen%i.pkl'%setting.passage_len))
    w2id = utils.pickleLoader(os.path.join(HELPER_DIR, 'w2id_plen%i.pkl'%setting.passage_len))
    c2id = utils.pickleLoader(os.path.join(HELPER_DIR, 'c2id_plen%i.pkl'%setting.passage_len))

    tokenized_training_set = tokenize_dataset(training_set)
    tokenized_validation_set = tokenize_dataset(validation_set)

    print('shape of training data: ', np.shape(tokenized_training_set))
    # print('posi: %d, neg: %d'%(posi_exp_cnt, neg_exp_cnt))
    # print(neg_exp_cnt / posi_exp_cnt)
    wrapper(tokenized_training_set, tokenized_validation_set)
