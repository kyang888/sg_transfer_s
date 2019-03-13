import os, jieba, random, h5py
import numpy as np
import utils
from settings import Settings
setting = Settings()

CLEAN_DIR = os.path.join('data', 'clean')
HELPER_DIR = os.path.join('data', 'helper')
DATASET_DIR = os.path.join('data', 'dataset')
if not os.path.isdir(DATASET_DIR):
    os.makedirs(DATASET_DIR)

qcache = {}
c2id = {}
w2id = {}

xps = setting.xps


def wrapper(tdata, datasetDir=os.path.join(DATASET_DIR, 'dataset_plen50.h5')):
    # random.shuffle(tdata)
    print(len(tdata))
    div = int(len(tdata) * 0.05)
    te, tr = tdata[:div], tdata[div:]
    utils.pickleWriter(os.path.join(DATASET_DIR, 'validation_raw_plen50.pkl'), te)
    utils.pickleWriter(os.path.join(DATASET_DIR, 'train_raw_plen50.pkl'), tr)
    random.shuffle(tr)
    tXs, tY = MakeMatrix(te, 0, len(te))
    XXs, YY = MakeMatrix(tr, 0, len(tr))
    with h5py.File(datasetDir, 'w') as dfile:
        dfile.create_dataset('YY', data=YY)
        dfile.create_dataset('tY', data=tY)
        for i in range(len(xps)):
            dfile.create_dataset('XXs%d' % i, data=XXs[i])
            dfile.create_dataset('tXs%d' % i, data=tXs[i])


def MakeMatrix(tdata, low, high):
    num = high - low
    YY = np.zeros((2, num, setting.passage_len))
    XXs = [
            np.zeros((num, xps[0][0], xps[0][1]), dtype='int32'),
            np.zeros((num, xps[1][0]), dtype='int32'),
            np.zeros((num, xps[2][0], xps[2][1]), dtype='int32'),
            np.zeros((num, xps[3][0]), dtype='int32')
           ]
    for k, lln in enumerate(tdata[low:high]):
        YY[0][k] = lln[0][0]
        YY[1][k] = lln[0][1]
        for i in range(len(xps)):
            if len(lln[i+1]) == 0: continue
            llns = lln[i+1]
            le = xps[i][0]
            if len(xps[i]) > 1:
                llns = llns[:le]
                for kk, vv in enumerate(llns):
                    for k2, v2 in enumerate(vv):
                        XXs[i][k][le-len(llns)+kk][xps[i][1] - len(vv) + k2] = v2
            else:
                llns = llns[:le]
                for kk, vv in enumerate(llns):
                    XXs[i][k][le-len(llns)+kk] = vv
    return XXs, YY


posi_exp_cnt = 0
neg_exp_cnt = 0
def get_answer_arr(train_data):
    global posi_exp_cnt, neg_exp_cnt
    answer = set(train_data['answer'])
    passage_text = train_data['passage']['passage_text']
    y0 = np.zeros(setting.passage_len)
    y1 = np.zeros(setting.passage_len)
    if len(train_data['answer_idx']) > 0:
        for i in range(len(train_data['answer_idx'])):
            s_idx = train_data['answer_idx'][i]['s_idx']
            e_idx = train_data['answer_idx'][i]['e_idx']
            y0[s_idx] = 1
            y1[e_idx] = 1
            posi_exp_cnt += 1
    else:
        neg_exp_cnt += 1
        for i in range(setting.passage_len):
            mratio = 0.5
            for j in range(i + 1, setting.passage_len):
                if j - i > 10: break
                ans = set(passage_text[i: j])
                vcomm = len(ans.intersection(answer))
                vratio = vcomm / (len(ans) + len(answer) - vcomm + 1e-10)
                if vratio > mratio and vratio > y0[i] and vratio > y1[j]:
                    mratio = vratio
                    y0[i] = y1[j - 1] = vratio
    return y0, y1


def get_c_ids(tokens):
    global setting
    tokens = tokens[:setting.char_feature_len]
    ret = []
    for tok in tokens:
        c_l = [c2id.get(char, 1) for char in tok]
        ret.append(c_l[:10])
    return ret


if __name__ == '__main__':
    training_set = utils.pickleLoader(os.path.join(CLEAN_DIR, 'teared_train_factoid_plen50.pkl'))
    w2id = utils.pickleLoader(os.path.join(HELPER_DIR, 'w2id.pkl'))
    c2id = utils.pickleLoader(os.path.join(HELPER_DIR, 'c2id.pkl'))
    tokenized_traing_set = []
    for train_data in training_set:
        passage_toks = train_data['passage']['passage_toks']
        query_toks = train_data['query']['query_toks']
        y0, y1 = get_answer_arr(train_data)
        passage_w_ids = [w2id.get(tok, 1) for tok in passage_toks]
        passage_c_ids = get_c_ids(passage_toks)
        query_w_ids = [w2id.get(tok, 1) if tok in w2id else 1 for tok in query_toks]
        query_c_ids = get_c_ids(query_toks)
        tokenized_traing_set.append([[y0, y1], query_c_ids, query_w_ids, passage_c_ids, passage_w_ids, train_data])
    print('shape of training data: ', np.shape(tokenized_traing_set))
    print('posi: %d, neg: %d'%(posi_exp_cnt, neg_exp_cnt))
    print(neg_exp_cnt / posi_exp_cnt)
    wrapper(tokenized_traing_set)
