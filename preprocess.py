# -*-coding: utf-8 -*-
import jieba, os, re, random
import editdistance
import numpy as np
import utils
from settings import Settings

setting = Settings()

TRAINING_FILE = os.path.join('data', 'train_factoid_1.json')
VALIDATE_FILE = os.path.join('data', 'valid_factoid.json')
W2V_FILE = os.path.join('data', 'sgns.zhihu.bigram-char')
CLEAN_DIR = os.path.join('data', 'clean')
HELPER_DIR = os.path.join('data', 'helper')
if not os.path.isdir(HELPER_DIR):
    os.makedirs(HELPER_DIR)
if not os.path.isdir(CLEAN_DIR):
    os.makedirs(CLEAN_DIR)

word_dic = {}
char_dic = {}
id2w = []
id2c = []
c2id = {}
w2id = {}
id2vec = {}


def clean_passages(passages):
    cleaned_passages = []
    for passage in passages:
        passage_text = passage['passage_text']
        passage_text = clean_text(passage_text)
        passage_toks = get_toks(passage_text)
        if len(passage_toks) > 2 * setting.passage_len:
            t1 = passage_toks[:setting.passage_len]
            t2 = passage_toks[setting.passage_len: 2* setting.passage_len]
            p1 = ''.join(t1)
            p2 = ''.join(t2)
            passage['passage_text'] = p1
            passage['passage_toks'] = t1
            cleaned_passages.append(passage)
            passage['passage_text'] = p2
            passage['passage_toks'] = t2
            cleaned_passages.append(passage)
        else:
            passage['passage_toks'] = passage_toks[:setting.passage_len]
            passage['passage_text'] = ''.join(passage['passage_toks'])
            cleaned_passages.append(passage)
    return cleaned_passages


def clean_text(text):
    text = text.lower()
    text = text.replace('\u3000', ':')
    text = re.sub(r'\s+', ' ', text)
    text = DBC2SBC(text)
    text = mark_transfer(text)
    text = text.replace('...', '')
    return text


def mark_transfer(text):
    marks = [['，', ','], ['。', '.'], ['“', '"'], ['”', '"'], ['：', ':'], ['？', '?'], ['！', '!'],
             ['%', '%'], ['‘', '\''], ['’', '\''], ['．', ''], ['、', ','], ['【', '['], ['】', ']'],
             ['（', '('], ['）', ')'], ['＂', '"'], ['＞', '>'],['…', ''], ['；', ';'], ['☁', ''],
             ['╗', ''], ['↑', ''], ['✔', ''], ['::', ':'], [':-:', ''], ['``', '`'], ['>>', '>'], ['.:', '.'],
             ['☀', ''], ['✈', ''], ['◆', ''], ['█', ''], ['→', ''], ['▪', ''], ['--', '-'], [':*:a:', 'a']]
    for item in marks:
        text = text.replace(item[0], item[1])
    return text


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
    return rstring


def get_toks(text):
    tokens = jieba.cut(text)
    ret_lis = []
    # 将英文、数字拆分为单个字母和数字
    for token in tokens:
        if token.isdigit() or re.match('^[a-zA-Z]+$', token):
            for i in token:
                ret_lis.append(i)
        else:
            ret_lis.append(token)
    return ret_lis


def get_w2id(tokens, isquery=False):
    global word_dic
    for i in tokens:
        if isquery:
            word_dic[i] = word_dic.get(i, 0) + 10
        else:
            word_dic[i] = word_dic.get(i, 0) + 1


def get_c2id(text, isquery=False):
    global char_dic
    for i in text:
        if isquery:
            char_dic[i] = char_dic.get(i, 0) + 10
        else:
            char_dic[i] = char_dic.get(i, 0) + 1


def get_id2vec_dict():
    global id2vec
    print('loading pretrained word embedding ...')
    with open(W2V_FILE, 'r', encoding='utf-8') as f:
        data = f.readline()
        print('pretrained word embedding size:　', data)
        while data:
            s = data.split()
            word = s[0]
            try:
                vec = [float(x) for x in s[1:]]
            except:
                data = f.readline()
                continue
            if word in w2id:
                id = w2id[word]
                id2vec[id] = vec
            data = f.readline()


def save_id2vec_data(id2vec_dict):
    global id2w
    vec = np.random.randn(len(id2w), setting.w_emb_size)
    for id in id2vec_dict.keys():
        vec[id] = id2vec_dict[id]
    print('vec size: ', vec.shape)
    np.save(os.path.join(HELPER_DIR, 'vec_plen%i.npy'%setting.passage_len), vec)


def get_teared_dataset(dataset):
    teared_dataset = []
    random.shuffle(dataset)
    cnt = 0
    print('start tearing %i examples in dataset' % len(dataset))
    for data in dataset:
        cnt += 1
        if cnt % 1000 == 0:
            print('-'*10, 'got to %i piece of data' % cnt, '-'*10)
        passages = data['passages']
        passage_list = [pas['passage_toks'] for pas in passages]
        answer = data['answer']
        query = data['query']
        query_id = data['query_id']
        query_toks = data['query_toks']
        for passage in passages:
            pas = passage['passage_text']
            pas_toks = passage['passage_toks']
            item = {
                'query': {
                    'query_id': query_id,
                    'query_text': query,
                    'query_toks': query_toks,
                    'query_context_features': get_context_features(pas, query_toks, setting.query_len),
                    'query_char_features': get_char_features(query_toks, setting.query_len, setting.word_len, pas),
                },
                'passage': {
                    'passage_text': pas,
                    'passage_toks': pas_toks,
                    'passage_context_features':
                        get_context_features(query, pas_toks, setting.passage_len, passage_list),
                    'passage_char_features':
                        get_char_features(pas_toks, setting.passage_len, setting.word_len, query),
                },
                'answer': answer
            }
            teared_dataset.append(item)
    return teared_dataset


def get_context_features(ref_sequence, tokens, max_len, passage_list=None):
    ret = np.zeros((max_len, (4 if passage_list else 2)))

    for i, t in enumerate(tokens):
        ret[i, 0] = 1 if t in ref_sequence else 0
        valid = sum(1 for c in t if c in ref_sequence)
        ret[i, 1] = valid / len(t)

    if passage_list:
        # passage_text = ''.join(tokens)
        # temp = compute_jaccard(ref_sequence, passage_text)
        # idx = 0
        # for i in range(len(tokens)):
        #     idx_new = idx + len(tokens[i])
        #     fz = 0.0
        #     fm = len(tokens[i])
        #     for j in range(idx, idx_new):
        #         fz += temp[j]
        #     ret[1][i] = fz/fm
        #     idx = idx_new
        #
        # temp = compute_jaccard(jieba.lcut(ref_sequence), tokens)
        # for i in range(min(len(temp), max_len)):
        #     ret[2][i] = temp[i]
        #
        # temp = compute_editdistance(ref_sequence, tokens)
        # for i in range(min(len(temp), max_len)):
        #     ret[3][i] = temp[i]

        for i, t in enumerate(tokens):
            for passage in passage_list:
                if t in passage:
                    ret[i, 2] += 1.0 / len(passage_list)
            valid = 0.0
            total = 0.0
            for c in t:
                for passage in passage_list:
                    if c in passage:
                        valid += 1.0
                    total += 1.0
            ret[i, 3] = valid / total
    return ret


def compute_jaccard(qWordList, tokens):
    l = -int(len(qWordList)/2)
    r = int(len(qWordList)/2)

    count = 0.0
    for i in range(l, r):
        if i >= 0 and i < len(tokens) and (tokens[i] in qWordList):
            count += 1.0
    ret = np.zeros(len(tokens))
    for i, token in enumerate(tokens):
        ret[i] = count/len(qWordList)
        if l >= 0 and l < len(tokens) and (tokens[l] in qWordList):
            count -= 1.0
        if r+1 >= 0 and r+1 < len(tokens) and (tokens[r+1] in qWordList):
            count += 1.0
        l += 1
        r += 1
    return ret


def compute_editdistance(question,tokens):
    context = ''.join(tokens)
    ret = []
    i = 0
    for token in tokens:
        j = i+(len(token)/2)
        L = int(j-len(question)/2)
        R = int(j+len(question)/2)
        ret.append(1.0*editdistance.eval(context[max(0, L):min(R, len(context))], question)/len(question))
        i += len(token)
    return ret


def get_char_features(tokens, max_seq_len, max_word_len, sentence):
    ret = np.zeros((max_seq_len, max_word_len, 1))
    for i, t in enumerate(tokens):
        t = t[:max_word_len]
        for j, c in enumerate(t):
            ret[i, j, 0] = 1 if c in sentence else 0
    return ret


def clean_dataset(dataset):
    for data in dataset:
        cleaned_p = clean_passages(data['passages'])
        data['passages'] = cleaned_p
        cleaned_q = clean_text(data['query'])
        cleaned_q_toks = get_toks(cleaned_q)[:setting.query_len]
        cleaned_q = ''.join(cleaned_q_toks)
        data['query'] = cleaned_q
        data['query_toks'] = cleaned_q_toks

        for i in cleaned_p:
            get_w2id(i['passage_toks'])
            get_c2id(i['passage_text'])
        get_w2id(cleaned_q_toks, isquery=True)
        get_c2id(cleaned_q, isquery=True)
    return dataset


if __name__ == '__main__':
    training_set = utils.json_file_to_dict(TRAINING_FILE)
    validation_set = utils.json_file_to_dict(VALIDATE_FILE)

    training_set = clean_dataset(training_set)
    validation_set = clean_dataset(validation_set)

    teared_training_set = get_teared_dataset(training_set)
    teared_validation_set = get_teared_dataset(validation_set)

    print('training set size:　', len(teared_training_set))
    print('validation set size: ', len(teared_validation_set))

    utils.pickleWriter(os.path.join(CLEAN_DIR, 'teared_train_factoid_plen%i.pkl'%setting.passage_len), teared_training_set)
    utils.pickleWriter(os.path.join(CLEAN_DIR, 'teared_validation_factoid_plen%i.pkl'%setting.passage_len), teared_validation_set)

    id2c = ['<PAD>', '<UNK>'] + utils.toSinList(char_dic, 5)
    c2id = {v: k for k, v in enumerate(id2c)}
    utils.pickleWriter(os.path.join(HELPER_DIR, 'id2c_plen%i.pkl'%setting.passage_len), id2c)
    utils.pickleWriter(os.path.join(HELPER_DIR, 'c2id_plen%i.pkl'%setting.passage_len), c2id)
    print('char size: ', len(id2c))

    id2w = ['<PAD>', '<UNK>'] + utils.toSinList(word_dic, 5)
    w2id = {v: k for k, v in enumerate(id2w)}
    utils.pickleWriter(os.path.join(HELPER_DIR, 'id2w_plen%i.pkl'%setting.passage_len), id2w)
    utils.pickleWriter(os.path.join(HELPER_DIR, 'w2id_plen%i.pkl'%setting.passage_len), w2id)
    print('vocab size: ', len(id2w))

    utils.pickleWriter(os.path.join(HELPER_DIR, 'size_log_plen%i.pkl'%setting.passage_len), [len(id2w), len(id2c)])

    if setting.w_emb_size == 300:
        get_id2vec_dict()
        print('id2vec size: ', len(id2vec))
        save_id2vec_data(id2vec)
