# -*-coding: utf-8 -*-
import jieba, os, re, random
import numpy as np
import utils
from settings import Settings

setting = Settings()

training_set_dir = os.path.join('data', 'train_factoid_1.json')
raw_pretrained_w2vec_dir = os.path.join('data', 'sgns.zhihu.bigram-char')
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
        if len(passage_text) > 2 * setting.passage_len:
            p1 = passage_text[:setting.passage_len]
            p2 = passage_text[setting.passage_len: 2* setting.passage_len]
            t1 = get_toks(p1)
            t2 = get_toks(p2)
            passage['passage_text'] = p1
            passage['passage_toks'] = t1
            cleaned_passages.append(passage)
            passage['passage_text'] = p2
            passage['passage_toks'] = t2
            cleaned_passages.append(passage)
        else:
            passage['passage_text'] = passage_text[:setting.passage_len]
            passage['passage_toks'] = get_toks(passage_text[:setting.passage_len])
            cleaned_passages.append(passage)
    return cleaned_passages


def clean_text(text):
    text = text.lower()
    text = text.replace('\u3000', ':')
    text = re.sub(r'\s+', ' ', text)
    text = mark_transfer(text)
    text = DBC2SBC(text)
    text = text.replace('::', ':').replace('...', '').replace('.:', '.')
    return text


def mark_transfer(text):
    marks = [['，', ','], ['。', '.'], ['“', '"'], ['”', '"'], ['：', ':'], ['？', '?'], ['！', '!'],
             ['%', '%'], ['‘', '\''], ['’', '\''], ['．', ''], ['、', ','], ['【', '['], ['】', ']'],
             ['（', '('], ['）', ')'], ['＂', '"'], ['＞', '>'],['…', ''], ['；', ';']]
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


def get_w2id(tokens):
    global word_dic
    for i in tokens:
        word_dic[i] = word_dic.get(i, 0) + 1


def get_c2id(text):
    global char_dic
    for i in text:
        char_dic[i] = char_dic.get(i, 0) + 1


def get_id2vec_dict():
    global id2vec
    print('loading pretrained word embedding ...')
    with open(raw_pretrained_w2vec_dir, 'r', encoding='utf-8') as f:
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
    np.save(os.path.join(HELPER_DIR, 'vec.npy'), vec)


def get_match_idxs(ori_str, tar_str):
    res = []
    reg = re.compile(r"" + tar_str)
    it = reg.finditer(ori_str)
    for i in it:
        data = {'s_idx': i.start(), 'e_idx': i.end() - 1}
        res.append(data)
    return res


def get_teared_training_set(training_set):
    teared_training_set = []
    random.shuffle(training_set)
    for training_data in training_set:
        passages = training_data['passages']
        answer = training_data['answer']
        query = training_data['query']
        query_id = training_data['query_id']
        query_toks = training_data['query_toks']
        for passage in passages:
            pas = passage['passage_text']
            ans = get_match_idxs(pas, answer)
            data = {
                'query': {
                    'query_id': query_id,
                    'query_text': query,
                    'query_toks': query_toks,
                },
                'passage': {
                    'passage_text': pas,
                    'passage_toks': passage['passage_toks'],
                    'url': passage['url']
                },
                'answer': answer,
                'answer_idx': ans
            }
            teared_training_set.append(data)
    return teared_training_set


if __name__ == '__main__':
    training_set = utils.json_file_to_dict(training_set_dir)
    for training_data in training_set:
        cleaned_p = clean_passages(training_data['passages'])
        training_data['passages'] = cleaned_p
        cleaned_q = clean_text(training_data['query'])
        cleaned_q_toks = get_toks(cleaned_q)
        training_data['query'] = cleaned_q
        training_data['query_toks'] = cleaned_q_toks
        for i in cleaned_p:
            get_w2id(i['passage_toks'])
            get_c2id(i['passage_text'])
        get_w2id(cleaned_q_toks)
        get_c2id(cleaned_q)

    teared_training_set = get_teared_training_set(training_set)
    print('training set size:　', len(teared_training_set))
    utils.pickleWriter(os.path.join(CLEAN_DIR, 'teared_train_factoid_plen50.pkl'), teared_training_set)

    id2c = ['<PAD>', '<UNK>'] + utils.toSinList(char_dic, 2)
    print('char dic: ', id2c)
    c2id = {v: k for k, v in enumerate(id2c)}
    utils.pickleWriter(os.path.join(HELPER_DIR, 'id2c.pkl'), id2c)
    utils.pickleWriter(os.path.join(HELPER_DIR, 'c2id.pkl'), c2id)
    print('char size: ', len(id2c))

    id2w = ['<PAD>', '<UNK>'] + utils.toSinList(word_dic, 2)
    print('vocab dic: ', id2w)
    w2id = {v: k for k, v in enumerate(id2w)}
    utils.pickleWriter(os.path.join(HELPER_DIR, 'id2w.pkl'), id2w)
    utils.pickleWriter(os.path.join(HELPER_DIR, 'w2id.pkl'), w2id)
    print('vocab size: ', len(id2w))

    get_id2vec_dict()
    print('id2vec size: ', len(id2vec))
    save_id2vec_data(id2vec)
