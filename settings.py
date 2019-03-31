import logging, os
import utils

HELPER_DIR = os.path.join('data', 'helper')
LOG_DIR = os.path.join('data', 'log')


class Settings():
    def __init__(self):
        self.w_emb_size = 50
        self.c_emb_size = 50
        self.query_len = 30
        self.passage_len = 50
        self.word_len = 8
        self.char_feature_len = 10

        size_file = os.path.join(HELPER_DIR, 'size_log_plen%i.pkl'%self.passage_len)
        if os.path.exists(size_file):
            self.nwords, self.nchars = utils.pickleLoader(size_file)

        self.xps = [
            [self.query_len, self.word_len],     # query_char_level raw text
            [self.query_len],                    # query_word_level raw text
            [self.passage_len, self.word_len],   # passage_char_level raw text
            [self.passage_len],                  # passage_word_level raw text
            [self.query_len, self.word_len, 1],     # query_char_level feature
            [self.query_len, 2],                 # query_word_level feature
            # [self.query_len],                    # query_word_level feature
            [self.passage_len, self.word_len, 1],   # passage_char_level feature
            [self.passage_len, 4],               # passage_word_level feature
            # [self.passage_len],                  # passage_word_level feature1 query-passage similarity based on co-occurrence
            # [self.passage_len],                  # passage_word_level feature2 jaccard1
            # [self.passage_len],                  # passage_word_level feature3 jaccard2
            # [self.passage_len],                  # passage_word_level feature4 editdistance
            # [self.passage_len],                  # passage_word_level feature5 passage-passage similarity based on word co-occurrence
        ]

        self.conv_filter_num = 64
        self.conv_kernel_size = 7
        self.dropout_rate = 0.5

        self.batch_size = 1024
        self.epochs = 60

    @staticmethod
    def get_logger(level=logging.INFO):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        logger = logging.getLogger()
        logger.setLevel(level)
        fh = logging.FileHandler(os.path.join(LOG_DIR, 'model_trial.log'), mode='a')
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
