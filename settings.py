class Settings():
    w_emb_size = 300
    c_emb_size = 50
    query_len = 10
    passage_len = 50
    char_feature_len = 10

    nchars = 5416
    nwords = 62684
    xps = [[query_len, 10], [query_len], [passage_len, 10], [passage_len]]

    conv_filter_num = 64
    conv_kernel_size = 7
    dropout_rate = 0.5

    batch_size = 1024
    epochs = 60
