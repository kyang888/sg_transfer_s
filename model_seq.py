import os, h5py, logging
import numpy as np
from settings import Settings
import utils
setting = Settings()


HELPER_DIR = os.path.join('data', 'helper')
DATASET_DIR = os.path.join('data', 'dataset')
MODEL_DIR = os.path.join('models', 'trial')
LOG_DIR = os.path.join('data', 'log')

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, 'model_trial.log'), mode='a')
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

xps = setting.xps
with h5py.File(os.path.join(DATASET_DIR, 'dataset_plen50.h5'), 'r') as dfile:
    YY = dfile['YY'][:]
    tY = dfile['tY'][:]
    XXs, tXs = [0] * len(xps), [0] * len(xps)
    for i in range(len(xps)):
        XXs[i] = dfile['XXs%d' % i][:]
        tXs[i] = dfile['tXs%d' % i][:]

w2v = np.load(os.path.join(HELPER_DIR, 'vec.npy'))

model_dir = os.path.join(MODEL_DIR, 'model1.h5')


MODE = "train"
name = 'basic'

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.not_equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + 1e-8)) - \
               K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-8))
    return focal_loss_fixed


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint

quest_input = Input(shape=(setting.query_len,), dtype='int32')
cand_input = Input(shape=(setting.passage_len,), dtype='int32')

questC_input = Input(shape=(setting.query_len, 10), dtype='int32')
candC_input = Input(shape=(setting.passage_len, 10), dtype='int32')

embed_layer = Embedding(setting.nwords, setting.w_emb_size, weights=[w2v], trainable=True)
quest_emb = SpatialDropout1D(0.5)(embed_layer(quest_input))
cand_emb = SpatialDropout1D(0.5)(embed_layer(cand_input))

cembed_layer = Embedding(setting.nchars + 2, setting.c_emb_size)

char_input = Input(shape=(10,), dtype='int32')
c_emb = SpatialDropout1D(0.5)(cembed_layer(char_input))
cc = Conv1D(setting.c_emb_size, 3, padding='same')(c_emb)
cc = LeakyReLU()(cc)
cc = Lambda(lambda x: K.mean(x, 1), output_shape=lambda d: (d[0], d[2]))(cc)
# cc = Flatten()(cc)
char_model = Model(char_input, cc)

qc_emb = TimeDistributed(char_model)(questC_input)
cc_emb = TimeDistributed(char_model)(candC_input)

quest_emb = concatenate([quest_emb, qc_emb])
cand_emb = concatenate([cand_emb, cc_emb])

quest_vec = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(quest_emb)
quest_vec = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(quest_vec)

cand_vec = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(cand_emb)
cand_vec = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(cand_vec)
q_input = quest_vec
c_input = cand_vec

match = dot([c_input, q_input], axes=[2, 2])
match = Activation('softmax')(match)  # (samples, story_maxlen, query_maxlen)

response = concatenate([match, c_input])

merge_vec = Bidirectional(GRU(64, return_sequences=True, dropout=0.5))(response)

final_start = Flatten(name='s')(TimeDistributed(Dense(1, activation='sigmoid'))(merge_vec))
final_end = Flatten(name='e')(TimeDistributed(Dense(1, activation='sigmoid'))(merge_vec))


model = Model(inputs=[questC_input, quest_input, candC_input, cand_input], outputs=[final_start, final_end])
model.summary()


if __name__ == '__main__':
    logger.info('basic model, focal loss, using pretrained w2v, plen=50')
    model.compile(Adam(), focal_loss())

    best_model = None
    best_loss = 500
    best_epoch = 0
    for epoch in range(setting.epochs):
        if epoch - best_epoch > 5:
            print('-'*10, ' failed to update model for 5 epochs, end at epoch %d ' % epoch, '-'*10)
            print('best loss: ', best_loss)
            logger.info('best epoch: %d, best loss: %f'%(best_epoch, best_loss))
            print('-'*20)
            break
        hist = model.fit(XXs, [YY[0], YY[1]],
                         batch_size=setting.batch_size,
                         epochs=1,
                         callbacks=[ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True)])
        loss = model.evaluate(tXs, [tY[0], tY[1]], batch_size=setting.batch_size)
        y_pred_start, y_pred_end = model.predict(tXs, batch_size=setting.batch_size)

        if loss[0] < best_loss:
            print('-'*20)
            print('epoch:%d update model ' % epoch)
            print(loss)
            print('-'*20)
            best_loss = loss[0]
            best_epoch = epoch
            model.save(model_dir)


    model.load_weights(model_dir)
    y_pred_start, y_pred_end = model.predict(tXs, batch_size=setting.batch_size)

    utils.pickleWriter('y_pred_start.pkl', y_pred_start)
    utils.pickleWriter('y_pred_end.pkl', y_pred_end)