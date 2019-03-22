import os, h5py
import numpy as np
from settings import Settings
from weight_norm import AdamWithWeightnorm
import utils
setting = Settings()

HELPER_DIR = os.path.join('data', 'helper')
DATASET_DIR = os.path.join('data', 'dataset')
MODEL_DIR = os.path.join('models')
RESULT_DIR = os.path.join('results')
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)

logger = setting.get_logger()
xps = setting.xps
with h5py.File(os.path.join(DATASET_DIR, 'dataset_plen%i.h5'%setting.passage_len), 'r') as dfile:
    YY = dfile['YY'][:]
    tY = dfile['tY'][:]
    XXs, tXs = [0] * len(xps), [0] * len(xps)
    for i in range(len(xps)):
        XXs[i] = dfile['XXs%d' % i][:]
        tXs[i] = dfile['tXs%d' % i][:]

if setting.w_emb_size == 300:
    w2v = np.load(os.path.join(HELPER_DIR, 'vec.npy'))
else:
    w2v = np.random.randn(setting.nwords, setting.w_emb_size)


import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.not_equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + 1e-8)) - \
               K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-8))
    return focal_loss_fixed


def training(name, model):
    model_dir = os.path.join(MODEL_DIR, name + '.h5')
    print(model_dir)
    try:
        model.load_weights(model_dir)
    except:
        print('\n\nnew model\n')

    logger.info('%s model, focal loss, AdamWithWeightnorm, plen=%i' % (name, setting.passage_len))
    model.compile(AdamWithWeightnorm(0.001, clipvalue=1.0), focal_loss())

    best_loss = 5000
    best_epoch = 0
    for epoch in range(setting.epochs):
        if epoch - best_epoch > 20:
            print('-' * 10, ' failed to update model for 20 epochs, end at epoch %d ' % epoch, '-' * 10)
            print('best loss: ', best_loss)
            logger.info('best epoch: %d, best loss: %f' % (best_epoch, best_loss))
            print('-' * 20)
            break
        hist = model.fit(XXs, [YY[0], YY[1]],
                         batch_size=setting.batch_size,
                         epochs=1,
                         callbacks=[ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True)])
        loss = model.evaluate(tXs, [tY[0], tY[1]], batch_size=setting.batch_size)

        if loss[0] < best_loss:
            print('-' * 20)
            print('epoch:%d update model ' % epoch)
            print(loss)
            print('-' * 20)
            best_loss = loss[0]
            best_epoch = epoch
            model.save_weights(model_dir)

        if epoch % 4 == 3:
            newlr = K.get_value(model.optimizer.lr) * 0.7
            print('newlr: %.7f' % newlr)
            K.set_value(model.optimizer.lr, newlr)

    model.load_weights(model_dir)
    y_pred_start, y_pred_end = model.predict(tXs, batch_size=setting.batch_size)

    utils.pickleWriter(os.path.join(RESULT_DIR, 'y_pred_start_%s.pkl' % name), y_pred_start)
    utils.pickleWriter(os.path.join(RESULT_DIR, 'y_pred_end_%s.pkl' % name), y_pred_end)
