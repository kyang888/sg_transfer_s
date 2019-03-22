from training import training, setting, w2v
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
from keras.layers import *
from keras.optimizers import *
from keras.models import *

quest_input = Input(shape=(setting.query_len,), dtype='int32', name='quest_word_input')
cand_input = Input(shape=(setting.passage_len,), dtype='int32', name='passage_word_input')

questC_input = Input(shape=(setting.query_len, setting.word_len), dtype='int32', name='quest_char_input')
candC_input = Input(shape=(setting.passage_len, setting.word_len), dtype='int32', name='passage_char_input')

questA_input = Input(shape=(setting.query_len, 2), name='quest_word_feature_input')
candA_input = Input(shape=(setting.passage_len, 4), name='passage_word_feature_input')

questAc_input = Input(shape=(setting.query_len, setting.word_len, 1), name='quest_char_feature_input')
candAc_input = Input(shape=(setting.passage_len, setting.word_len, 1), name='passage_char_feature_input')


def MaskWord(x, dropout=0.2):
    # Mask to 1
    mask = K.cast(K.random_uniform(K.shape(x)) < dropout, 'float32')
    mask = K.cast(K.not_equal(x, 0), 'float32') * mask
    xm = K.cast(x, 'float32') * (1 - mask) + mask
    return K.in_train_phase(K.cast(xm, 'int32'), K.cast(x, 'int32'))


MaskWordLayer = Lambda(MaskWord)

qi = MaskWordLayer(quest_input)
ci = MaskWordLayer(cand_input)
qci = MaskWordLayer(questC_input)
cci = MaskWordLayer(candC_input)


embed_layer = Embedding(setting.nwords, setting.w_emb_size, weights=[w2v], trainable=True)
quest_emb = Dropout(setting.dropout_rate, (None, 1, None))( embed_layer(quest_input) )
cand_emb  = Dropout(setting.dropout_rate, (None, 1, None))( embed_layer(cand_input) )

cembed_layer = Embedding(setting.nchars + 2, setting.c_emb_size)


char_input = Input(shape=(setting.word_len,), dtype='int32')
c_emb = Dropout(setting.dropout_rate, (None, 1, None))( cembed_layer(char_input) )
cmask = Lambda(lambda x:K.cast(K.not_equal(x, 0), 'float32'))(char_input)
cc = Conv1D(setting.c_emb_size, 3, padding='same')(c_emb)
cc = LeakyReLU()(cc)
cc = multiply([cc, Reshape((-1,1))(cmask)])
cc = Lambda(lambda x:K.sum(x, 1))(cc)
char_model = Model(char_input, cc)

qc_emb = TimeDistributed(char_model)(qci)
cc_emb = TimeDistributed(char_model)(cci)

quest_emb = concatenate([quest_emb, qc_emb, questA_input])
cand_emb  = concatenate([cand_emb, cc_emb, candA_input])

quest_emb = Dense(128, activation='relu')(quest_emb)
cand_emb  = Dense(128, activation='relu')(cand_emb)


class Highway:
    def __init__(self, dim, layers=2):
        self.linears = [Dense(dim, activation='relu') for _ in range(layers+1)]
        self.gates = [Dense(dim, activation='sigmoid') for _ in range(layers)]
    def __call__(self, x):
        for linear, gate in zip(self.linears, self.gates):
            g = gate(x)
            z = linear(x)
            x = Lambda(lambda x:x[0]*x[1]+(1-x[0])*x[2])([g, z, x])
        return x

highway1 = Highway(dim=128, layers=2)
highway2 = Highway(dim=128, layers=2)


quest_emb = highway1(quest_emb)
cand_emb  = highway1(cand_emb)

lstm_dim = 60

def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

add_layer = Lambda(lambda x:x[0]+x[1], output_shape=lambda x:x[0])


class AddPosEncoding:
    def __call__(self, x):
        _, max_len, d_emb = K.int_shape(x)
        pos = GetPosEncodingMatrix(max_len, d_emb)
        x = Lambda(lambda x: x + pos)(x)
        return x


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads)
            attn = Concatenate()(attns)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class FeedForward():
    def __init__(self, d_hid, dropout=0.1):
        self.forward = Dense(d_hid, activation='relu')
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        x = self.forward(x)
        x = self.dropout(x)
        return x


class LayerDropout:
    def __init__(self, dropout=0.2):
        self.dropout = dropout

    def __call__(self, old, new):
        def func(args):
            old, new = args
            pred = K.random_uniform([]) < self.dropout
            ret = K.switch(pred, old, old + K.dropout(new, self.dropout))
            return K.in_train_phase(ret, old + new)

        return Lambda(func)([old, new])


class ConvBlock:
    def __init__(self, dim, n_conv=2, kernel_size=7, dropout=0.1):
        self.convs = [SeparableConv1D(dim, kernel_size, activation='relu', padding='same') for _ in range(n_conv)]
        self.norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        for i in range(len(self.convs)):
            z = self.norm(x)
            if i % 2 == 0: z = self.dropout(z)
            z = self.convs[i](z)
            x = add_layer([x, z])
        return x


class EncoderBlock:
    def __init__(self, dim, n_head, n_conv, kernel_size):
        self.conv = ConvBlock(dim, n_conv=n_conv, kernel_size=kernel_size)
        self.self_att = MultiHeadAttention(n_head=n_head, d_model=dim,
                                           d_k=dim // n_head, d_v=dim // n_head,
                                           dropout=0.1, use_norm=False)
        self.feed_forward = PositionwiseFeedForward(dim, dim, dropout=0.1)
        self.norm = LayerNormalization()

    def __call__(self, x, mask):
        x = AddPosEncoding()(x)
        x = self.conv(x)
        z = self.norm(x)
        z, _ = self.self_att(z, z, z, mask)
        x = add_layer([x, z])
        z = self.norm(x)
        z = self.feed_forward(z)
        x = add_layer([x, z])
        return x


class Encoder:
    def __init__(self, dim=128, n_head=8, n_conv=2, n_block=1, kernel_size=7):
        self.dim = dim
        self.n_block = n_block
        self.conv_first = SeparableConv1D(dim, 1, padding='same')
        self.enc_block = EncoderBlock(dim, n_head=n_head, n_conv=n_conv, kernel_size=kernel_size)

    def __call__(self, x, mask):
        if K.int_shape(x)[-1] != self.dim:
            x = self.conv_first(x)
        for i in range(self.n_block):
            x = self.enc_block(x, mask)
        return x

emb_enc1  = Encoder(128, n_head=2, n_conv=4, n_block=1, kernel_size=7)
emb_enc2  = Encoder(128, n_head=2, n_conv=4, n_block=1, kernel_size=7)
main_enc1 = Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)
main_enc2 = Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)
main_enc3 = Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)

Qmask = Lambda(lambda x:K.cast(K.not_equal(x,0), 'float32'))(quest_input)
Cmask = Lambda(lambda x:K.cast(K.not_equal(x,0), 'float32'))(cand_input)

Q = emb_enc1(quest_emb, Qmask)
C = emb_enc1(cand_emb,  Cmask)

Qsmask = Lambda(lambda x:(1-x)*(-1e+9))(Qmask)
Csmask = Lambda(lambda x:(1-x)*(-1e+9))(Cmask)

CC = Lambda(lambda C:K.repeat_elements(C[:,:,None,:], setting.query_len, 2))(C)
QQ = Lambda(lambda Q:K.repeat_elements(Q[:,None,:,:], setting.passage_len, 1))(Q)
S_hat = concatenate([CC, QQ, multiply([CC, QQ])])
S = Reshape((setting.passage_len, setting.query_len))( TimeDistributed(TimeDistributed(Dense(1, use_bias=False)))(S_hat) )

aa = Activation('softmax')( add([S, Qsmask]) )
U_hat = Lambda(lambda x:K.batch_dot(x[0], x[1]))([aa, Q])

SS = Lambda(lambda x: K.max(x, 2))(S)
bb = Activation('softmax')( add([SS, Csmask]) )
h_hat = Lambda( lambda x: K.batch_dot(K.expand_dims(x[0],1), x[1]) )([bb, C])
H_hat = Lambda( lambda x: K.repeat_elements(x, setting.passage_len, 1) )(h_hat)


G = concatenate([C, U_hat, multiply([C,U_hat]), multiply([C,H_hat])])

G0 = main_enc1(G,  Cmask)
M1 = main_enc2(G0, Cmask)
M2 = main_enc3(M1, Cmask)

GM1 = concatenate([G0, M1])
GM2 = concatenate([G0, M2])

F1 = TimeDistributed(Dense(1, use_bias=False))(GM1)
F2 = TimeDistributed(Dense(1, use_bias=False))(GM2)

final_start = Activation('sigmoid', name='s')(Flatten()( F1 ))
final_end   = Activation('sigmoid', name='e')(Flatten()( F2 ))

model = Model(inputs=[questC_input, quest_input, candC_input, cand_input, questAc_input, questA_input, candAc_input, candA_input], outputs=[final_start, final_end])
model.summary()


if __name__ == '__main__':
    name = 'qanet'
    training(name, model)
