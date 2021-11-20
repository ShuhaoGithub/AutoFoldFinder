from prokit.models.base_model import BaseModel
from prokit.utils.protein_utils import *
import tensorflow as tf


class trRosettaModelPred(BaseModel):
    def __init__(self, config):
        super(trRosettaModelPred, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # network params
        n2d_layers   = self.config.trRosetta_n2d_layers
        n2d_filters  = self.config.trRosetta_n2d_filters
        window2d     = self.config.trRosetta_window2d
        wmin         = self.config.trRosetta_wmin
        ns           = self.config.trRosetta_ns

        # for short
        activation = tf.nn.elu
        conv2d = tf.layers.conv2d
        instance_norm = tf.contrib.layers.instance_norm
        dropout = tf.keras.layers.Dropout
        softmax = tf.nn.softmax
        
        with tf.name_scope('input'):
            # self.self.ncol =  tf.compat.v1.placeholder(dtype=tf.int32, shape=())
            self.ncol = tf.placeholder(dtype=tf.int32, shape=())
            self.nrow = tf.placeholder(dtype=tf.int32, shape=())
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
            self.is_train = tf.placeholder(tf.bool, name='is_train')

        #
        # collect features
        #
        msa1hot  = tf.one_hot(self.msa, ns, dtype=tf.float32)
        w = reweight(msa1hot, wmin)

        # 1D features
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, w)

        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,self.ncol,42])

        # 2D features
        f2d_dca = tf.cond(self.nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([self.ncol,self.ncol,442], tf.float32))
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)

        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,self.ncol,1]), 
                        tf.tile(f1d[:,None,:,:], [1,self.ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,self.ncol,self.ncol,442+2*42])


        #
        # 2D network
        #
        layers2d = [f2d]
        layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
        layers2d.append(instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1]))

        # stack of residual blocks with dilations
        dilation = 1
        for _ in range(n2d_layers):
            layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
            layers2d.append(instance_norm(layers2d[-1]))
            layers2d.append(activation(layers2d[-1]))
            layers2d.append(dropout(rate=0.15)(layers2d[-1], training=self.is_train))
            layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
            layers2d.append(instance_norm(layers2d[-1]))
            layers2d.append(activation(layers2d[-1] + layers2d[-7]))
            dilation *= 2
            if dilation > 16:
                dilation = 1

        with tf.name_scope("loss"):
            # anglegrams for theta
            logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')
            self.prob_theta = softmax(logits_theta)

            # anglegrams for phi
            logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')
            self.prob_phi = softmax(logits_phi)

            # symmetrize
            layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

            # distograms
            logits_dist = conv2d(layers2d[-1], 37, 1, padding='SAME')
            self.prob_dist = softmax(logits_dist)

            # beta-strand pairings (not used)
            logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')
            self.prob_bb = softmax(logits_bb)

            # anglegrams for omega
            logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')
            self.prob_omega = softmax(logits_omega)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.compat.v1.train.Saver()

