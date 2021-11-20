from .base_model import BaseModel
import tensorflow as tf

class BackgroundModel(BaseModel):
    def __init__(self, config):
        super(BackgroundModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        # network params
        n2d_layers   = self.config.bgmodel_n2d_layers
        n2d_filters  = self.config.bgmodel_n2d_filters
        window2d     = self.config.bgmodel_window2d

        # for short
        activation = tf.nn.elu
        conv1d = tf.layers.conv1d
        conv2d = tf.layers.conv2d
        instance_norm = tf.contrib.layers.instance_norm
        dropout = tf.keras.layers.Dropout
        softmax = tf.nn.softmax
        
        with tf.name_scope('input'):
            self.ncol =  tf.compat.v1.placeholder(dtype=tf.int32, shape=())

        # 2D network starts
        layers2d = [tf.random.normal([5,self.ncol,self.ncol,n2d_filters])]
        # layers2d = [tf.zeros([5,self.ncol, self.ncol,n2d_filters], tf.float32)]  # xyj test

        layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
        layers2d.append(instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1]))

        # dilated resnet
        dilation = 1
        for _ in range(n2d_layers):
            layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
            layers2d.append(instance_norm(layers2d[-1]))
            layers2d.append(activation(layers2d[-1]))
            layers2d.append(dropout(rate=0.15)(layers2d[-1], training=False))
            layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
            layers2d.append(instance_norm(layers2d[-1]))
            layers2d.append(activation(layers2d[-1] + layers2d[-7]))
            dilation *= 2
            if dilation > 16:
                dilation = 1

        with tf.name_scope("loss"):
            # loss on theta
            logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')
            self.prob_theta = softmax(logits_theta)

            # loss on phi
            logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')
            self.prob_phi = softmax(logits_phi)

            # symmetrize
            layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

            # loss on distances
            logits_dist = conv2d(layers2d[-1], 37, 1, padding='SAME')
            self.prob_dist = softmax(logits_dist)

            # loss on beta-strand pairings
            logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')
            self.prob_bb = softmax(logits_bb)

            # loss on omega
            logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')
            self.prob_omega = softmax(logits_omega)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # self.saver = tf.compat.v1.train.Saver()

