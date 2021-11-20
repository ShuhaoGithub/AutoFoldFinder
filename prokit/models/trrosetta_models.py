from prokit.models.base_model import BaseModel
from prokit.utils.protein_utils import *
import tensorflow as tf
import os
import logging


class TrRosettaModelKL(BaseModel):
    def __init__(self, config):
        super(TrRosettaModelKL, self).__init__(config)
        self.build_model_zsh()

    def build_model_v1(self):
        # T0,N,coef,M = self.config.schedule
        # beta = 1./T0
        # nsave = self.config.num_save_step

        n2d_layers   = self.config.trRosetta_n2d_layers
        aa_weight = self.config.trRosetta_aa_weight

        w, b, beta_, gamma_ = self.init_weights_v1()

        Activation   = tf.nn.elu
        
        # inputs
        with tf.name_scope('input'):
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
  
        ncol = tf.shape(self.msa)[1]
            
        # background distributions
        bkg = self.config.background_dist_dict
        bd = tf.constant(bkg['dist'], dtype=tf.float32)
        bo = tf.constant(bkg['omega'], dtype=tf.float32)
        bt = tf.constant(bkg['theta'], dtype=tf.float32)
        bp = tf.constant(bkg['phi'], dtype=tf.float32)
        
        # aa bkgr composition in natives
        aa_bkgr = tf.constant(getBackgroundAAComposition(), dtype = tf.float32)

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(self.msa, 21, dtype=tf.float32)

        # collect features
        weight = reweight(msa1hot, 0.8)
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weight)
        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,ncol,42])
        f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)
        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(4)]

        for i in range(len(w)):

            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(Activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1


            # probabilities for theta and phi
            preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)
        #pb = tf.reduce_mean(tf.stack(preds[4]),axis=0)

        with tf.name_scope("loss"):
            self.loss_dist = -tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd/bd),axis=-1))
            self.loss_omega = -tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po/bo),axis=-1))
            self.loss_theta = -tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt/bt),axis=-1))
            self.loss_phi = -tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp/bp),axis=-1))

            # aa composition loss
            aa_samp = tf.reduce_sum(msa1hot[0,:,:20], axis=0)/tf.cast(ncol,dtype=tf.float32)+1e-7
            aa_samp = aa_samp/tf.reduce_sum(aa_samp)
            self.loss_aa = tf.reduce_sum(aa_samp*tf.log(aa_samp/aa_bkgr))

            # total loss
            self.loss = self.loss_dist + self.loss_omega + self.loss_theta + self.loss_phi + aa_weight*self.loss_aa


    def init_weights_v1(self):
        # load networks in RAM
        w,b = [],[]
        beta_,gamma_ = [],[]

        DIR = self.config.trRosetta_model_dir

        for filename in os.listdir(DIR):
            if not filename.endswith(".index"):
                continue
            mname = DIR+"/"+os.path.splitext(filename)[0]
            w.append([
                tf.train.load_variable(mname, 'conv2d/kernel')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
                for i in range(128)])

            b.append([
                tf.train.load_variable(mname, 'conv2d/bias')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
                for i in range(128)])

            beta_.append([
                tf.train.load_variable(mname, 'InstanceNorm/beta')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
                for i in range(123)])

            gamma_.append([
                tf.train.load_variable(mname, 'InstanceNorm/gamma')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
                for i in range(123)])

            print('Success to load weights from:', mname)
        return w, b, beta_, gamma_

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.compat.v1.train.Saver()
        
    
    def build_model_zsh(self):

        n2d_layers   = self.config.trRosetta_n2d_layers
        aa_weight = self.config.trRosetta_aa_weight

        w, b, beta_, gamma_ = self.init_weights_v1()

        Activation   = tf.nn.elu
        
        # inputs
        with tf.name_scope('input'):
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
  
        ncol = tf.shape(self.msa)[1]
            
        # background distributions
        bkg = self.config.background_dist_dict
        bd = tf.constant(bkg['dist'], dtype=tf.float32)
        bo = tf.constant(bkg['omega'], dtype=tf.float32)
        bt = tf.constant(bkg['theta'], dtype=tf.float32)
        bp = tf.constant(bkg['phi'], dtype=tf.float32)
        
        # ref distributions
        ref = []
        refd = []
        refo = []
        reft = []
        refp = []
        for npz in self.config.npz_list:
            new_ref = np.load(npz)
            ref.append(new_ref)
            new_refd = tf.constant(new_ref['dist'], dtype=tf.float32)
            refd.append(new_refd)
            new_refo = tf.constant(new_ref['omega'], dtype=tf.float32)
            refo.append(new_refo)
            new_reft = tf.constant(new_ref['theta'], dtype=tf.float32)
            reft.append(new_reft)
            new_refp = tf.constant(new_ref['phi'], dtype=tf.float32)
            refp.append(new_refp)
        
        # aa bkgr composition in natives
        aa_bkgr = tf.constant(getBackgroundAAComposition(), dtype = tf.float32)

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(self.msa, 21, dtype=tf.float32)

        # collect features
        weight = reweight(msa1hot, 0.8)
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weight)
        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,ncol,42])
        f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)
        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(4)]

        for i in range(len(w)):

            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(Activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1


            # probabilities for theta and phi
            preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)
        #pb = tf.reduce_mean(tf.stack(preds[4]),axis=0)

        with tf.name_scope("loss"):
            self.loss_dist_bkg = self.formula(pd,bd)
            self.loss_omega_bkg = self.formula(po,bo)
            self.loss_theta_bkg = self.formula(pt,bt)
            self.loss_phi_bkg = self.formula(pp,bp)
            
            self.loss_dist = self.loss_dist_bkg
            self.loss_omega = self.loss_omega_bkg
            self.loss_theta = self.loss_theta_bkg
            self.loss_phi = self.loss_phi_bkg
            
    
            total = []
            for i in range(0,len(ref)):
                self.loss_dist += self.config.weight_list[i]*self.formula(pd,refd[i])
                self.loss_omega += self.config.weight_list[i]*self.formula(po,refo[i])
                self.loss_theta += self.config.weight_list[i]*self.formula(pt,reft[i])
                self.loss_phi += self.config.weight_list[i]*self.formula(pp,refp[i])
#                 total.append(self.formula(pd,refd[i]) + self.formula(po,refo[i]) + self.formula(pt,reft[i]) + self.formula(pp,refp[i]))
                

            # aa composition loss
            aa_samp = tf.reduce_sum(msa1hot[0,:,:20], axis=0)/tf.cast(ncol,dtype=tf.float32)+1e-7
            aa_samp = aa_samp/tf.reduce_sum(aa_samp)
            self.loss_aa = tf.reduce_sum(aa_samp*tf.log(aa_samp/aa_bkgr))

            # total loss
            self.loss = (self.loss_dist + self.loss_omega + self.loss_theta + self.loss_phi)/7 + aa_weight*self.loss_aa
#             self.loss = tf.reduce_min(total) + self.loss_dist + self.loss_omega + self.loss_theta + self.loss_phi + aa_weight*self.loss_aa
    
    def formula(self,a,b):
        return -tf.math.reduce_mean(tf.math.reduce_sum(a*tf.math.log(a/b),axis=-1))

class TrRosettaModelKL_CML(BaseModel):
    def __init__(self, config):
        super(TrRosettaModelKL_CML, self).__init__(config)
        self.build_model_v1()

    def build_model_v1(self):
        # T0,N,coef,M = self.config.schedule
        # beta = 1./T0
        # nsave = self.config.num_save_step

        n2d_layers   = self.config.trRosetta_n2d_layers
        aa_weight = self.config.trRosetta_aa_weight

        ref_map_dir = self.config.ref_map_dir
        temp_map_file = self.config.temp_map_file


        w, b, beta_, gamma_ = self.init_weights_v1()

        Activation   = tf.nn.elu
        
        # inputs
        with tf.name_scope('input'):
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
  
        ncol = tf.shape(self.msa)[1]
            
        # background distributions
        bkg = self.config.background_dist_dict
        bd = tf.constant(bkg['dist'], dtype=tf.float32)
        bo = tf.constant(bkg['omega'], dtype=tf.float32)
        bt = tf.constant(bkg['theta'], dtype=tf.float32)
        bp = tf.constant(bkg['phi'], dtype=tf.float32)
        
        # aa bkgr composition in natives
        aa_bkgr = tf.constant(getBackgroundAAComposition(), dtype = tf.float32)

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(self.msa, 21, dtype=tf.float32)

        # collect features
        weight = reweight(msa1hot, 0.8)
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weight)
        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,ncol,42])
        f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)
        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(4)]

        for i in range(len(w)):

            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(Activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1


            # probabilities for theta and phi
            preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)
        #pb = tf.reduce_mean(tf.stack(preds[4]),axis=0)

        with tf.name_scope("loss"):
            self.loss_dist = -tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd/bd),axis=-1))
            self.loss_omega = -tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po/bo),axis=-1))
            self.loss_theta = -tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt/bt),axis=-1))
            self.loss_phi = -tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp/bp),axis=-1))

            # aa composition loss
            aa_samp = tf.reduce_sum(msa1hot[0,:,:20], axis=0)/tf.cast(ncol,dtype=tf.float32)+1e-7
            aa_samp = aa_samp/tf.reduce_sum(aa_samp)
            self.loss_aa = tf.reduce_sum(aa_samp*tf.log(aa_samp/aa_bkgr))
            
            self.dist_bin = pd
            
            # total loss
            self.loss = self.loss_dist + self.loss_omega + self.loss_theta + self.loss_phi + aa_weight*self.loss_aa


    def init_weights_v1(self):
        # load networks in RAM
        w,b = [],[]
        beta_,gamma_ = [],[]

        DIR = self.config.trRosetta_model_dir

        for filename in os.listdir(DIR):
            if not filename.endswith(".index"):
                continue
            mname = DIR+"/"+os.path.splitext(filename)[0]
            w.append([
                tf.train.load_variable(mname, 'conv2d/kernel')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
                for i in range(128)])

            b.append([
                tf.train.load_variable(mname, 'conv2d/bias')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
                for i in range(128)])

            beta_.append([
                tf.train.load_variable(mname, 'InstanceNorm/beta')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
                for i in range(123)])

            gamma_.append([
                tf.train.load_variable(mname, 'InstanceNorm/gamma')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
                for i in range(123)])

            print('Success to load weights from:', mname)
        return w, b, beta_, gamma_

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.compat.v1.train.Saver()

class TrRosettaModelPred(BaseModel):
    def __init__(self, config):
        super(TrRosettaModelPred, self).__init__(config)
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
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # self.saver = tf.compat.v1.train.Saver()


class TrRosettaModelKLINV(BaseModel):
    def __init__(self, config):
        super(TrRosettaModelKLINV, self).__init__(config)
        self.build_model()

    def build_model(self):
        # T0,N,coef,M = self.config.schedule
        # beta = 1./T0
        # nsave = self.config.num_save_step

        n2d_layers   = self.config.trRosetta_n2d_layers
        aa_weight = self.config.trRosetta_aa_weight

        w, b, beta_, gamma_ = self.init_weights()

        Activation   = tf.nn.elu
        
        # inputs
        with tf.name_scope('input'):
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
  
        ncol = tf.shape(self.msa)[1]
            
        # background distributions
        bkg = self.config.groundtruth_dist_dict
        bd = tf.constant(bkg['dist'], dtype=tf.float32)
        bo = tf.constant(bkg['omega'], dtype=tf.float32)
        bt = tf.constant(bkg['theta'], dtype=tf.float32)
        bp = tf.constant(bkg['phi'], dtype=tf.float32)
        
        # aa bkgr composition in natives
        aa_bkgr = tf.constant(getBackgroundAAComposition(), dtype = tf.float32)

        # convert inputs to 1-hot
        msa1hot  = tf.one_hot(self.msa, 21, dtype=tf.float32)

        # collect features
        weight = reweight(msa1hot, 0.8)
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weight)
        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,ncol,42])
        f2d_dca = tf.zeros([ncol,ncol,442], tf.float32)
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)
        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(4)]

        for i in range(len(w)):

            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(Activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1


            # probabilities for theta and phi
            preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        # average over all branches
        pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
        pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
        pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
        po = tf.reduce_mean(tf.stack(preds[3]),axis=0)
        #pb = tf.reduce_mean(tf.stack(preds[4]),axis=0)

        with tf.name_scope("loss"):
            # remove the sign of negative symbols, minimum the loss, close to the groundtruth
            self.loss_dist = tf.math.reduce_mean(tf.math.reduce_sum(pd*tf.math.log(pd/bd),axis=-1))
            self.loss_omega = tf.math.reduce_mean(tf.math.reduce_sum(po*tf.math.log(po/bo),axis=-1))
            self.loss_theta = tf.math.reduce_mean(tf.math.reduce_sum(pt*tf.math.log(pt/bt),axis=-1))
            self.loss_phi = tf.math.reduce_mean(tf.math.reduce_sum(pp*tf.math.log(pp/bp),axis=-1))

            # aa composition loss, minimum this loss, close to the AA composition of background
            aa_samp = tf.reduce_sum(msa1hot[0,:,:20], axis=0)/tf.cast(ncol,dtype=tf.float32)+1e-7
            aa_samp = aa_samp/tf.reduce_sum(aa_samp)
            self.loss_aa = tf.reduce_sum(aa_samp*tf.log(aa_samp/aa_bkgr))
            # total loss
            self.loss = self.loss_dist + self.loss_omega + self.loss_theta + self.loss_phi + aa_weight * self.loss_aa

    def init_weights(self):
        # load networks in RAM
        w,b = [],[]
        beta_,gamma_ = [],[]

        DIR = self.config.trRosetta_model_dir

        for filename in os.listdir(DIR):
            if not filename.endswith(".index"):
                continue
            mname = DIR+"/"+os.path.splitext(filename)[0]
            w.append([
                tf.train.load_variable(mname, 'conv2d/kernel')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
                for i in range(128)])

            b.append([
                tf.train.load_variable(mname, 'conv2d/bias')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
                for i in range(128)])

            beta_.append([
                tf.train.load_variable(mname, 'InstanceNorm/beta')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
                for i in range(123)])

            gamma_.append([
                tf.train.load_variable(mname, 'InstanceNorm/gamma')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
                for i in range(123)])

            print('Success to load weights from:', mname)
        return w, b, beta_, gamma_

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.compat.v1.train.Saver()




class TrRosettaModelPredMany(BaseModel):
    def __init__(self, config):
        super(TrRosettaModelPredMany, self).__init__(config)
        self.build_model()

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

        w, b, beta_, gamma_ = self.init_weights()

        with tf.name_scope('input'):
            self.ncol = tf.placeholder(dtype=tf.int32, shape=())
            self.nrow = tf.placeholder(dtype=tf.int32, shape=())
            self.msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))


        # collect features
        msa1hot  = tf.one_hot(self.msa, ns, dtype=tf.float32)
        weights = reweight(msa1hot, wmin)

        # 1D features
        f1d_seq = msa1hot[0,:,:20]
        f1d_pssm = msa2pssm(msa1hot, weights)

        f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, [1,self.ncol,42])

        # 2D features
        f2d_dca = tf.cond(self.nrow>1, lambda: fast_dca(msa1hot, weights), lambda: tf.zeros([self.ncol,self.ncol,442], tf.float32))
        f2d_dca = tf.expand_dims(f2d_dca, axis=0)

        f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,self.ncol,1]), tf.tile(f1d[:,None,:,:], [1,self.ncol,1,1]), f2d_dca], axis=-1)
        f2d = tf.reshape(f2d, [1,self.ncol,self.ncol,442+2*42])

        # store ensemble of networks in separate branches
        layers2d = [[] for _ in range(len(w))]
        preds = [[] for _ in range(5)]

        Activation   = tf.nn.elu

        for i in range(len(w)):
            layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
            layers2d[i].append(activation(layers2d[i][-1]))

            # resnet
            idx = 1
            dilation = 1
            for _ in range(n2d_layers):
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(activation(layers2d[i][-1]))
                idx += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
                layers2d[i].append(activation(layers2d[i][-1] + layers2d[i][-6]))
                idx += 1
                dilation *= 2
                if dilation > 16:
                    dilation = 1

            # probabilities for theta and phi
            preds[0].append(softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
            preds[1].append(softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

            # symmetrize
            layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

            # probabilities for dist and omega
            preds[2].append(softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
            preds[3].append(softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
            preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

        with tf.name_scope("loss"):
            # average over all branches
            self.prob_theta = tf.reduce_mean(tf.stack(preds[0]),axis=0)
            self.prob_phi   = tf.reduce_mean(tf.stack(preds[1]),axis=0)
            self.prob_dist  = tf.reduce_mean(tf.stack(preds[2]),axis=0)
            self.prob_omega = tf.reduce_mean(tf.stack(preds[3]),axis=0)
            self.prob_bb = tf.reduce_mean(tf.stack(preds[4]),axis=0)


    def init_weights(self):
        # load networks in RAM
        w,b = [],[]
        beta_,gamma_ = [],[]

        DIR = self.config.trRosetta_model_dir

        for filename in os.listdir(DIR):
            if not filename.endswith(".index"):
                continue
            mname = DIR+"/"+os.path.splitext(filename)[0]
            w.append([
                tf.train.load_variable(mname, 'conv2d/kernel')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
                for i in range(128)])

            b.append([
                tf.train.load_variable(mname, 'conv2d/bias')
                if i==0 else
                tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
                for i in range(128)])

            beta_.append([
                tf.train.load_variable(mname, 'InstanceNorm/beta')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
                for i in range(123)])

            gamma_.append([
                tf.train.load_variable(mname, 'InstanceNorm/gamma')
                if i==0 else
                tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
                for i in range(123)])

            print('Success to load weights from:', mname)
        return w, b, beta_, gamma_

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = tf.compat.v1.train.Saver()
        