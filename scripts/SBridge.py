import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import utils
#import horovod.tensorflow as hvd
from architecture import Resnet
from tqdm import tqdm
import gc
utils.SetStyle()

tf.random.set_seed(12345)

class SBridge(keras.Model):
    """Schrodinger Bridge Tensorflow implementation"""
    def __init__(self, name='lhco',config=None):
        super(SBridge, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")
        
        self.num_dim=self.config['NDIM']
        self.num_embed=self.config['EMBED']
        self.num_steps =self.config['NSTEPS']
        self.title=name
        self.ema=0.999
        self.activation = layers.LeakyReLU(alpha=0.01)
        
        gamma_space = self.config['gamma_space']
        gamma_min = self.config['gamma_min']
        gamma_max = self.config['gamma_max']
        self.mean_match=False
        
        n = self.num_steps//2
        self.mean_final = 0.0
        self.var_final = 1.0*10**3 

        if gamma_space == 'linspace':
            gamma_half = np.linspace(float(gamma_min), float(gamma_max), n)
        elif gamma_space == 'geomspace':
            gamma_half = np.geomspace(float(gamma_min), float(gamma_max), n)


        self.gammas = tf.cast(tf.concat([gamma_half, np.flip(gamma_half)],0),tf.float32)
        
        self.T = tf.cast(tf.reduce_sum(self.gammas),tf.float32)
        self.time = tf.cumsum(self.gammas, 0)/self.T
        self.T=1.0

        
        
        inputs_time = Input((1))
        self.projection = self.GaussianFourierProjection(scale = 16)
        time_forward = self.Embedding(inputs_time,self.projection)
        time_backward = self.Embedding(inputs_time,self.projection)
        
        

        inputs_f,outputs_f = Resnet(
            self.num_dim,
            self.num_dim,
            time_forward,
            self.num_embed,
            num_layer = self.config['NLAYERS'],
            mlp_dim = self.config['LAYER_SIZE'],
        )
        inputs_b,outputs_b = Resnet(
            self.num_dim,
            self.num_dim,
            time_backward,
            self.num_embed,
            num_layer = self.config['NLAYERS'],
            mlp_dim = self.config['LAYER_SIZE'],
        )

        self.forward = keras.models.Model([inputs_f,inputs_time], outputs_f, name="forward")
        self.backward = keras.models.Model([inputs_b,inputs_time], outputs_b, name="backward")
        
        self.ema_f = keras.models.clone_model(self.forward)
        self.ema_b = keras.models.clone_model(self.backward)
        
    def GaussianFourierProjection(self,scale = 30):
        return tf.constant(tf.random.normal(shape=(1,self.num_embed//2),seed=100))*scale*2*np.pi


    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding
        
    def compile(self,opt_f, opt_b):
        super(SBridge, self).compile(experimental_run_tf_function=False,
                                        #run_eagerly=True
        )
        self.opt_f = opt_f
        self.opt_b = opt_b

    

    def propagate(self,sample,model,direction='forward'):
        
        x = sample

        N = tf.shape(x)[0]
        time = tf.cast(tf.repeat(tf.reshape(self.time,(1, self.num_steps, 1)),N,0),tf.float32)
        gammas = tf.cast(tf.repeat(tf.reshape(self.gammas,(1, self.num_steps, 1)),N,0),tf.float32)


        for idx in range(self.num_steps):
            gamma = self.gammas[idx]
            t_old = model([x,time[:,idx]])

            if not self.mean_match:
                t_old += x

            if (idx == self.num_steps-1):
                x = t_old
            else:
                z = tf.random.normal(tf.shape(x))            
                x = t_old + tf.cast(tf.sqrt(2 * gamma),tf.float32)*z

            t_new = model([x,time[:,idx]])

            if not self.mean_match:
                t_new += x
                
            if idx==0:
                xs = tf.expand_dims(x,1)
                out = tf.expand_dims(t_old - t_new,1)
            else:
                xs = tf.concat([xs,tf.expand_dims(x,1)],1)
                out = tf.concat([out,tf.expand_dims(t_old - t_new,1)],1)



        return tf.convert_to_tensor(xs,dtype=tf.float32),tf.convert_to_tensor(out,dtype=tf.float32), time



    def propagate_first(self,sample,direction='forward'):
        #First difufsion without any learning parameter to initialize the model
        x = sample

        N = tf.shape(x)[0]
        time = tf.cast(tf.repeat(tf.reshape(self.time,(1, self.num_steps, 1)),N,0),tf.float32)
        gammas = tf.cast(tf.repeat(tf.reshape(self.gammas,(1, self.num_steps, 1)),N,0),tf.float32)
 
        for idx in range(self.num_steps):

            def grad_gauss(x,mean,var):
                return -(x-mean)/var

            
            gamma = self.gammas[idx]
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_old = x + gamma * gradx
            
            z = tf.random.normal(tf.shape(x))
            x = t_old + tf.cast(tf.sqrt(2 * gamma),tf.float32)*z
            
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_new = x + gamma * gradx

            if idx ==0:
                xs = tf.expand_dims(x,1)
                out = tf.expand_dims(t_old - t_new,1)
            else:
                xs = tf.concat([xs,tf.expand_dims(x,1)],1)
                out = tf.concat([out,tf.expand_dims(t_old - t_new,1)],1)

        return tf.convert_to_tensor(xs,dtype=tf.float32),tf.convert_to_tensor(out,dtype=tf.float32), time

    
    def get_loss_fn(self):
        @tf.function
        def loss_fn(sample,policy_opt, policy_impt,
                    sample_direction,opt,
                    ema_opt,
                    stage=1,first_epoch=False):
        
            if stage==0 and sample_direction == 'backward':        
                #training forward sampling from backward
                train_xs,train_zs,steps = self.propagate_first(sample,sample_direction)
            else:
                train_xs,train_zs,steps = self.propagate(sample,policy_impt,sample_direction)
                
            
            with tf.GradientTape() as tape:
                # prepare training data
                
                # -------- handle for batch_x and batch_t ---------
                # (batch, T, xdim) --> (batch*T, xdim)
                xs      = tf.reshape(train_xs,(-1,self.num_dim)) 
                zs_impt = tf.reshape(train_zs,(-1,self.num_dim))
                ts = tf.reshape(self.T -tf.cast(steps,dtype=tf.float32),(-1,1))

                

                # -------- compute loss and backprop --------

                pred = policy_opt([xs, ts])
                if self.mean_match:
                    pred -= xs
                        
                loss = tf.square(pred - zs_impt)
                loss = tf.reduce_mean(loss)

            # tape = hvd.DistributedGradientTape(tape)
            variables = policy_opt.trainable_variables
            grads = tape.gradient(loss, variables)
            grads = [tf.clip_by_norm(grad, 1)
                     for grad in grads]
            opt.apply_gradients(zip(grads, variables))
            

            for weight, ema_weight in zip(policy_opt.weights, ema_opt.weights):
                ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
            
            # if first_epoch:
            #     hvd.broadcast_variables(policy_opt.variables, root_rank=0)
            #     hvd.broadcast_variables(opt.variables(), root_rank=0)
                    
                
            return loss
        return loss_fn


    def reset_opt(self,opt):
        for var in opt.variables():
            var.assign(tf.zeros_like(var))
        return tf.constant(10)


    def sb_alternate_train_stage(self,iterator,stage, epoch, direction,loss_fn,NUM_STEPS):
        policy_opt, policy_impt, sample_direction,opt,ema_opt = {
            'forward':  [self.forward, self.backward,'backward',self.opt_f,self.ema_f], # train forwad,   sample from backward
            'backward': [self.backward, self.forward,'forward',self.opt_b,self.ema_b], # train backward, sample from forward
        }.get(direction)

        epochs = tqdm(range(epoch))
        # if hvd.rank()==0:
        #     epochs = tqdm(range(epoch))
        # else:
        #     epochs = range(epoch)

        patience = 0
        stop_ep=0
        early_stopping=10
        min_epoch = np.inf
        for ep in epochs:
            sum_loss = []
            #print('Training epoch {}'.format(ep))
            for step in range(NUM_STEPS):
                first_epoch = stage==self.start and step==0 and ep ==0
                sample = iterator.get_next()
                loss = loss_fn(sample,
                               policy_opt, policy_impt,
                               sample_direction,opt,
                               ema_opt,
                               stage,first_epoch=first_epoch)
                sum_loss.append(loss.numpy())
            if np.mean(sum_loss) < min_epoch:
                min_epoch=np.mean(sum_loss)
                patience=0
            else:
                patience+=1
            if patience >=early_stopping and stop_ep==0:
                stop_ep = ep
                
            epochs.set_description("Loss: {:.2f}".format(loss*10e4))
            # if hvd.rank()==0:
            #    epochs.set_description("Loss: {}".format(loss))
            gc.collect()
        # self.reset_opt(opt)
        # self.optimizer.learning_rate.assign(self.config['LR']*hvd.size()/(2**stage))
        return tf.reduce_mean(sum_loss),stop_ep


    def fit_bridge(self,
                   prior,
                   posterior,
                   NUM_STAGE,
                   NUM_EPOCHS,
                   NUM_STEPS,
                   start=0,
                   
    ):        
        iterator_prior = iter(prior)
        iterator_posterior = iter(posterior)
        loss_fn_f = self.get_loss_fn()
        loss_fn_b = self.get_loss_fn()
        self.start=start
        for stage in range(start,NUM_STAGE):
            #if hvd.rank()==0:
            print("Training stage {}".format(stage))
            # Note: first stage of forward policy must converge;
            # otherwise it will mislead backward policy
            forward_ep = 200 if stage ==0 else NUM_EPOCHS
            backward_ep =  NUM_EPOCHS

            # train forward policy
            loss,ep = self.sb_alternate_train_stage(
                iterator_posterior,stage,
                forward_ep, 'forward',
                loss_fn_f,NUM_STEPS)
            
            #if hvd.rank()==0:
            print("Trained forward model with loss {} in {} epochs".format(loss,ep))

            gc.collect()
            # train backward policy;            
            loss,ep = self.sb_alternate_train_stage(
                iterator_prior,stage,
                backward_ep, 'backward',
                loss_fn_b,NUM_STEPS)
            
            #if hvd.rank()==0:
            print("Trained backward model with loss {} in {} epochs".format(loss,ep))
            gc.collect()
                                    
            #if hvd.rank()==0:
            cfs_folder = '/global/cfs/cdirs/m3929/SB'
            checkpoint_folder = '{}/checkpoints_{}_SB_Simple_{}'.format(cfs_folder,self.title,stage)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            self.save_weights('{}/{}'.format(checkpoint_folder,'checkpoint'),save_format='tf')
    
if __name__ == '__main__':
    pass
