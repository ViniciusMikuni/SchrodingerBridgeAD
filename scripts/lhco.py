import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import utils
#import horovod.tensorflow as hvd
import argparse
from SBridge import SBridge

import pandas as pd
import gc

from sklearn.preprocessing import StandardScaler,MinMaxScaler

# 0 pxj1, 1 pyj1, 2 pzj1 ,3 mj1 ,4 tau1j1 ,5 tau2j1 ,6 tau3j1
# 7 pxj2,8 pyj2,9 pzj2,10 mj2,11 tau1j2,12 tau2j2,13 tau3j2


def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    # print(tf.data.experimental.cardinality(test_data).numpy(),"cardinality")
    # input()
    return train_data,test_data


def computemjj_np(event):
    px1 = event[:,0]
    py1 = event[:,1]
    pz1 = event[:,2]
    pE1 = np.sqrt(px1**2+py1**2+pz1**2+event[:,3]**2)
    
    px2 = event[:,7]
    py2 = event[:,8]
    pz2 = event[:,9]
    pE2 = np.sqrt(px2**2+py2**2+pz2**2+event[:,10]**2)
    
    m2 = (pE1+pE2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2
    return np.sqrt(m2)

def get_cathode_features(events,use_log=True,
                         min_feat=None,max_feat=None,                        
                         mean=None,std=None):
    #mjj,m1,m2-m1,tau121,tau221
    feat = np.zeros((events.shape[0],5),dtype=np.float32)
    
    feat[:,0] = computemjj_np(events)
    feat[:,1] = np.minimum(events[:,3],events[:,10])
    feat[:,2] = np.abs(events[:,10]-events[:,3])
    
    #Compare the masses of the jets to identify whos J1 and J2
    mask = events[:,3]>events[:,10]
    
    feat[mask,3] = np.ma.divide(events[:,5],events[:,4]).filled(0)[mask]
    feat[mask,4] = np.ma.divide(events[:,12],events[:,11]).filled(0)[mask]

    feat[mask==False,3] = np.ma.divide(events[:,12],events[:,11]).filled(0)[mask==False]
    feat[mask==False,4] = np.ma.divide(events[:,5],events[:,4]).filled(0)[mask==False]


    if min_feat is not None:
        feat = (feat-min_feat)/(max_feat-min_feat)

    else:
        scaler = MinMaxScaler()
        scaler.fit(feat)
        feat = scaler.transform(feat)
        min_feat = scaler.data_min_
        max_feat=scaler.data_max_
    
    if use_log:
        #Logit transform
        alpha = 1e-6
        x = alpha + (1 - 2*alpha)*feat
        feat = np.ma.log(x/(1-x)).filled(0)

    if mean is  None:
        mean = np.mean(feat,0)
        std = np.std(feat,0)
        
    feat = np.ma.divide(feat-mean,std).filled(0)
    
    return feat, mean,std,min_feat,max_feat

def reverse_transform(events,mean,std,min_feat,max_feat,use_log=True):    
    data = events.copy()*std + mean
    if use_log:
        alpha = 1e-6
        exp = np.exp(data)    
        x = exp/(1+exp)
        data = (x-alpha)/(1 - 2*alpha)
        
    data = data*(max_feat-min_feat)+min_feat
    return data

if __name__ == '__main__':
    # hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # if gpus:
    #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder to store plots')
    parser.add_argument('--config', default='config_lhco.json', help='Config file with training parameters')
    parser.add_argument('--stage', default=0,type=int, help='IPF iteration to load')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')

    
    
    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)
    NUM_EPOCHS = config['EPOCH']
    NUM_STAGE=50
    LR = config['LR']
    BATCH_SIZE=config['BATCH']

    #Loading the background in the sidebands
    bkg = pd.read_hdf(
        os.path.join(flags.data_folder,"events_anomalydetection_DelphesPythia8_v2_qcd_features.h5")).to_numpy().astype(np.float32)
    #[hvd.rank()::hvd.size()].astype(np.float32)
    mjj = computemjj_np(bkg)
    SB1 = bkg[(mjj<3300)& (mjj>3100)].astype(np.float32)
    SB2 = bkg[(mjj>3700)& (mjj<4000)].astype(np.float32)
    

    #Data preprocessing
    _,mean,std,minf,maxf = get_cathode_features(np.concatenate([SB1,SB2],0))
    SB1 = get_cathode_features(SB1,mean=mean,std=std,min_feat=minf,max_feat=maxf)[0]
    SB2 = get_cathode_features(SB2,mean=mean,std=std,min_feat=minf,max_feat=maxf)[0]
    nevts = min(SB1.shape[0],SB2.shape[0])
    
    tf_data1 = tf.data.Dataset.from_tensor_slices(SB1).shuffle(nevts).repeat().batch(BATCH_SIZE)
    tf_data2 = tf.data.Dataset.from_tensor_slices(SB2).shuffle(nevts).repeat().batch(BATCH_SIZE)
    del SB1,SB2,mjj,bkg,
    gc.collect()

    #Prepare the model for training
    model = SBridge(config=config)
    if flags.load:
        cfs_folder = '/global/cfs/cdirs/m3929/SB'
        checkpoint_folder = '{}/checkpoints_lhco_SB_Simple_{}'.format(cfs_folder,flags.stage)
        model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()
        assert flags.stage > 0
        
    opt_b = tf.optimizers.Adam(learning_rate=LR)
    opt_f = tf.optimizers.Adam(learning_rate=LR)
        
    model.compile(opt_f=opt_f,opt_b=opt_b)

    #print(nevts//BATCH_SIZE)
    model.fit_bridge(
        tf_data1,
        tf_data2,
        NUM_STAGE,
        NUM_EPOCHS,
        nevts//BATCH_SIZE,
        #nevts//(hvd.size()*BATCH_SIZE),
        start=flags.stage
    )


