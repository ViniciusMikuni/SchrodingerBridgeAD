import utils
import matplotlib.pyplot as plt
import numpy as np
from SBridge import SBridge
from lhco import computemjj_np,get_cathode_features,reverse_transform
import argparse
import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
utils.SetStyle()



def match_mass(mass,diffused_sample,direction='forward'):
    nevts = SR.shape[0]
    if nevts>diffused_sample.shape[0]:
        diffused_sample = np.resize(diffused_sample,
                                    (nevts,diffused_sample.shape[1],diffused_sample.shape[2]))
    if direction=='forward':    
        sort=np.argsort(diffused_sample[:,-1,0],0)
    else:
        sort=np.argsort(diffused_sample[:,0,0],0)
        
    diffused_sample=np.take_along_axis(diffused_sample,np.reshape(sort,(-1,1,1)),0)

                               
    SR_mass = np.expand_dims(np.sort(mass,0),-1)            
    delta = np.abs(SR_mass-diffused_sample[:nevts,:,0]) #compare end points of the diffusion with the masses to be sampled
    idx = np.reshape(np.argmin(delta,-1),(-1,1,1)) #find the time step where the diffused is closest to the value we want
    diffused_sample = np.squeeze(np.take_along_axis(diffused_sample[:nevts],idx,1)) #keep only that time step
    return diffused_sample



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder to load the data')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
    parser.add_argument('--config', default='config_lhco.json', help='Config file with training parameters')
    parser.add_argument('--nevts', default=3000,type=int, help='Number of signal events to load')
    parser.add_argument('--stage', default=3,type=int, help='IPF iteration to load')
    
    flags = parser.parse_args()    
    bkg = pd.read_hdf(
        os.path.join(flags.data_folder,"events_anomalydetection_DelphesPythia8_v2_qcd_features.h5")).to_numpy()

    sig = pd.read_hdf(
        os.path.join(flags.data_folder,"events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5")).to_numpy()[:flags.nevts]

    if not os.path.exists(flags.plot_folder):
        os.makedirs(flags.plot_folder)

    config = utils.LoadJson(flags.config)    
    mjj = computemjj_np(bkg)
    mjj_sig = computemjj_np(sig)

    SB1 = bkg[(mjj<3300)& (mjj>3100)]
    SB2 = bkg[(mjj>3700)&(mjj<4000)]
    SR = bkg[(mjj>3300)&(mjj<3700)]
    
    _,mean,std,minf,maxf = get_cathode_features(np.concatenate([SB1,SB2],0))
    
    SB1 = get_cathode_features(SB1,mean=mean,std=std,min_feat=minf,max_feat=maxf)[0]
    SB2 = get_cathode_features(SB2,mean=mean,std=std,min_feat=minf,max_feat=maxf)[0]
    SR = get_cathode_features(SR,mean=mean,std=std,min_feat=minf,max_feat=maxf)[0]
    
    sig_features = get_cathode_features(sig,
                                        min_feat=minf,
                                        max_feat=maxf,
                                        mean=mean,std=std)[0]

    SR_sig = sig_features[(mjj_sig>3300)&(mjj_sig<3700)]
    nsample = 2 #number of times to oversample the background prediction

    
    model = SBridge(config=config)
    cfs_folder = '/global/cfs/cdirs/m3929/SB'
    checkpoint_folder = '{}/checkpoints_lhco_SB_Simple_{}'.format(cfs_folder,flags.stage)
    model.load_weights('{}/{}'.format(checkpoint_folder,'checkpoint')).expect_partial()

    SR_forward = []
    SR_backward = []
    for _ in range(nsample):
        #Use EMA to sample
        forward_xs= model.propagate(SB1,model.ema_f,'forward')[0].numpy()
        backward_xs= model.propagate(SB2,model.ema_b,'backward')[0].numpy()
        
        SR_forward.append(match_mass(SR[:,0],forward_xs))
        SR_backward.append(match_mass(SR[:,0],backward_xs))

    SR_forward = np.concatenate(SR_forward)
    SR_backward = np.concatenate(SR_backward)
        
    SB1 = reverse_transform(SB1,mean,std,min_feat=minf,max_feat=maxf)
    SB2 = reverse_transform(SB2,mean,std,min_feat=minf,max_feat=maxf)
    forward = reverse_transform(forward_xs[:,-1],mean,std,min_feat=minf,max_feat=maxf)
    backward = reverse_transform(backward_xs[:,-1],mean,std,min_feat=minf,max_feat=maxf)
        
    SR_forward = reverse_transform(SR_forward,mean,std,min_feat=minf,max_feat=maxf)
    SR_backward = reverse_transform(SR_backward,mean,std,min_feat=minf,max_feat=maxf)
    combined = np.concatenate([SR_backward,SR_forward],0)
    SR = reverse_transform(SR,mean,std,min_feat=minf,max_feat=maxf)
    

    for i in range(0,5):        
        feed_dict={
            'SB1':SB1[:,i],
            'SB2':SB2[:,i],
            'SB1_forward':forward[:,i],
            'SB2_backward':backward[:,i],

            # 'SB1':SB1[:,i]*var[i]+mean[i],
            # 'SB2':SB2[:,i]*var[i]+mean[i],
            # 'SB1_forward':forward_xs[:,-1,i]*var[i]+mean[i],
            # 'SB2_backward':backward_xs[:,0,i]*var[i]+mean[i],
            
            
        }
    
        fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                   #binning=np.linspace(3000,4100,20) if i==0 else None,
                                   binning=np.linspace(2500,4500,20) if i==0 else None,
                                   xlabel='{}'.format(i),logy=False,
                                   ylabel='Normalized events',
                                   reference_name='SB1')

        fig.savefig('{}/SBs{}.pdf'.format(flags.plot_folder,i))

    for i in range(0,5):        
        feed_dict={
            'SR':SR[:,i],
            'combined':combined[:,i]
            # 'SB1_forward':SR_forward[:,i],
            # 'SB2_backward':SR_backward[:,i],
        }
    
        fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                   binning=np.linspace(3300,3700,20) if i==0 else None,
                                   xlabel='{}'.format(i),logy=False,
                                   ylabel='Normalized events',
                                   reference_name='SR')

        fig.savefig('{}/SR{}.pdf'.format(flags.plot_folder,i))

        


def Classifier():
    from tensorflow import keras
    import tensorflow as tf
    from sklearn.utils import shuffle
    NUM_EPOCHS=100
    BATCH_SIZE=128
    train = np.concatenate([np.concatenate([SR,SR_sig],0),combined],0)    
    #train = np.concatenate([np.concatenate([SR,SR_sig],0),SR_forward],0)
    
    scaler = StandardScaler()
    scaler.fit(train)
    
    train = scaler.transform(train)
    labels = np.concatenate([np.ones((SR.shape[0]+SR_sig.shape[0])),
                             np.zeros((combined.shape[0]))],0)
    nevts = combined.shape[0]+SR.shape[0]+SR_sig.shape[0]
    class_weight = {
        0:1.0*nevts/combined.shape[0],
        1:1.0*nevts/(SR.shape[0]+SR_sig.shape[0]),
    }
    train,labels=shuffle(train,labels)
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ])

    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    model.fit(train, labels,
              batch_size=BATCH_SIZE,
              class_weight = class_weight,
              epochs=NUM_EPOCHS,shuffle=True,)


    val = scaler.transform(np.concatenate([SR,SR_sig],0))
    pred = model.predict(val)

    labels_anomaly = np.concatenate([np.zeros((SR.shape[0])),
                                     np.ones((SR_sig.shape[0]))],0)
    fpr, tpr, _ = roc_curve(labels_anomaly,pred, pos_label=1)    
    auc_res =auc(fpr, tpr)
    print("Max SIC: {}".format(np.max(np.ma.divide(tpr,np.sqrt(fpr)).filled(0))))
    print("s/b: {}, s/sqrt(b): {}, s: {}, b: {}".format(SR_sig.shape[0]*1.0/SR.shape[0]*100,
                                                        SR_sig.shape[0]*1.0/np.sqrt(SR.shape[0]),
                                                        SR_sig.shape[0],
                                                        SR.shape[0]
    ))
    
    plt.figure(figsize=(10,8))
    plt.plot(tpr, 1.0/fpr,
             "-", label='Diffusion (auc = %.1f%%)'%(auc_res*100.), linewidth=1.5)
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.semilogy()
        
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('{}/roc.pdf'.format(flags.plot_folder))

    plt.figure(figsize=(10,8))
    plt.plot(1.0/fpr, tpr/np.sqrt(fpr),"-", label='Diffusion', linewidth=1.5)
    plt.xlabel("Rejection")
    plt.ylabel("SIC")
    plt.semilogx()
        
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('{}/sic.pdf'.format(flags.plot_folder))
    
Classifier()
