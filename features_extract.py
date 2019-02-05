
from python_speech_features import mfcc
from python_speech_features import delta
from sklearn import preprocessing
import numpy as np
import scipy.io.wavfile as wav

#      mfcc (signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,
#            nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
#            ceplifter=22,appendEnergy=True)

      
#      delta (feat, N-> For each frame, calculate delta features based on preceding and following N frames)



def mfcc_delta(wavfile):
    (rate,audio)=wav.read(wavfile)
    mfcc_feat=mfcc(audio,rate,numcep=13,nfft=2048)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    mfcc_delta1= delta(mfcc_feat, 2)
    mfcc_delta2= delta(mfcc_delta1, 2)
    combo=np.hstack((mfcc_feat,mfcc_delta1,mfcc_delta2))
    return combo
