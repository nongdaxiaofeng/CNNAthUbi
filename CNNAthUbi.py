import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
#from sklearn.metrics import roc_curve,auc

import os
os.chdir('/home/wangxf/fansu/')
aminoacids='ARNDCQEGHILKMFPSTWYV-'

def binary_encode(file):        
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    #aa2v['-']=np.zeros(20)
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([aa2v[x] for x in (s[5:25]+s[26:46])])
        label.append(col[3])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label

def binary_encode1(file):        
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    #aa2v['-']=np.zeros(20)
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        code=[]
        for x in (s[5:25]+s[26:46]):
            code+=aa2v[x]
        encoding.append(code)
        label.append(col[3])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label


def aac_encode(file):
    encoding=[]
    label=[]
    dict1={x:y for x,y in zip(aminoacids,range(21))}
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        code=np.zeros(21)
        for aa in s[5:25]+s[26:46]:
            code[dict1[aa]]+=1
        encoding.append(code)
        label.append(col[3])
    f.close()
    encoding=np.array(encoding)/40
    label=np.array(label).astype('float32')
    return encoding,label

def cksaap_encode(file):
    aa='ARNDCQEGHILKMFPSTWYV'
    encoding=[]
    label=[]
    dict0={}
    x=0
    for a1 in aa:
        for a2 in aa:
            for k in range(5):
                dict0[a1 + 'X'*k +a2]=x
                x+=1
    f=open(file)
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        s=s[5:25]+'k'+s[26:46]
        s=s.replace('-','')
        code=np.zeros(2000)
        seqlen=len(s)
        for k in range(5):
            for p in range(seqlen-k-1):
                a1,a2=s[p],s[p+k+1]
                if a1 in aa and a2 in aa:
                    code[dict0[a1 + 'X'*k + a2]]+=1
        for a1 in aa:
            for a2 in aa:
                for k in range(5):
                   code[dict0[a1 + 'X'*k + a2]]/=(seqlen-k-1) 
        encoding.append(code)
        label.append(col[3])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label
            

def cnn():
    l1,l2,gamma,lr,w1,w2=32,256,0.0001,0.001,3,2
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w1,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(40,21),padding='same'))
    model.add(layers.MaxPooling1D(w2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

def nn(l0,l1,l2,gamma,lr):
    model=models.Sequential()
    model.add(layers.Dense(l1,activation='relu',kernel_regularizer=regularizers.l1(gamma),kernel_initializer='he_normal',input_shape=(l0,)))
    #model.add(layers.Dense(l2,activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid',kernel_initializer='he_normal',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model



def cv1(encode,k):
    np.random.seed(1234)
    posTrainCoding,posTrainLabel=encode('pos_train_dataset')
    negTrainCoding,negTrainLabel=encode('neg_train_dataset')
    posNum=len(posTrainLabel)
    negNum=len(negTrainLabel)
    posMode=np.arange(posNum)%k
    negMode=np.arange(negNum)%k
    np.random.shuffle(posMode)
    np.random.shuffle(negMode)
    posScore=np.zeros(posNum)
    negScore=np.zeros(negNum)
    for fold in range(k):
        trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
        trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
        m=cnn()
        m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
        testPos=posTrainCoding[posMode==fold]
        testNeg=negTrainCoding[negMode==fold]
        posScore[posMode==fold]=m.predict(testPos).reshape(len(testPos))
        negScore[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
        
    trainData=np.concatenate((np.repeat(posTrainCoding,3,axis=0),negTrainCoding))
    trainLabel=np.concatenate((np.repeat(posTrainLabel,3),negTrainLabel))
    m=cnn()
    m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
    posTestCoding,posTestLabel=encode('pos_test_dataset')
    negTestCoding,negTestLabel=encode('neg_test_dataset')
    posTestScore=m.predict(posTestCoding).reshape(len(posTestCoding))
    negTestScore=m.predict(negTestCoding).reshape(len(negTestCoding))
    return posScore,negScore,posTestScore,negTestScore

paraDict={'binary':[128,32,0.001,0.001],'aac':[16,64,0.0001,0.01],'cksaap':[256,128,1.0e-06,0.0001]}
def cv2(encode,encodeStr,k):
    np.random.seed(1234)
    posTrainCoding,posTrainLabel=encode('pos_train_dataset')
    negTrainCoding,negTrainLabel=encode('neg_train_dataset')
    posNum=len(posTrainLabel)
    negNum=len(negTrainLabel)
    posMode=np.arange(posNum)%k
    negMode=np.arange(negNum)%k
    np.random.shuffle(posMode)
    np.random.shuffle(negMode)
    posScore=np.zeros(posNum)
    negScore=np.zeros(negNum)
    l0=posTrainCoding.shape[-1]
    l1,l2,gamma,lr=paraDict[encodeStr]
    for fold in range(k):
        trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
        trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
        m=nn(l0,l1,l2,gamma,lr)
        m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
        testPos=posTrainCoding[posMode==fold]
        testNeg=negTrainCoding[negMode==fold]
        posScore[posMode==fold]=m.predict(testPos).reshape(len(testPos))
        negScore[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
        
    trainData=np.concatenate((np.repeat(posTrainCoding,3,axis=0),negTrainCoding))
    trainLabel=np.concatenate((np.repeat(posTrainLabel,3),negTrainLabel))
    m=nn(l0,l1,l2,gamma,lr)
    m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
    posTestCoding,posTestLabel=encode('pos_test_dataset')
    negTestCoding,negTestLabel=encode('neg_test_dataset')
    posTestScore=m.predict(posTestCoding).reshape(len(posTestCoding))
    negTestScore=m.predict(negTestCoding).reshape(len(negTestCoding))
    return posScore,negScore,posTestScore,negTestScore

posScore,negScore,posTestScore,negTestScore=cv1(binary_encode,5)
np.savez('cnnScore',posScore=posScore,negScore=negScore,posTestScore=posTestScore,negTestScore=negTestScore)

posScore,negScore,posTestScore,negTestScore=cv2(binary_encode1,'binary',5)
np.savez('binaryScore',posScore=posScore,negScore=negScore,posTestScore=posTestScore,negTestScore=negTestScore)

posScore,negScore,posTestScore,negTestScore=cv2(aac_encode,'aac',5)
np.savez('aacScore',posScore=posScore,negScore=negScore,posTestScore=posTestScore,negTestScore=negTestScore)

posScore,negScore,posTestScore,negTestScore=cv2(cksaap_encode,'cksaap',5)
np.savez('cksaapScore',posScore=posScore,negScore=negScore,posTestScore=posTestScore,negTestScore=negTestScore)

