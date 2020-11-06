import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import models,layers,optimizers,regularizers

def aaindex(file):
    index=pd.read_table('aaindex31',sep='\s+',header=None)
    index=index.subtract(index.min(axis=1),axis=0).divide((index.max(axis=1)-index.min(axis=1)),axis=0)
    index=index.to_numpy().T
    index={x:y for x,y in zip('ARNDCQEGHILKMFPSTWYV',index.tolist())}
    index['X']=np.zeros(31).tolist()
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([index[x] for x in (s[0:20]+s[21:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label

def binary(file):
    aminoacids='ARNDCQEGHILKMFPSTWYVX'
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    #aa2v['-']=np.zeros(20)
    encoding=[]
    label=[]
    f=open(file,'r')
    for line in f:
        col=line.strip().split('\t')
        s=col[0]
        encoding.append([aa2v[x] for x in (s[0:20]+s[21:])])
        label.append(col[-1])
    f.close()
    encoding=np.array(encoding)
    label=np.array(label).astype('float32')
    return encoding,label

def cnn(l1,l2,depth,gamma,lr,w):
    model=models.Sequential()
    model.add(layers.Conv1D(l1,w,activation='relu',kernel_regularizer=regularizers.l1(gamma),input_shape=(40,depth),padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(l2,activation='relu',kernel_regularizer=regularizers.l1(gamma)))
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(gamma)))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    return model

k=5
np.random.seed(1234)
posTrainCoding,posTrainLabel=aaindex('dataset/pos_train_dataset')
negTrainCoding,negTrainLabel=aaindex('dataset/neg_train_dataset')
posNum=len(posTrainLabel)
negNum=len(negTrainLabel)
posMode=np.arange(posNum)%k
negMode=np.arange(negNum)%k
np.random.shuffle(posMode)
np.random.shuffle(negMode)
posScore=np.zeros(posNum)
negScore=np.zeros(negNum)
np.random.seed(1234)
label=np.concatenate((posTrainLabel,negTrainLabel))

for l1 in 2**np.arange(4,8):
    for l2 in 2**np.arange(4,10):
        for gamma in 10.0**np.arange(-1,-6,-1):
            for lr in 10.0**np.arange(-1,-6,-1):
                for w in range(2,10):
                    for fold in range(k):
                        trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
                        trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
                        m=cnn(l1=l2,l2=l2,depth=31,gamma=gamma,lr=lr,w=w)
                        m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
                        testPos=posTrainCoding[posMode==fold]
                        testNeg=negTrainCoding[negMode==fold]
                        posScore[posMode==fold]=m.predict(testPos).reshape(len(testPos))
                        negScore[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
                

                    fpr,tpr,_=roc_curve(label,np.concatenate((posScore,negScore)))
                    auroc=auc(fpr,tpr)
                    acc=(3*np.sum(posScore>=0.5)+np.sum(negScore<0.5))/(3*posNum+negNum)
                    f=open('cnnParaResult_property','a')
                    f.write('{:d}\t{:d}\t{:f}\t{:f}\t{:d}\t{:f}\t{:f}\n'.format(l1,l2,gamma,lr,w,auroc,acc))
                    f.close()


posTrainCoding,posTrainLabel=binary('dataset/pos_train_dataset')
negTrainCoding,negTrainLabel=binary('dataset/neg_train_dataset')
posScore=np.zeros(posNum)
negScore=np.zeros(negNum)

for l1 in 2**np.arange(4,8):
    for l2 in 2**np.arange(4,10):
        for gamma in 10.0**np.arange(-1,-6,-1):
            for lr in 10.0**np.arange(-1,-6,-1):
                for w in range(2,10):
                    for fold in range(k):
                        trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
                        trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
                        m=cnn(l1=l1,l2=l2,depth=21,gamma=gamma,lr=lr,w=w)
                        m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
                        testPos=posTrainCoding[posMode==fold]
                        testNeg=negTrainCoding[negMode==fold]
                        posScore[posMode==fold]=m.predict(testPos).reshape(len(testPos))
                        negScore[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
                

                    fpr,tpr,_=roc_curve(label,np.concatenate((posScore,negScore)))
                    auroc=auc(fpr,tpr)
                    acc=(3*np.sum(posScore>=0.5)+np.sum(negScore<0.5))/(3*posNum+negNum)
                    f=open('cnnParaResult_binary','a')
                    f.write('{:d}\t{:d}\t{:f}\t{:f}\t{:d}\t{:f}\t{:f}\n'.format(l1,l2,gamma,lr,w,auroc,acc))
                    f.close()