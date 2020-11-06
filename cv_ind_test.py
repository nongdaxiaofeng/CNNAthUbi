import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import models,layers,optimizers,regularizers
import matplotlib.pyplot as plt

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


l1,l2,gamma,lr,w=16,64,1e-4,1e-3,3
k=5
np.random.seed(1234)
posTrainCoding,posTrainLabel=binary('dataset/pos_train_dataset')
negTrainCoding,negTrainLabel=binary('dataset/neg_train_dataset')
posNum=len(posTrainLabel)
negNum=len(negTrainLabel)
posMode=np.arange(posNum)%k
negMode=np.arange(negNum)%k
np.random.shuffle(posMode)
np.random.shuffle(negMode)
posScoreb=np.zeros(posNum)
negScoreb=np.zeros(negNum)
label=np.concatenate((posTrainLabel,negTrainLabel))
for fold in range(k):
    trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
    trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
    m=cnn(l1=l1,l2=l2,depth=21,gamma=gamma,lr=lr,w=3)
    m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
    testPos=posTrainCoding[posMode==fold]
    testNeg=negTrainCoding[negMode==fold]
    posScoreb[posMode==fold]=m.predict(testPos).reshape(len(testPos))
    negScoreb[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
fprb,tprb,thrb=roc_curve(label,np.concatenate((posScoreb,negScoreb)))
aurocb=auc(fprb,tprb)
accb=(3*np.sum(posScoreb>0.5)+np.sum(negScoreb<=0.5))/(3*posNum+negNum)

trainData=np.concatenate((np.repeat(posTrainCoding,3,axis=0),negTrainCoding))
trainLabel=np.concatenate((np.repeat(posTrainLabel,3),negTrainLabel))
posTestCoding,posTestLabel=binary('dataset/pos_test_dataset')
negTestCoding,negTestLabel=binary('dataset/neg_test_dataset')
testData=np.concatenate((posTestCoding,negTestCoding))
testLabel=np.concatenate((posTestLabel,negTestLabel))
m=cnn(l1=l1,l2=l2,depth=21,gamma=gamma,lr=lr,w=3)
m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
testScoreb=m.predict(testData).reshape(len(testData))
testfprb,testtprb,testthrb=roc_curve(testLabel,testScoreb)
testaurocb=auc(testfprb,testtprb)
testaccb=(3*np.sum(testScoreb[testLabel==1]>0.5)+np.sum(testScoreb[testLabel==0]<=0.5))/(3*np.sum(testLabel==1)+np.sum(testLabel==0))


l1,l2,gamma,lr,w=32,512,1e-4,1e-3,3
posTrainCoding,posTrainLabel=aaindex('dataset/pos_train_dataset')
negTrainCoding,negTrainLabel=aaindex('dataset/neg_train_dataset')
posScorea=np.zeros(posNum)
negScorea=np.zeros(negNum)
label=np.concatenate((posTrainLabel,negTrainLabel))
for fold in range(k):
    trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
    trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
    m=cnn(l1=l1,l2=l2,depth=31,gamma=gamma,lr=lr,w=3)
    m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
    testPos=posTrainCoding[posMode==fold]
    testNeg=negTrainCoding[negMode==fold]
    posScorea[posMode==fold]=m.predict(testPos).reshape(len(testPos))
    negScorea[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
fpra,tpra,thra=roc_curve(label,np.concatenate((posScorea,negScorea)))
auroca=auc(fpra,tpra)
acca=(3*np.sum(posScorea>0.5)+np.sum(negScorea<=0.5))/(3*posNum+negNum)

trainData=np.concatenate((np.repeat(posTrainCoding,3,axis=0),negTrainCoding))
trainLabel=np.concatenate((np.repeat(posTrainLabel,3),negTrainLabel))
posTestCoding,posTestLabel=aaindex('dataset/pos_test_dataset')
negTestCoding,negTestLabel=aaindex('dataset/neg_test_dataset')
testData=np.concatenate((posTestCoding,negTestCoding))
m=cnn(l1=l1,l2=l2,depth=31,gamma=gamma,lr=lr,w=3)
m.fit(trainData,trainLabel,batch_size=100,epochs=50,verbose=0)
testScorea=m.predict(testData).reshape(len(testData))
testfpra,testtpra,testthra=roc_curve(testLabel,testScorea)
testauroca=auc(testfpra,testtpra)
testacca=(3*np.sum(testScorea[testLabel==1]>0.5)+np.sum(testScorea[testLabel==0]<=0.5))/(3*np.sum(testLabel==1)+np.sum(testLabel==0))

chencv=pd.read_table('AraUbiSite_cv_score.txt',header=0,usecols=[4,5,6]).to_numpy()
fprc,tprc,thrc=roc_curve(chencv[:,2],chencv[:,0])
aurocc=auc(fprc,tprc)
accc=(3*np.sum(chencv[chencv[:,2]==1,0]>=0.5)+np.sum(chencv[chencv[:,2]==-1,0]<0.5))/(3*posNum+negNum)

chentest=pd.read_table('AraUbiSite_ind_score.txt',header=0,usecols=[4,5,6]).to_numpy()
testfprc,testtprc,testthrc=roc_curve(chentest[:,2],chentest[:,0])
testaurocc=auc(testfprc,testtprc)
testaccc=(3*np.sum(chentest[chentest[:,2]==1,0]>=0.5)+np.sum(chentest[chentest[:,2]==-1,0]<0.5))/(3*np.sum(testLabel==1)+np.sum(testLabel==0))

lw = 2
plt.figure(figsize=[12,4.5])
plt.subplot(121)
plt.plot(fprb, tprb, color='red',lw=lw, label= 'CNN_Binary        AUC = {:.3f}'.format(aurocb))
plt.plot(fpra, tpra, color='blue',lw=lw, label='CNN_Property     AUC = {:.3f}'.format(auroca))
plt.plot(fprc, tprc, color='green',lw=lw, label= 'AraUbiSite          AUC = {:.3f}'.format(aurocc))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

plt.subplot(122)
plt.plot(testfprb, testtprb, color='red',lw=lw, label= 'CNN_Binary        AUC = {:.3f}'.format(testaurocb))
plt.plot(testfpra, testtpra, color='blue',lw=lw, label='CNN_Property     AUC = {:.3f}'.format(testauroca))
plt.plot(testfprc, testtprc, color='green',lw=lw, label=  'AraUbiSite          AUC = {:.3f}'.format(testaurocc))
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()