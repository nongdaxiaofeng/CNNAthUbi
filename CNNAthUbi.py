import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers

aminoacids='ARNDCQEGHILKMFPSTWYV-'

def binary_encode(file):        
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
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

#define the topology and optimization of the convolutional neural network
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

def valid(encode,k):
    np.random.seed(1234)
    posTrainCoding,posTrainLabel=encode('pos_train_dataset')
    negTrainCoding,negTrainLabel=encode('neg_train_dataset')
    posNum=len(posTrainLabel)
    negNum=len(negTrainLabel)
    posMode=np.arange(posNum)%k
    negMode=np.arange(negNum)%k
    np.random.shuffle(posMode)
    np.random.shuffle(negMode)
    poscvScore=np.zeros(posNum)
    negcvScore=np.zeros(negNum)
    
    #k-fold cross-validation
    for fold in range(k):
        trainData=np.concatenate((np.repeat(posTrainCoding[posMode!=fold],3,axis=0),negTrainCoding[negMode!=fold]))
        trainLabel=np.concatenate((np.repeat(posTrainLabel[posMode!=fold],3),negTrainLabel[negMode!=fold]))
        m=cnn()
        m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
        testPos=posTrainCoding[posMode==fold]
        testNeg=negTrainCoding[negMode==fold]
        poscvScore[posMode==fold]=m.predict(testPos).reshape(len(testPos))
        negcvScore[negMode==fold]=m.predict(testNeg).reshape(len(testNeg))
    
    #independent test    
    trainData=np.concatenate((np.repeat(posTrainCoding,3,axis=0),negTrainCoding))
    trainLabel=np.concatenate((np.repeat(posTrainLabel,3),negTrainLabel))
    m=cnn()
    m.fit(trainData,trainLabel,batch_size=300,epochs=100,verbose=0)
    posTestCoding,posTestLabel=encode('pos_test_dataset')
    negTestCoding,negTestLabel=encode('neg_test_dataset')
    posTestScore=m.predict(posTestCoding).reshape(len(posTestCoding))
    negTestScore=m.predict(negTestCoding).reshape(len(negTestCoding))
    return posScore,negScore,posTestScore,negTestScore

poscvScore,negcvScore,posTestScore,negTestScore=valid(binary_encode,5) #prediction scores of positive samples and negative sample in five-fold cross validation and independent test
np.savez('cnnScore',posScore=posScore,negScore=negScore,posTestScore=posTestScore,negTestScore=negTestScore) #store the scores into a file


