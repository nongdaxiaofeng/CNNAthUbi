import numpy as np
from tensorflow.keras import models

def binary_encode(peptides):
    aminoacids='ARNDCQEGHILKMFPSTWYVX'
    aa2v={x:y for x,y in zip(aminoacids,np.eye(21,21).tolist())}
    #aa2v['-']=np.zeros(20)
    encoding=[]
    for s in peptides:
        encoding.append([aa2v[x] for x in (s)])
    encoding=np.array(encoding)
    return encoding

def predict_seq(locus,seq):
    sitePosition=[]
    peptides=[]
    seqLen=len(seq)
    ind=seq.find('K',0)
    while ind!=-1:
        sitePosition.append(ind)
        if ind<20:
            left='X'*(20-ind)+seq[0:ind]
        else:
            left=seq[ind-20:ind]
        if ind+21>seqLen:
            if ind==seqLen-1:
                right='X'*20
            else:
                right=seq[ind+1:]+'X'*(ind+21-seqLen)
        else:
            right=seq[ind+1:ind+21]
        peptides.append(left+right)
        ind=seq.find('K',ind+1)
    if peptides:
        encoding=binary_encode(peptides)
        score=model.predict(encoding).reshape(len(peptides))
        f1=open('proteome_score','a')
        for i in range(len(peptides)):
            f1.write(locus+','+str(sitePosition[i])+','+peptides[i]+','+str(score[i])+'\n')
        f1.close()

model=models.load_model('Model.h5')


f=open('sequences.fasta','r')
seq=''
line=f.readline()
locus=line[1:line.index('|')-1]
for line in f:
    if line[0]=='>':
        if seq:
            seq=seq.replace('*','')
            predict_seq(locus,seq)
        locus=line[1:line.index('|')-1]
        seq=''
    else:
        seq+=line.strip()
f.close()
