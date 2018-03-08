import os
from sklearn.externals import joblib
lemma_clf=joblib.load('lemma.model') 
tag_clf=joblib.load('tag.model')
from meta import feats_lemma,feats_tag
for file in os.listdir('/home/tomaz/Project/SSJ/Gigafida/MetaTag/gf2tag'):
    f=open('out/'+file,'w')
    for line in open('/home/tomaz/Project/SSJ/Gigafida/MetaTag/gf2tag/'+file):
        if line.strip()=='':
            f.write('\n')
            continue
        token,tag1,lemma1,tag2,lemma2=line[:-1].split('\t')
        f.write(token+'\t')
        if tag2=='':
            f.write(tag1+'\t')
        else:
            pred=tag_clf.predict(feats_tag(tag1,tag2))[0]
            if pred==1:
                f.write(tag1+'\t')
            else:
                f.write(tag2+'\t')
        if lemma2=='':
            f.write(lemma1+'\n')
        else:
            if tag2=='':
                tag2=tag1
            pred=lemma_clf.predict(feats_lemma(tag1,tag2))[0]
            if pred==1:
                f.write(lemma1+'\n')
            else:
                f.write(lemma2+'\n')
    f.close()