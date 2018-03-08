#!-*-coding:utf8-*-
"""
import xml.etree.ElementTree as et
corpus=et.parse('master.xml')
tags1={}
tags2={}
for sentence in corpus.getroot():
  sid=sentence.attrib['{http://www.w3.org/XML/1998/namespace}id']
  tid=0
  for token in sentence:
    if token.tag in ('w','pc'):
      tid+=1
      tags1[sid+'.'+str(tid)]=token.attrib['ana'].split(' ')[0][4:]
      tags2[sid+'.'+str(tid)]=token.attrib['ana'].split(' ')[-1][4:]
import cPickle as pickle
pickle.dump(tags1,open('tags1.pickle','w'),1)
pickle.dump(tags1,open('tags2.pickle','w'),1)
"""
slen=dict([e.strip().split('\t')[1:] for e in open('josMSD.tbl')])
slfeats=dict([e.strip().split('\t') for e in open('josMSD-canon-sl.tbl')])
for tag in slfeats:
  slfeats[tag]=dict([e.split('=') for e in slfeats[tag].split(' ')[1:]])

def feats_lemma(tag1,tag2):
  instance={
  'tag1':tag1,
  'tag2':tag2,
  'pos1':tag1[0],
  'pos2':tag2[0],
  'pos21':tag1[:2],
  'pos22':tag2[:2],
  }
  for feat,value in slfeats.get(tag1,{}).items():
    instance[feat+'1']=value
  for feat,value in slfeats.get(tag2,{}).items():
    instance[feat+'2']=value
  return instance

def feats_tag(tag1,tag2):
  instance={
  'tag1':tag1,
  'tag2':tag2,
  'pos1':tag1[0],
  'pos2':tag2[0],
  'pos21':tag1[:2],
  'pos22':tag2[:2],
  }
  for feat,value in slfeats.get(tag1,{}).items():
    instance[feat+'1']=value
  for feat,value in slfeats.get(tag2,{}).items():
    instance[feat+'2']=value
  return instance

if __name__=='__main__':
  import cPickle as pickle
  tags1=pickle.load(open('tags1.pickle'))
  tags2=pickle.load(open('tags2.pickle'))
  def check_agreement(tag1,tag2,feat):
    if tag1 in slfeats and tag2 in slfeats:
      if slfeats[tag1][feat]=='0' or slfeats[tag2][feat]=='0':
        return -1
      if slfeats[tag1][feat]==slfeats[tag2][feat]:
        return 1
      else:
        return 0
    return -1
  #print slfeats
  ytag=[]
  Xtag=[]
  Xtag_str=[]
  ylemma=[]
  Xlemma=[]
  i=0
  triplets=[]
  for line in open('meta.csv'):
    token,tag1,tag2,lemma1,lemma2,tagc1,tagc2,lemmac1,lemmac2,tid=line[:-1].split(',')
    i+=1
    if lemmac1!='-':
      if lemma1!=lemmac1 or lemma2!=lemmac2:
        if lemma1==lemmac1:
          ylemma.append(1)
        elif lemma2==lemmac2:
          ylemma.append(2)
        else:
          print 'error',line.strip(),i
        if tag1=='-':
          tag1=tags1[tid]
        if tag2=='-':
          tag2=tags2[tid]
        instance=feats_lemma(tag1,tag2)
        """
        instance={
        'tag1':tag1,
        'tag2':tag2,
        'pos1':tag1[0],
        'pos2':tag2[0],
        'pos21':tag1[:2],
        'pos22':tag2[:2],
        #'sametag':tag1=='-'
        }
        if True:
          for feat,value in slfeats.get(tag1,{}).items():
            instance[feat+'1']=value
          for feat,value in slfeats.get(tag2,{}).items():
            instance[feat+'2']=value
        """
        Xlemma.append(instance)
    if tagc1!='' and tagc2!='':
      continue
    triplets.append((slen.get(tag1,tag1),slen.get(tag2,tag2),1 if tagc1==tag1 else 2))
    if tagc1==tag1:
      ytag.append(1)
    elif tagc2==tag2:
      ytag.append(2)
    else:
      print 'error',line.strip(),i
    tidsplit=tid.split('.')
    tidtid=int(tidsplit[-1])
    instance=feats_tag(tag1,tag2)
    """
    instance={
    'tag1':tag1,
    'tag2':tag2,
    'pos1':tag1[0],
    'pos2':tag2[0],
    'pos21':tag1[:2],
    'pos22':tag2[:2],
    #'tag12':tag1+'_'+tag2,
    #'pos12':tag1[0]+tag2[0],
    #'pos212':tag1[:2]+'_'+tag2[:2],
    }
    if True:
      for feat,value in slfeats.get(tag1,{}).items():
        instance[feat]=value
      for feat,value in slfeats.get(tag2,{}).items():
        instance[feat]=value
    if False:
      prev=tags2.get('.'.join(tidsplit[:-1])+'.'+str(tidtid-1))
      next=tags2.get('.'.join(tidsplit[:-1])+'.'+str(tidtid+1))
      if prev is not None:
        instance['sklonprev']=check_agreement(prev,tag2,'sklon')
        instance['spolprev']=check_agreement(prev,tag2,'spol')
        instance['steviloprev']=check_agreement(prev,tag2,'število')      
        #print instance['sklonprev'],prev,tag1
      if next is not None:
        instance['sklonnext']=check_agreement(next,tag2,'sklon')
        instance['spolnext']=check_agreement(next,tag2,'spol')
        instance['stevilonext']=check_agreement(next,tag2,'število')      
    if False:
      prev=tags2.get('.'.join(tidsplit[:-1])+'.'+str(tidtid-1))
      prev2=tags2.get('.'.join(tidsplit[:-1])+'.'+str(tidtid-2))
      if prev is not None:
        instance['prev']=prev[0]+'_'+tag2
      if prev2 is not None:
        instance['prev2']=prev2[0]+'_'+tag2
    """
    Xtag.append(instance)
    Xtag_str.append((token,tag1,tag2))
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC,LinearSVC
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.linear_model import SGDClassifier
  #from sklearn.neural_network import MLPClassifier
  from sklearn.dummy import DummyClassifier
  from sklearn.model_selection import cross_val_predict
  #from sklearn.cross_validation import cross_val_predict
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
  from sklearn.preprocessing import OneHotEncoder,LabelEncoder
  from sklearn.feature_extraction import DictVectorizer
  from sklearn.pipeline import Pipeline
  for X,y,out in ((Xlemma,ylemma,'lemma'),(Xtag,ytag,'tag')):
    #dv=DictVectorizer(sparse=False)
    #X_trans=dv.fit_transform(X)
    #print 'no_features',len(X[0])
    print '###',out,'###'
    dclf=Pipeline([('vect',DictVectorizer(sparse=False)),('clf',DummyClassifier(strategy='most_frequent'))])
    #clf=RandomForestClassifier(n_estimators=20)
    #clf=MultinomialNB()
    clf=Pipeline([('vect',DictVectorizer(sparse=False)),('clf',LinearSVC(C=0.1))])
    gclf=Pipeline([('vect',DictVectorizer(sparse=False)),('clf',SVC())])
    #clf=MLPClassifier(hidden_layer_sizes=(100,100))
    #clf=SGDClassifier()
    #clf=SVC()
    gsrch=GridSearchCV(gclf,[{'clf__C': [0.1, 1, 10, 100, 1000], 'clf__gamma': [0.01, 0.001, 0.0001]}],n_jobs=20)
    gsrch.fit(X,y)
    print 'full grid search'
    print gsrch.best_score_
    best=gsrch.best_estimator_
    best.fit(X,y)
    from sklearn.externals import joblib
    joblib.dump(best,out+'.grid.model')
    y_pred=cross_val_predict(clf,X,y,cv=10)
    print 'full'
    print classification_report(y,y_pred)
    print accuracy_score(y,y_pred)
    print confusion_matrix(y,y_pred)
    y_dummy=cross_val_predict(dclf,X,y)
    print 'dummy'
    print classification_report(y,y_dummy)
    print accuracy_score(y,y_dummy)
    print 'first'
    print classification_report(y,[1]*len(y))
    print accuracy_score(y,[1]*len(y))
    print 'second'
    print classification_report(y,[2]*len(y))
    print accuracy_score(y,[2]*len(y))
    clf.fit(X,y)
    joblib.dump(clf,out+'.model')
  corrs={}
  for confusion,pred,true in zip(triplets,y_pred,y):
    if confusion not in corrs:
      corrs[confusion]=[0.,0.]
    if pred==true:
      corrs[confusion][0]+=1
    else:
      corrs[confusion][1]+=1
  ratios={}
  for triple,freqs in corrs.items():
    ratios[triple]=(freqs[0]+1)/(sum(freqs)+2)
  #print 'ratios distr'
  triples2=set()
  for triple,freq in sorted(ratios.items(),key=lambda x:-x[1])[:100]:
    if triple in triples2:
      continue
    triple2=(triple[0],triple[1],2 if triple[2]==1 else 1)
    triples2.add(triple2)
    #print triple,corrs.get(triple),triple2,corrs.get(triple2,[None,None])
    #(triple[0],triple[1],2 if triple[2]==1 else 1),ratios.get((triple[0],triple[1],2 if triple[2]==1 else 1),0.0),errs.get((triple[0],triple[1],2 if triple[2]==1 else 1),0)
