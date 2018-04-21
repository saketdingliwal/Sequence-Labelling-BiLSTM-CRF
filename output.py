from itertools import chain
import nltk
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random
import json 
import numpy as np
import sys
import pickle

disease_next_word_set = set(["syndrome","disease","deficiency"])
treatment_next_word_set = set(["therapy","test"])


disease_set = set()
treatment_set = set()
datastore1 = json.loads(open("disease1.json").read())
datastore2 = json.loads(open("disease2.json").read())
datastore3 = json.loads(open("treatment.json").read())

for i in range(len(datastore1)):
	disease_comp_name = datastore1[i]["disease"].split()
	for part in disease_comp_name:
		if len(part) > 4:
			disease_set.add(part)
for i in range(len(datastore2)):
	disease_comp_name = datastore2[i]["disease"].split()
	for part in disease_comp_name:
		if len(part) > 4:
			disease_set.add(part)
for i in range(len(datastore3)):
	disease_comp_name = datastore3[i]["procedure"].split()
	for part in disease_comp_name:
		if len(part) > 4:
			treatment_set.add(part)


def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][1] 
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.length()': len(word),
        'pos' : pos,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]' : word[:3],
        'word[:2]' : word[:2],
        'word[-4:]': word[-4:],
        'word[:4]' : word[:4],
        'if_disease': word in disease_set,
        'if_treatment': word in treatment_set,
    }
    if i > 0:
        word1 = sent[i-1][0]
        pos = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
        	'-1:word[-3:]': word1[-3:],
        	'-1:word[:3]' : word1[:3],
        })
    else:
        features['BOS1'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        pos = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
        	'+1:word[-3:]': word1[-3:],
        	'+1:word[:3]' : word1[:3],
            '+1:dis' : word1 in disease_next_word_set,
        })
    else:
        features['EOS1'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,pos in sent]


def read_data(filename):
	data = []
	with open(filename,"r",encoding="ISO-8859-2") as fp:  
   		line = fp.readline()
   		senten = []
   		while line:
   			if len(line.split())==0:
   				pos_tags = nltk.pos_tag(senten)
   				data.append(pos_tags)
   				senten = []
   			else:
   				senten.append(line.split()[0])
   			line = fp.readline()
	return data

filename = sys.argv[1]
all_data = read_data(filename)
test_data = all_data

with open("model0.pkl",'rb') as f:
    crf = pickle.load(f)

X_test = [sent2features(s) for s in test_data]
y_pred = crf.predict(X_test)

output_file = sys.argv[2]
file = open(output_file, "w",encoding="ISO-8859-2")
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        file.write(test_data[i][j][0]+" "+y_pred[i][j]+"\n")
    file.write("\n")
