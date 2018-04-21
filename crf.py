from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import random
from sklearn import metrics
import json 
import numpy as np

data_dir = "../data/"
out_dir = "../models/"
dis_dir = data_dir + "disease_list/web-scrapers/medical_ner/"


trigram_disease_set = set(["phr","sia","sys","age","nia","ites","une","ial"])
disease_next_word_set = set(["syndrome","disease","deficiency"])
treatment_next_word_set = set(["therapy","test"])


disease_set = set()
treatment_set = set()
datastore1 = json.loads(open(dis_dir + "disease1.json").read())
datastore2 = json.loads(open(dis_dir + "disease2.json").read())
datastore3 = json.loads(open(dis_dir + "treatment.json").read())

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



# print (len(disease_set))
# exit(0)


def word2features(sent, i):
    word = sent[i][0]
    pos = sent[i][2] 
    # com_trigram = 0
    # for j in range(len(word)-3):
    # 	trigram = word[j:j+3]
    # 	if trigram in trigram_disease_set:
    # 		com_trigram = 1
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
        # 'trigram' : com_trigram
    }
    if i > 0:
        word1 = sent[i-1][0]
        pos = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            # '-1:pos': pos,
        	'-1:word[-3:]': word1[-3:],
        	'-1:word[:3]' : word1[:3],
        	# '-1:in_disease' : word1 in disease_set,
        	# '-1:word[-2:]': word1[-2:],


        })
    else:
        features['BOS1'] = True
    # if i > 1:
    #     word1 = sent[i-2][0]
    #     pos = sent[i-2][2]
    #     features.update({
    #         '-2:word.lower()': word1.lower(),
    #         # '-2:pos': pos,

    #     })
    # else:
    #     features['BOS2'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        pos = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            # '+1:pos': pos,
        	'+1:word[-3:]': word1[-3:],
        	'+1:word[:3]' : word1[:3],
        	# '+1:in_disease' : word1 in disease_set,

        	# '+1:word[-2:]': word1[-2:],

            # '+1:tion' : word1=="therapy",
            '+1:dis' : word1 in disease_next_word_set,
            # '+1:treat' : word1 in treatment_next_word_set,


        })
    else:
        features['EOS1'] = True
    # if i < len(sent)-2:
    #     word1 = sent[i+2][0]
    #     pos = sent[i+2][2]
    #     features.update({
    #         '+2:word.lower()': word1.lower(),
    #         '+2:pos': pos,

    #     })
    # else:
    #     features['EOS2'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label,pos in sent]

def sent2tokens(sent):
    return [token for token, label,pos in sent]


def read_data():
	data = []
	filepath = data_dir + "ner.txt"  
	with open(filepath,"r",encoding='utf-8', errors='ignore') as fp:  
   		line = fp.readline()
   		word_tag = []
   		senten = []
   		while line:
   			if len(line.split())==0:
   				pos_tags = nltk.pos_tag(senten)
   				word_pos_tag = []
   				for i in range(len(pos_tags)):
   					word_pos_tag.append((word_tag[i][0],word_tag[i][1],pos_tags[i][1]))
   				data.append(word_pos_tag)
   				word_tag = []
   				senten = []
   			else:
   				senten.append(line.split()[0])
	   			word_tag.append((line.split()[0],line.split()[1]))
   			line = fp.readline()
	return data

all_data = read_data()
tr_test_r = 10
total_data = len(all_data)
train_len = (tr_test_r * total_data) // 100
np.random.seed(3141)
np.random.shuffle(all_data)
# with open(out_dir + 'data.pkl','wb') as f :
    # pickle.dump(all_data,f)
avg_fscore = 0

for i in range(100//tr_test_r):
	test_data = all_data[i*train_len:(i+1)*train_len]
	train_data = all_data[:i*train_len] + all_data[(i+1)*train_len:]
	X_train = [sent2features(s) for s in train_data]
	y_train = [sent2labels(s) for s in train_data]

	X_test = [sent2features(s) for s in test_data]
	y_test = [sent2labels(s) for s in test_data]

	crf = sklearn_crfsuite.CRF(
	    algorithm='lbfgs', 
	    c1=0.1, 
	    c2=0.1, 
	    max_iterations=100,
	    all_possible_transitions=True
	)
	crf.fit(X_train, y_train)
	y_pred = crf.predict(X_test)
	tag_to_ix = {"O": 0, "D": 1, "T": 2}
	y_predicted = []
	for j in range(len(y_pred)):
		for k in range(len(y_pred[j])):
			# if k>0 and k < len(y_pred[j])-1:
			# 	prev_tag = tag_to_ix[y_pred[j][k-1]]
			# 	next_tag = tag_to_ix[y_pred[j][k+1]]
			# 	if prev_tag==next_tag and not prev_tag==0 and not prev_tag==tag_to_ix[y_pred[j][k]]:
			# 		y_predicted.append(prev_tag)
					# print (X_test[j][k]["-1:word.lower()"],X_test[j][k]["word.lower()"],X_test[j][k]["+1:word.lower()"],y_test[j][k],prev_tag)
			# 		continue
			y_predicted.append(tag_to_ix[y_pred[j][k]])
	y_actual = []
	for j in range(len(y_test)):
		for k in range(len(y_test[j])):
			y_actual.append(tag_to_ix[y_test[j][k]])
	print(metrics.classification_report(y_actual, y_predicted))
	print("Macro Accuracy:", metrics.precision_score(y_actual,y_predicted,average='macro'))
	print("Accuracy Score:", metrics.accuracy_score(y_actual,y_predicted))
	print("Macro F1 Score:",metrics.f1_score(y_actual,y_predicted,average='macro'))
	f_score = metrics.f1_score(y_actual, y_predicted, average=None)
	avg_fscore += (f_score[1] + f_score[2])
	print((f_score[1] + f_score[2])/2)
print ("avg",avg_fscore/(200//tr_test_r))