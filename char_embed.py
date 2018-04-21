import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score

torch.manual_seed(1)

data_dir = "/home/cse/btech/cs1150254/Desktop/A2/data/"
out_dir = "/home/cse/btech/cs1150254/Desktop/A2/models/"


def read_data():
	data = []
	filepath = data_dir + "ner.txt"  
	with open(filepath,"r",encoding='utf-8', errors='ignore') as fp:  
   		line = fp.readline()
   		sentence = []
   		tags = []
   		caps = []
   		while line:
   			if len(line.split())==0:
   				data.append((sentence,tags,caps))
   				sentence = []
   				tags = []
   				caps = []
   			else:
	   			sentence.append(line.split()[0])
	   			if line.split()[0][0].isupper() and not len(caps)==0:
	   				caps.append(1)
	   			else:
	   				caps.append(0)
	   			tags.append(line.split()[1])
   			line = fp.readline()
	return data

all_data = read_data()
tr_test_r = 85
total_data = len(all_data)
train_len = tr_test_r * total_data // 100
train_data = all_data[0:train_len]
test_data = all_data[train_len:]

class LSTM_MODEL(torch.nn.Module) :
   def __init__(self,charsize,embedding_dim,hidden_dim,num_classes):
      super(LSTM_MODEL,self).__init__()
      self.hidden_dim = hidden_dim
      self.embeddings = nn.Embedding(charsize, embedding_dim)
      self.lstm = nn.LSTM(embedding_dim,hidden_dim)
      self.linearOut = nn.Linear(hidden_dim,num_classes)
   def forward(self,inputs):
      x = self.embeddings(inputs).view(len(inputs),1,-1)
      hidden = self.init_hidden()
      lstm_out,lstm_h = self.lstm(x,hidden)
      x = lstm_out[-1]
      x = self.linearOut(x)
      x = F.log_softmax(x)
      return x
   def init_hidden(self):
      h0 = Variable(torch.zeros(1,1,self.hidden_dim).cuda())
      c0 = Variable(torch.zeros(1,1,self.hidden_dim).cuda())
      return (h0,c0)


char_dict = {}
for sentence, tags,caps in train_data:
   for word in sentence:
      for char in word:
         if char not in char_dict:
            char_dict[char] = len(char_dict)
char_dict["<UNCH>"] = len(char_dict)


def get_sequence(word,char_dict):
   word_vect = []
   for char in word:
      if char in char_dict:
         word_vect.append(char_dict[char])
      else:
         word_vect.append(char_dict["<UNCH>"])
   return word_vect

with open(out_dir + 'char_dict.pkl','wb') as f :
   pickle.dump(char_dict,f)

emebed_dim = 2
hidden_dim = 20
char_size = len(char_dict)
model = LSTM_MODEL(char_size,emebed_dim,hidden_dim,3)
model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 10
tag_to_ix = {"O": 0, "D": 1, "T": 2}

for i in range(epochs):
   loss_sum = 0
   total_acc = 0.0
   for sentence, tags,caps in train_data:
      for iterr in range(len(sentence)):
         word = sentence[iterr]
         tag = tags[iterr]
         target_data = [tag_to_ix[tag]]
         rand_no = random.randint(1,100)
         if tag_to_ix[tag]==0 and rand_no < 50:
            continue
         target_data = Variable(torch.LongTensor(target_data).cuda())
         input_data = get_sequence(word,char_dict)
         input_data = Variable(torch.LongTensor(input_data).cuda())
         class_pred = model(input_data)
         model.zero_grad()
         loss = loss_function(class_pred,target_data)
         loss_sum += loss.data[0]
         loss.backward()
         optimizer.step()
   predicted = []
   actual_label = []
   for j in range(len(test_data)):
      for k in range(len(test_data[i][0])):
         test_input = get_sequence(test_data[i][0][k],char_dict)
         test_input = Variable(torch.LongTensor(test_input).cuda())
         pred = model(test_input)
         _, pred = torch.max(pred.data, 1)
         predicted.append(int(pred))
         actual_label.append(tag_to_ix[test_data[i][1][k]])
   f_scores = f1_score(actual_label, predicted, average=None)
   f_sc = (f_scores[1]+f_scores[2])/2
   print (f_scores)
   print ("epochs->",i,"f scores->",f_sc)
   torch.save(model.state_dict(), out_dir+'char_embed_sec_try' + str(f_sc)+ "__" + str(i+1)+'.pth')