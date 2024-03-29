import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle
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


def to_scalar(var):
    # returns a python 
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
	idxs = []
	for w in seq:
		if w in to_ix:
			idxs.append(to_ix[w])
		else:
			idxs.append(to_ix["<UNCH>"])
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim+1, self.tagset_size)
        # self.char_embeds = nn.Embedding(num_chars, char_embed)        
        # self.lstm_char = nn.LSTM(embedding_dim, shape_dim, num_layers=1)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence,caps):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        num_words = len(caps)
        caps = torch.FloatTensor(caps)
        caps = autograd.Variable(caps.view(num_words,1))
        features_lstm = torch.cat((lstm_out,caps),1)
        lstm_feats = self.hidden2tag(features_lstm)
        return lstm_feats
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence,caps, tags):
    	char_embed = self._get_char_embed(sentence)
        feats = self._get_lstm_features(sentence,caps,char_embed)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence,caps):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence,caps)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 50
HIDDEN_DIM = 50

# Make up some training data
training_data = train_data

word_to_ix = {}
for sentence, tags,caps in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_to_ix["<UNCH>"] = len(word_to_ix)

tag_to_ix = {"O": 0, "D": 1, "T": 2, START_TAG: 3, STOP_TAG: 4}
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
with open(out_dir + 'dict.pkl','wb') as f :
    pickle.dump(word_to_ix,f)


actual_test_labels = []
for i in range(len(test_data)):
	for j in range(len(test_data[i][1])):
		name_tag = test_data[i][1][j]
		actual_test_labels.append(tag_to_ix[name_tag])
# print (len(word_to_ix))
# Check predictions before training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
# print((precheck_sent))
# exit(0)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    iterr = 0
    for sentence, tags,caps in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])

        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in,caps, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()
        iterr+=1
        if iterr%500==0:
        	predicted_labels = []
        	print ('epoch :',epoch, 'iterations :',iterr)
    for i in range(len(test_data)):
        _,tags = model(prepare_sequence(test_data[i][0],word_to_ix),test_data[i][2])
        predicted_labels =  predicted_labels + tags
    f_scores = f1_score(actual_test_labels, predicted_labels, average=None)
    f_sc = (f_scores[1]+f_scores[2])/2
    print ("epochs->",epoch,"f scores->",f_sc)
    torch.save(model.state_dict(), out_dir+'simple_cap' + str(f_sc)+ "__" + str(epoch+1)+'.pth')
    # predicted_labels = []
    # for i in range(len(test_data)):
    # 	_,tags = model(prepare_sequence(test_data[i][0],word_to_ix))
    # 	predicted_labels = tags + predicted_labels
    # f_scores = f1_score(actual_test_labels, predicted_labels, average=None)
    # print ("epochs->",epoch,"f scores->",(f_scores[1]+f_scores[2])/2)




# Check predictions after training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# print(model(precheck_sent))


