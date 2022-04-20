#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import warnings
import numpy as np
import pandas as pd
import pickle

from nltk import ngrams
from sklearn.model_selection import train_test_split


# In[2]:



url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
hashtag_regex = r"#(\w+)"
mention_regex = r"\B@(?!(?:[a-z0-9.]*_){2})(?!(?:[a-z0-9_]*\.){2})[._a-z0-9]{3,24}\b"
start = ['<pad>','<pad>','<pad>','<pad>']
# end = np.array(['<pad>','<pad>','<pad>'])
punc = [',', '.', '!', ';', '?']
def tokenizer(messages):
    tokenized_data = []
#     for messages in data:
    messages = messages.lower()
#         messages = re.sub(url_regex,'<URL>',messages)
#         messages = re.sub(hashtag_regex,'<HASHTAG>',messages)
#         messages = re.sub(mention_regex, '<MENTION>',messages)
#         newmessage = ''
#         for i in range(len(messages)):
#             if re.match(r'[^\w\s]', messages[i]) and messages[i]!='<' and messages[i]!='>':
#                 newmessage+=' '+messages[i]+' '
#             else:
#                 newmessage+=messages[i]

#         if newmessage == '':
#             continue
    messages = re.sub(r'[^\w\s]','',messages)
    words = re.split('\s|\t|\n', messages)
#         words = words[words!='']
#         if len(words)==0:
#             continue
#         m = len(words)
#         for i  in range(m):
#             if len(words[i])!=1:
#                 continue
#             if words[i] in punc and words[i] not in ['s','e','<URL>','<HASHTAG>','<MENTION>']:
#                 words[i] = re.sub(r'[^\w\s]','',words[i])
#         words = words[words!='']
#         for wo in words:
#             tokenized_data.append(wo)
        #tokenized_data.append(np.concatenate((start, words, end)))
    return start+words


# In[3]:


def read_corpus(path):
    df = []
    si = 0
    tot = 0
    with open(path) as infile:
        for ob in infile:
            flag = rand.randint(0,100)
            tot+=1
            if flag>2:
                continue
            si+=1
            # if si%10000 == 0:
            #     print(si,tot)
            obj = json.loads(ob)
            cur = obj['reviewText']
            df.append(tokenizer(cur))
    return df


# In[4]:


def read_text(path):
    """ Function to read input data
    Args:
        path (string): the parent path of the folder containing the input text files
    Returns:
        string: The complete text read from input files appended in a single string.
    """
    text = ' '
    f = open(path, 'r')
    text = f.readlines()
    return text


def preprocess_text(text):
    """ Function for basic cleaning and pre-processing of input text
    Args:
        text (string): raw input text
    Returns:
        string: cleaned text
    """
    data = []
    for line in text:
        data += tokenizer(line)
    return data
#     text = text.lower()
#     text = re.sub(r"'s\b", "", text)
#     text = re.sub("[^a-zA-Z]", " ", text)
#     text = ' '.join([word for word in text.split() if len(word) >= 3]).strip()

#     return text


def prepare_text(data, n):
    """ Function to prepare text in sequence of ngrams
    Args:
        text (string): complete input text
    Returns:
        list : a list of text sequence with 31 characters each
    """
    n_grams = []
    for i in range(len(data)-n):
        n_grams.append(data[i:i+n])
    return n_grams

def create_data(data, word_id, id_word):
    """ Function to encode the character sequence into number sequence
    Args:
        text (string): cleaned text
        sequence (list): character sequence list
    Returns:
        dict: dictionary mapping of all unique input charcters to integers
        list: number encoded charachter sequences
    """
    x = []
    y = []
    for i in data:
        inp = []
        for j in range(4):
            if i[j] not in word_id.keys():
                inp.append(word_id['<UNKNOWN>'])
            else:
                inp.append(word_id[i[j]])
        if i[4] not in word_id.keys():
            y.append(word_id['<UNKNOWN>'])
        else:
            y.append(word_id[i[4]])
        x.append(inp)
    x = np.array(x)
    y = np.array(y)
    return x,y
    
    


def split_data(mapping, encoded_sequence):
    """ Function to split the prepared data in train and test
    Args:
        mapping (dict): dictionary mapping of all unique input charcters to integers
        encoded_sequence (list): number encoded charachter sequences
    Returns:
        numpy array : train and test split numpy arrays
    """

    encoded_sequence_ = np.array(encoded_sequence)
    X, y = encoded_sequence_[:, :-1], encoded_sequence_[:, -1]
    y = to_categorical(y, num_classes=len(mapping))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def create_vocab(data, threshold =  5):
    freq = {}
    for i in data:
        if i not in freq.keys():
            freq[i] = 0
        freq[i]+=1
    for i in range(len(data)):
        if freq[data[i]] <= threshold:
            data[i] = '<UNKNOWN>'
    word_id = {}
    id_word = {}
    cur = 0
    for i in data:
        if i not in word_id.keys():
            word_id[i] = cur
            id_word[cur] = i
            cur+=1
    return data, word_id, id_word


# In[5]:


text = read_text('./dataset/europarl-corpus/train.europarl')


# In[6]:


data = preprocess_text(text)


# In[7]:


data, word_id, id_word = create_vocab(data)


# In[ ]:


n_grams = prepare_text(data, 5)


# In[ ]:


x, y = create_data(n_grams, word_id, id_word)


# In[8]:


vocab_size = len(list(word_id.keys()))
embedding_size = 50


# In[9]:


# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np 

#get_ipython().run_line_magic('matplotlib', 'inline')

# for reproducibility
torch.manual_seed(1)


# In[10]:


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.lstm_layers = 4
        self.batch_size = 128
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers=self.lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(4*embedding_size , vocab_size)
    
    def forward(self, x):
        batch_size = len(x)
        hidden = (torch.randn(self.lstm_layers, batch_size, embedding_size).to(device),torch.randn(self.lstm_layers, batch_size, embedding_size).to(device))
        x = x.int()
        y = self.embedding(x)
        out, hidden = self.lstm(y, hidden)
        out = out.contiguous()
        out = out.view(batch_size,-1)
        y = self.fc1(out)
        y = F.softmax(y, dim = 1)
        return y


# In[11]:


def trainvalAdam(model, train_data, valid_data, device, batch_size=128, num_iters=1, learn_rate=0.01):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle after every epoch
    # val_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.90, 0.98), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = optim.Adadelta(model.parameters(),lr=1.0, rho=0.95, eps=1e-08, weight_decay=0.0)
    iters, losses, val_losses, train_acc, val_acc = [], [], [], [], []
    # training
    si = len(train_data)
    n = 0 # the number of iterations
    for n in tqdm(range(num_iters)):
        tot_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # optimizer.zero_grad()         # a clean up step for PyTorch
            out = model(imgs.float())             # forward pass
            loss = criterion(out, labels) # compute the total loss
#             print(loss)
            tot_loss += loss
            optimizer.zero_grad()
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
        print('epoch '+str(n)+' = '+str(tot_loss/si))
        losses.append(tot_loss/si)   
#             if n % 10 == 9:
#                 iters.append(n)
#                 losses.append(float(loss)/batch_size)        # compute *average* loss
# #                 train_accuracy = get_accuracy(model, train_data, device)
# #                 val_accuracy = get_accuracy(model, valid_data, device)
# #                 print(train_accuracy,val_accuracy)
#                 for im, lb in val_loader:
#                     im, lb = im.to(device), lb.to(device)
#                     val_out = model(im.float())
#                     val_loss = criterion(val_out, lb.float())
#                 val_losses.append(float(val_loss)/batch_size)
                
                
#                 train_acc.append(train_accuracy) # compute training accuracy 
#                 val_acc.append(val_accuracy)   # compute validation accuracy
    return losses


# In[ ]:


train_data = []
for i in range(len(x)):
  if y[i]==word_id['<pad>']:
    continue
  train_data.append((np.array(x[i]),y[i]))


# In[ ]:


# model = Classifier()
# print(model)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# print(device)
# model.to(device)
# train_losses = trainvalAdam(model, train_data, train_data, device, num_iters=20)


# In[ ]:


# torch.save(model.state_dict(),'./model-4')


# In[13]:


model = Classifier()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load('./models/model-4', map_location=torch.device(device)))


# In[20]:


def perplexity(train_data):
  acc = 0.0
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False) # shuffle after every epoch
  probab = []
  with torch.no_grad():
    model.eval()
    pro = 1.0
    le = 0
    for imgs, labels in train_loader:
        if imgs[0][-1] == word_id['<pad>']:
#         print('Hello ',pro,le)
            if pro==0:
                pro = np.random.uniform(1e-9,1e-8)
            per = (1/(pro+1e-10))**(1/(le+4))
            probab.append(per)
            pro = 1.0
            le = 0
        out = model(imgs.float().to(device))
        pro*=out[0][labels[0]]
        le+=1
  if pro==0:
      pro = np.random.uniform(1e-9,1e-8)
  per = (1/(pro+1e-10))**(1/(le+4))
  probab.append(per)
  return probab

# per = perplexity(train_data)


# In[ ]:


# def avgPer(per):
#     sum = 0
#     for i in per:
#         sum += float(i)
#     sum/=len(per)
#     return sum

# avg_train_per = avgPer(per)
# print(avg_train_per)


# In[ ]:


# f = open('train-europarl-lm.txt','w')
# f.write('Average perplexity = '+str(avg_train_per)+'\n')
# for i in range(len(text)):
#     f.write(text[i]+'     '+str(float(per[i]))+'\n')


# In[22]:


# f = open('test-europarl-lm.txt','w')
# f.write('Average perplexity = '+str(avg_train_per)+'\n')
# for i in range(len(text_test)):
#     f.write(text_test[i]+'     '+str(float(per[i]))+'\n')


# In[ ]:


sent = input("Enter the sentence : \n")
text_test = [sent]
data_test = preprocess_text(text_test)
n_grams_test = prepare_text(data_test, 5)
test_x, test_y = create_data(n_grams_test, word_id, id_word)
test_data = []
for i in range(len(test_x)):
  test_data.append((np.array(test_x[i]),test_y[i]))

per = perplexity(test_data)
def avgPer(per):
    sum = 0
    for i in per:
        sum += float(i)
    sum/=len(per)
    return sum

avg_train_per = avgPer(per)
print('The perplexity is : ' , avg_train_per)

