import sys
# import nltk
import os
import string
import nltk
import io
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import simplejson
import csv
import math
from decimal import Decimal
import random
import pickle

def make_bigramvocab(filename):
	file_x=open(filename,"r")
	i=0
	while(1):
		mystring=file_x.readline()
		if mystring=="":
			break
		newstr=re.findall(r'\w+', mystring)
		# print(newstr)
		for i in range(len(newstr)-1):
			vocab.add((newstr[i],newstr[i+1]))

	# size=len(vocab)
	# vocab_count=np.zeros((1,size),dtype=np.int)
	# i=0
	# for item in vocab:




vocab=set()

make_bigramvocab('newtrain_text.txt')




def save_vocab():
	f = open('savebigram_vocab.txt', 'w')
	for item in vocab:
		# print(item)
		f.write(item[0])
		f.write(" ")
		f.write(item[1])
		f.write('\n')
	f.close()

save_vocab()


# # --------------turns the saved vocab into a dict readable format

def read_vocab():
	data=[]
	myfile=open('savebigram_vocab.txt', 'r')
	i=0
	while(1):
		temp=myfile.readline().replace('\n', '')
		if temp=="":
			break
		data.append((temp,i))
		i+=1
	return data

vocab_list=dict(read_vocab())
vocab_size=len(vocab_list)
print(vocab_size)
# print(vocab_list)
# print(vocab_list)


# # this saves the data file with words replaced by their vocab indices after removing the unneccesary characters


# # ----------------------------------------------------------------------------------------------------------------------------------------

# # this function calculates the Theta and the total words in a class and then save_trained can be used to save them to file

# # initialise words_in _class, neede only when new theta needs to be learnt, laplace smoothing

# # ----------------------------------------------------------------------------------------------------------------------------------

POS_TAG_MULTIPLIER=3


def clean_and_enumerate_pos(readfrom, writeto):
	file_y=open(writeto,'w')
	file_x=open(readfrom,'r')
	# print(vocab_list)

	i=0
	while(1):
		mystring=file_x.readline()
		if mystring=="":
			break
		newstr=re.findall(r'\w+', mystring)

		tagged_words=nltk.pos_tag(newstr)


		for i in range(len(tagged_words)-1):
			if tagged_words[i][1].startswith('JJ') or tagged_words[i+1][1].startswith('JJ'):
				for appender in range(POS_TAG_MULTIPLIER):
					# print("worddd= ", word)
					newstr.append(newstr[i])
					newstr.append(newstr[i+1])
		# print(len(newstr))
		# print((newstr[0],newstr[1]))
		# print(vocab_list.keys())

		# print(vocab_list.get(('love','movi')))

		for i in range(len(newstr)-1):
			mytr=newstr[i]+" "+ newstr[i+1]
			if mytr in vocab_list.keys():
				# print("yes")
				file_y.write("%s " % vocab_list.get(mytr))
				# else:
				# 	print("no ")
		
		file_y.write("\n")

	file_y.close()
	file_x.close()

def clean_and_enumerate(readfrom, writeto):
	file_y=open(writeto,'w')
	file_x=open(readfrom,'r')
	# print(vocab_list)

	i=0
	while(1):
		mystring=file_x.readline()
		if mystring=="":
			break
		newstr=re.findall(r'\w+', mystring)


		for i in range(len(newstr)-1):
			mytr=newstr[i]+" "+ newstr[i+1]
			if mytr in vocab_list.keys():
				# print("yes")
				file_y.write("%s " % vocab_list.get(mytr))
				# else:
				# 	print("no ")
		
		file_y.write("\n")

	file_y.close()
	file_x.close()


clean_and_enumerate('newtrain_text.txt', 'newmodified.txt' )

words_in_class=np.zeros((1,10),dtype=np.int)
for k in range(0,10):
	words_in_class[0,k]=vocab_size

Theta=np.ones((10,vocab_size),dtype=np.double)
P_ofy=np.zeros((1,10), dtype=np.double)

# # this function calculates the Theta and the total words in a class and then save_trained can be used to save them to file


def training():

	file_y=open('imdb_train_labels.txt','r')
	file_x=open('newmodified_data.txt','r')

	labels=file_y.read().splitlines()
	# print(labels)
	i=0
	gg=[]

	while(1):

		text=file_x.readline().split()
		if text==[]:
			break
		# matrix update	
		for item in text:
			lb=int(labels[i])
	# 		ind=word_index.index(item+"\n")
			it=int(item)
			Theta[lb-1, it]+=1.0
			words_in_class[0,lb-1]+=1
			# print(Theta[lb-1,it])
	# 			# print(lb, ind)


		i+=1


	for i in range(10):
		for j in range(vocab_size):
			Theta[i,j]/=words_in_class[0,i]

training()

with open('trained_models/nb_theta_bigram_pos.pkl', 'wb') as f:
    pickle.dump(Theta, f)



def make_PofY():
	ff=open('imdb_train_labels.txt', 'r')
	for_py=ff.read().splitlines()

	for item in for_py:
		P_ofy[0,int(item)-1]+=1/25000

make_PofY()

with open('trained_models/P_ofy_q1e.pkl', 'wb') as f:
    pickle.dump(P_ofy, f)

with open('trained_models/nb_theta_bigram_pos.pkl', 'rb') as f:
    Theta=pickle.load(f)


def testing():
	clean_and_enumerate('newtest_text.txt','newmodified_test.txt')


	file_x=open('newmodified_test.txt','r')
	file_y=open('imdb_test_labels.txt','r')
	accuracy=0
	total_size=0

	testfile=file_x.read().splitlines()
	testlabel=file_y.read().splitlines()
	j=0
	for item in testfile:
		mylist=item.split()
		probarray=np.zeros((1,10),dtype=np.double)
		for i in range(10):
			for obj in mylist:
				# print("obj= ",obj)
				probarray[0,i]+=(math.log(Decimal(Theta[int(i)][int(obj)])))
			
			if P_ofy[0,i] !=0.0:
				probarray[0,i]+=(math.log(P_ofy[0,i]))
			else:
				probarray[0,i]+=-100000000

		maxind=np.argmax(probarray, axis=1)

		if int(testlabel[j])-1==maxind:
			# print(maxind+1)
			accuracy+=1
		j+=1

	return accuracy/j

acc=testing()
print(acc)






































