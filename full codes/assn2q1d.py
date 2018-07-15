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

def make_vocab(filename):
	file_x=open(filename,"r")
	i=0
	while(1):
		mystring=file_x.readline()
		if mystring=="":
			break
		newstr=re.findall(r'\w+', mystring)
		# print(newstr)
		for item in newstr:
			if len(item)>=2 and item!='br':
				vocab.add(item.lower())

vocab=set()


make_vocab('stemmed_train_text.txt')


def save_vocab():
	f = open('saved_vocab_stemmed.txt', 'w')
	for item in vocab:
		f.write("%s\n" % item)
	f.close()

save_vocab()


# --------------turns the saved vocab into a dict readable format

def read_vocab():
	data=[]
	myfile=open('saved_vocab_stemmed.txt', 'r')
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

# initialisation

Theta=np.ones((10,vocab_size),dtype=np.double)
words_in_class=np.zeros((1,10),dtype=np.int)
P_ofy=np.zeros((1,10), dtype=np.double)


# this saves the data file with words replaced by their vocab indices after removing the unneccesary characters


# ----------------------------------------------------------------------------------------------------------------------------------------

def clean_and_enumerate(readfrom, writeto):
	file_x=open(readfrom,"r")
	file_y=open(writeto,"w")

	i=0
	while(1):
		mystring=file_x.readline()
		if mystring=="":
			break
		newstr=re.findall(r'\w+', mystring)
	
		for item in newstr:
			if len(item)>=2 and item!='br':
				if item.lower() in vocab_list:
					file_y.write("%s " % vocab_list.get(item.lower()))
		
		file_y.write("\n")

	file_y.close()
	file_x.close()


clean_and_enumerate('stemmed_train_text.txt', 'stemmed_modified_text.txt' )



# initialise words_in _class, neede only when new theta needs to be learnt, laplace smoothing

# ----------------------------------------------------------------------------------------------------------------------------------
for k in range(0,10):
	words_in_class[0,k]=vocab_size

# this function calculates the Theta and the total words in a class and then save_trained can be used to save them to file

def training():
	file_y=open('imdb_train_labels.txt','r')
	file_x=open('stemmed_modified_text.txt','r')

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

with open('theta_q1d.pkl', 'wb') as f:
	pickle.dump(Theta,f)


def make_PofY():
	ff=open('imdb_train_labels.txt', 'r')
	for_py=ff.read().splitlines()

	for item in for_py:
		P_ofy[0,int(item)-1]+=1/25000


make_PofY()

with open('P_ofy_q1d.pkl','wb') as f:
	pickle.dump(P_ofy,f)


# with open('theta_q1d.pkl','rb') as f:
# 	Theta=pickle.load(f)

# with open('P_ofy_q1d.pkl','rb') as f:
# 	P_ofy=pickle.load(f)


# final testing on data is done by this function

def testing():
	clean_and_enumerate('stemmed_test_text.txt','stemmed_modified_test.txt')

	file_x=open('stemmed_modified_test.txt','r')
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
		# print("probaray=  ",probarray)
		# print("maxind= ",maxind)
		if int(testlabel[j])-1==maxind:
			# print(maxind+1)
			accuracy+=1
		j+=1

	return accuracy/j


def test_random():
	file_y=open('imdb_test_labels.txt','r')
	testlabel=file_y.read().splitlines()
	accuracy=0
	counter=0
	for item in testlabel:
		rand=random.randint(1, 10)
		if int(item)==rand:
			accuracy+=1
		counter+=1
	return accuracy/counter

def test_maxoccur():
	make_PofY()
	indd=np.argmax(P_ofy,axis=1)

	file_y=open('imdb_test_labels.txt','r')
	testlabel=file_y.read().splitlines()
	accuracy=0
	counter=0
	for item in testlabel:
		if int(item)==indd+1:
			accuracy+=1
		counter+=1
	return accuracy/counter

def make_conf_matrix():
	make_PofY()
	file_x=open('modified_test_data.txt','r')
	file_y=open('imdb_test_labels.txt','r')
	conf_matrix=np.zeros((10,10), dtype=np.int)

	# x-axis is actual, y-axis is predicted

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
		# print("probaray=  ",probarray)
		# print("maxind= ",maxind)
		conf_matrix[int(maxind), int(testlabel[j])-1]+=1
		j+=1
	print(conf_matrix)

	# return accuracy	


acc=testing()
print(acc)

# clean_and_enumerate('newtrain_text.txt','newmodified.txt')








































