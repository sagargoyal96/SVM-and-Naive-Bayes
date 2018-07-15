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
# print(vocab_size)

Theta=np.ones((10,vocab_size),dtype=np.double)
# words_in_class=np.zeros((1,10),dtype=np.int)
P_ofy=np.zeros((1,10), dtype=np.double)

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


with open('nb_theta_bigram_pos.pkl','rb') as f:
	Theta=pickle.load(f)
with open('P_ofy_q1e.pkl','rb') as f:
	P_ofy=pickle.load(f)

print("Entering testing phase")

def testing(input_file, output_file):
	clean_and_enumerate(input_file,'final_modified_test.txt')
	print("done making modified file")

	file_x=open('final_modified_test.txt','r')
	file_y=open(output_file,'w')
	accuracy=0
	total_size=0

	testfile=file_x.read().splitlines()

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
		file_y.write(str(maxind[0]+1))
		file_y.write('\n')
		j+=1


testing(sys.argv[1] , sys.argv[2])
















