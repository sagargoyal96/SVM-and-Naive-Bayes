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
	myfile=open('saved_vocab_pure.txt', 'r')
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

Theta=np.ones((10,vocab_size),dtype=np.double)
words_in_class=np.zeros((1,10),dtype=np.int)
P_ofy=np.zeros((1,10), dtype=np.double)

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


# make p of y ka kuch karna hai

with open('theta_q1a.pkl','rb') as f:
	Theta=pickle.load(f)
with open('P_ofy_q1a.pkl','rb') as f:
	P_ofy=pickle.load(f)

print("Entering testing phase")

def testing(input_file, output_file):
	clean_and_enumerate(input_file,'modified_test_data_q1a.txt')
	print("done making modified file")

	file_x=open('modified_test_data_q1a.txt','r')
	file_y=open(output_file,'w')
	total_size=0

	testfile=file_x.read().splitlines()
	# testlabel=file_y.read().splitlines()
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
		file_y.write(str(maxind[0]+1))
		file_y.write('\n')




testing(sys.argv[1], sys.argv[2])











