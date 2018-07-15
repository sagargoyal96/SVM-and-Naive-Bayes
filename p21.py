import sys
# import nltk
import os
import string
import nltk
import io
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import simplejson
import csv
import math
from decimal import Decimal
import random
import pickle


f=open('svm_thetas_all.pkl', 'rb')
Thetas_all=pickle.load(f)
f.close()

f_=open('svm_b_all.pkl', 'rb')
b_all=pickle.load(f_)
f.close()

def read_testing_data(input_file):
	with open(input_file, 'r') as f:
		reader = csv.reader(f)
		my_list = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
		X_test=[]
		Y_test=[]

		for item in my_list:
			# if item[784]==class1 or item[784]==class2:
				temp=[]
				for i in range(785):
					if i==784:
					 # and item[i]==class1:
						Y_test.append(item[i])
					# elif i==784 and item[i]==class2:
					# 	Y_train.append(1)
					else:
						temp.append(item[i]/255)
				X_test.append(temp)

		# print(X_train[0],X_train[1], Y_train[1], Y_train[2])
		return X_test, Y_test
		
		# return X_train, Y_train


X_tes,Y_tes=read_testing_data(sys.argv[1])
X_test=np.array(X_tes)
Y_test=np.array(Y_tes)

def testing_data(output_file):

	itemcount=0
	accuracy=0
	file_y=open(output_file,'w')
	for item in X_test:
		count=0
		probs=np.zeros((1,10), dtype=np.int)
		for i in range(10):
			for j in range(i+1, 10):
				WtX=np.matmul(Thetas_all[count],np.transpose(item))
				if (WtX+b_all[count])<0:
					probs[0][int(i)]+=1
				else:
					probs[0][int(j)]+=1
				count+=1

				# print(i," ",j)
		index=0
		for k in range(10):
			if probs[0][k]>=probs[0][index]:
				index=k

		file_y.write(str(index[0]))
		file_y.write('\n')

	file_y.close()

		# if Y_test[itemcount]==index:
		# 	accuracy+=1
		# itemcount+=1

	# print(accuracy/itemcount)


testing_data(sys.argv[2])









