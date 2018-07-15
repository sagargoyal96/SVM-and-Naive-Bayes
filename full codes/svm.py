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

BATCH_SIZE=100
ITERATIONS=10000

def read_training_data():
	with open('mnist/train.csv', 'r') as f:
		reader = csv.reader(f)
		my_list = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
		X_train=[]
		Y_train=[]

		for item in my_list:
			# if item[784]==class1 or item[784]==class2:
				temp=[]
				for i in range(785):
					if i==784:
					 # and item[i]==class1:
						Y_train.append(item[i])
					# elif i==784 and item[i]==class2:
					# 	Y_train.append(1)
					else:
						temp.append(item[i]/255)
				X_train.append(temp)

		# print(X_train[0],X_train[1], Y_train[1], Y_train[2])

		return X_train, Y_train
		# Theta = list(list(map(int,rec) for rec in csv.reader(f, delimiter=',')))


X_trai, Y_trai=read_training_data()
X_train=np.array(X_trai)
Y_train=np.array(Y_trai)

def read_specific_data(class1, class2):
	modY=[]
	modX=[]
	k=0
	for item in Y_train:
		if item==class1 or item==class2:
			modX.append(X_train[k])
			if item==class1:
				modY.append(-1)
			else:
				modY.append(1)

		k+=1
	return modX, modY

def training(data, lamb, Y_train, C):
	Theta = np.zeros((1,784), dtype=np.float)
	b=0
	
	for i in range(ITERATIONS):
		n_=1/(lamb*(i+1))

		Sigma=np.zeros((1,784), dtype=np.float)
		B=0

		for j in range(BATCH_SIZE):
			rand=random.randint(0, len(data)-1)
			WtX=np.matmul(Theta,np.transpose(data[rand]))
			if Y_train[rand]*(WtX+b)<1:
				Sigma[0]+=(n_*Y_train[rand]/BATCH_SIZE)*np.array(data[rand])
				B+=(n_*Y_train[rand]/BATCH_SIZE)

		

		for k in range(784):
			Theta[0][k]=(1-1/(i+1))*Theta[0][k] + C*Sigma[0][k]
		b=(1-1/(i+1))*b + B
	
	return Theta, b	

def find_all_thetas():
	Thetas_all=[]
	b_all=[]

	for i in range(10):
		for j in range(i+1,10):
			X_trai, Y_trai=read_specific_data(i, j)
			X_train=np.array(X_trai)
			Y_train=np.array(Y_trai)
			Theta, b=training(X_train, 1, Y_train,1)

			Thetas_all.append(Theta)
			b_all.append(float(b))

	Thetas_all=np.array(Thetas_all)
	b_all=np.array(b_all)
	return Thetas_all, b_all


Thetas_all, b_all=find_all_thetas()

# with open('trainedddd.pkl', 'wb') as f:
# 	pickle.dump(Thetas_all,f)

def read_testing_data():
	with open('mnist/test.csv', 'r') as f:
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


X_tes,Y_tes=read_testing_data()
X_test=np.array(X_tes)
Y_test=np.array(Y_tes)

def testing_data():

	itemcount=0
	accuracy=0

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
		if Y_test[itemcount]==index:
			accuracy+=1
		itemcount+=1


	print(accuracy/itemcount)



f=open('trained_models/svm_thetas_all.pkl', 'rb')
Thetas_all=pickle.load(f)
f.close()

f_=open('trained_models/svm_b_all.pkl', 'rb')
b_all=pickle.load(f_)
f.close()

# with open('trained_models/svm_b_all.pkl', 'wb') as f:
# 	pickle.dump(b_all,f)

testing_data()

































