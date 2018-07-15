import csv
import numpy
import math
import sys
import io
import string

def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    finarr = []
    for row in reader:
        temp = []
        for item in row:
            temp.append(int(item))
        finarr.append(temp)
    return finarr    

def read_data():
    csv_path = sys.argv[1]
    with open(csv_path) as f_obj:
        inp = csv_reader(f_obj)
    inp_arr = numpy.array(inp)
    size = len(inp_arr)
    feature = len(inp_arr[0])
    return inp_arr, size, feature

def write_data(inp, size, feature):
    file = open(sys.argv[2],'w')
    for i in range(0,size):
        file.write(str(0)+" ")
        for j in range(0,feature):
            file.write(str(j)+":"+str(inp[i][j])+" ")
        file.write('\n')
    file.close()

  
inputx, size, feature = read_data()
write_data(inputx, size, feature)