#coding:utf-8

import GBRBM
import numpy
import os
import sys
import pickle
import random
import copy
import cupy
import chainer.cuda
import chainer
from chainer import computational_graph
from chainer import cuda, Variable
import multiprocessing as mp

def load_vec_data(file_name):
    if os.path.exists(file_name):
        f = open(file_name)
    else:
        print "ファイルがありません"
        exit()
    line = f.readline()
    array1 = []
    array2 = []
    while line:
        line1 = line.strip("\n").split(" ")
        line = f.readline()
        line2 = line.strip("\n").split(" ")
        line = f.readline()
        array1.append(map(float, line1))
        array2.append(map(float, line2))
    data1 = numpy.array(array1)
    data2 = numpy.array(array2)
    return data1,len(array1[0]), data2, len(array2[0])

def load_dic(dic_path):
    dic = {}
    if os.path.exists(dic_path):
         f = open("renso_normalized")
    else:
        print "ファイルがありません"
        exit()
    line = f.readline()
    while line:
        word = line.strip("\n").split(" ")[0]
        vecs = line.strip("\n").split(" ")[1:]
        dic[word] = map(float, vecs)
        line = f.readline()
    f.close()
    return dic

def learn_rbm(rbm, learning_rate = 0.0001, k = 1, training_epochs = 1000, batch_size = 10):
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr = learning_rate, k = k, batch_size = batch_size)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

def get_ans_gbrbm(data, gbrbm, reconstruct_num = 100):
    zero_data = numpy.array([[0 for i in range(200)] for j in range(len(data))])
    data_left = numpy.hsplit(data, [200])[0]
    data2 = gbrbm.make_memory(numpy.c_[data_left, zero_data], num = reconstruct_num, data = data_left, var = 0, simple_sample = True)
    data2 = gbrbm.make_memory(data2, num = reconstruct_num, var = 0, simple_sample = True)
    return numpy.hsplit(data2, [200])[1]

def sub_calc(ans, keys, dic):
    max_num = -10
    max_key = ""
    for key in keys:
        num = numpy.dot(ans, numpy.array(dic[key])) / (numpy.linalg.norm(ans) * numpy.linalg.norm(numpy.array(dic[key])))
        if max_num <= num:
            max_num = num
            max_key = key
    return {"word": max_key, "value": max_num}

def wrap_calc(data):
    return sub_calc(*data)

def calc(file_name, ans, dic, proc):
    if os.path.exists(file_name):
        f = open(file_name)
    words = dic.keys()
    nums = [(len(words) + i) // proc for i in range(proc)]
    pool = mp.Pool(proc)
    count = 0.0
    ans_num = 0.0
    for ans_1 in ans:
        line = f.readline()
        sum_nums = 0
        args = []
        for num in nums:
            args.append((ans_1, words[sum_nums:sum_nums + num], dic))
            sum_nums += num
        callback = pool.map(wrap_calc, args)
        max_num = -10
        max_key = ""
        for info in callback:
            if max_num <= info["value"]:
                max_num = info["value"]
                max_key = info["word"]
        print "\n" + line.strip("\n")
        print max_key + ":" + str(max_num)
        if line.strip("\n").split(" ")[1] == max_key:
            ans_num += 1.0
        count += 1.0
        print 100 * ans_num / count

load_gbrbm = False
pp = 0.0001
hidden_layer = 100
epoch = 10000
learning_file = "relation_3_1"
dic_path = "./renso_normalized"
reconstruct_num = 100
batch_size = 10
proc = 1

rng = cupy.random.RandomState(123)
data1,data1_len, data2, data2_len = load_vec_data("./" + learning_file + "_dir/" + learning_file + "_vec")
dic = load_dic(dic_path)
learning_data = numpy.c_[data1, data2]
learning_data_len = data1_len + data2_len

gbrbm = None
if load_gbrbm:
    f_gbrbm = open("gbrbm_dump")
    gbrbm = pickle.load(f_gbrbm)
    f_gbrbm.close()
else:
    gbrbm = GBRBM.GBRBM(input = copy.copy(learning_data), n_visible = learning_data_len, n_hidden = hidden_layer, cupy_rng = rng)

learn_rbm(gbrbm, training_epochs = epoch, learning_rate = pp, batch_size = batch_size)
f_gbrbm = open("gbrbm_dump", "w")
pickle.dump(gbrbm, f_gbrbm)
f_gbrbm.close()
ans = get_ans_gbrbm(learning_data, gbrbm, reconstruct_num = reconstruct_num)
calc(learning_file, chainer.cuda.to_cpu(ans), dic, proc)
