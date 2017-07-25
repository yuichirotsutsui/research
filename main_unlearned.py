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
        array1.append(map(float, line1))
    data1 = numpy.array(array1)
    return data1,len(array1[0])

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
    data2 = gbrbm.make_memory(numpy.c_[data_left, zero_data], num = reconstruct_num, data = data_left, var = 0.00000000001, simple_sample = True)
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
    f_dic = open("relation_1")
    line = f_dic.readline()
    dic_word = {}
    while line:
        word1 = line.strip("\n").split(" ")[0]
        word2 = line.strip("\n").split(" ")[1]
        if word1 not in dic_word:
            dic_word[word1] = [word2]
        else:
            dic_word[word1].append(word2)
        line = f_dic.readline()
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
        if max_key in dic_word[line.strip("\n").split(" ")[0]]:
            ans_num += 1.0
        count += 1.0
        print 100 * ans_num / count

load_gbrbm = True
pp = 0.0001
hidden_layer = 1000
epoch = 20000
learning_file = "words_3"
dic_path = "./renso_normalized"
reconstruct_num = 100
batch_size = 10
proc = 1

rng = cupy.random.RandomState(123)
data1,data1_len = load_vec_data("./" + learning_file + "_dir/" + learning_file + "_vec")
dic = load_dic(dic_path)
data2 = numpy.array([[0 for i in range(200)] for j in range(len(data1))])
learning_data = numpy.c_[data1, data2]

gbrbm = None
if load_gbrbm:
    f_gbrbm = open("gbrbm_dump")
    gbrbm = pickle.load(f_gbrbm)
    f_gbrbm.close()
else:
    gbrbm = GBRBM.GBRBM(input = copy.copy(learning_data), n_visible = learning_data_len, n_hidden = hidden_layer, cupy_rng = rng)

ans = get_ans_gbrbm(learning_data, gbrbm, reconstruct_num = reconstruct_num)
calc(learning_file, chainer.cuda.to_cpu(ans), dic, proc)
