#coding:utf-8

import bbrbm
import gbrbm
import numpy
import os
import sys
import pickle


def sigmoid(x):
    return 1. / (1 + numpy.exp(-1 * x))
numpy.seterr(all='ignore')

def load_vec_data(file_name):
    if os.path.exists(file_name):
        f = open(file_name)
    else:
        return 0
    array1 = []
    array2 = []
    line = f.readline()
    while line:
        tmp1= []
        tmp2 = []
        for i in range(4):
            num = (line.strip("\n")).split(" ")
            for j in range(len(num)):
                if i == 0:
                    tmp1.append(float(num[j]))
                elif i == 2:
                    tmp2 .append(float(num[j]))
            line = f.readline()
        array1.append(tmp1)
        array2.append(tmp2)
    data1 = numpy.array(array1)
    data2 = numpy.array(array2)
    return data1,len(array1[0]), data2, len(array2[0])

def learn_rbm(rbm, learning_rate = 0.0001, k = 1, training_epochs = 1000):
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr = learning_rate, k = k)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

def make_hidden(rbm, data):
    a, b = rbm.sample_h_given_v(data)
    print b
    return b, len(b[0])

def get_ans(data, gbrbm1, gbrbm2, bbrbm, hidden_num):
    print "a"
    #numpy.savetxt("f_input.csv", data[0], delimiter=",")
    a, data1 = gbrbm1.sample_h_given_v(data)
    print "b"
    #numpy.savetxt("f_gbrbm1.csv", data1[0], delimiter=",")
    zero_data = numpy.array([[0 for i in range(hidden_num)] for j in range(len(data1))])
    print "c"
    #numpy.savetxt("f_bbrbm_input.csv", numpy.c_[data1, zero_data][0], delimiter=",")
    data2 = bbrbm.make_memory2(numpy.c_[data1, zero_data], num = 50, data = data1)
    print "d"
    #numpy.savetxt("f_gbrbm_out.csv", data2[0], delimiter=",")
    a, data3 = gbrbm2.sample_v_given_h2((numpy.hsplit(data2, [len(data1[0])]))[1])
    print "e"
    #numpy.savetxt("f_gbrbm2_out.csv", data3[0], delimiter=",")
    return data3

def get_ans_gbrbm(data, gbrbm):
    print "a"
    zero_data = numpy.array([[0 for i in range(200)] for j in range(len(data))])
    print "b"
    data_left = numpy.hsplit(data, [200])[0]
    print "c"
    data2 = gbrbm.make_memory2(numpy.c_[data_left, zero_data], num = 100, data = data_left)
    print "d"
    return numpy.hsplit(data2, [200])[1]

def load_test_vec(file_name1, file_name2):
    if os.path.exists(file_name1) and os.path.exists(file_name2):
        f1 = open(file_name1)
        f2 = open(file_name2)
    else:
        return 0
    dic = {}
    line1 = f1.readline()
    line2 = f2.readline()
    while line1:
        tmp = (line1.strip("\n")).split(" ")
        for j in range(2):
            if tmp[j * 2] in dic:
                pass
            else:
                array = []
                tmp2 = (line2.strip("\n")).split(" ")
                for i in range(len(tmp2)):
                    array.append(tmp2[i])
                dic[tmp[j * 2]] = numpy.array(array)
            line2 = f2.readline()
            line2 = f2.readline()
        line1 = f1.readline()
    return dic

def calc_ans(file_name, data, dic, top = 1, print_result = True, file_num = 0):
    count = 0.0
    ans_num = 0.0
    if os.path.exists(file_name):
        f = open(file_name)
    word = dic.keys()
    if print_result == True:
        ff = open("relation_ans", "w")
    for i in range(len(data)):
        line = f.readline()
        num_list = []
        word_list = []
        for j in range(len(word)):
            tmp = []
            for k in range(len(dic[word[j]])):
                tmp.append(float(dic[word[j]][k]))
            num = numpy.dot(data[i], numpy.array(tmp)) / (numpy.linalg.norm(data[i]) * numpy.linalg.norm(numpy.array(tmp)))
            check = 0
            for k in range(len(num_list)):
                if num_list[k] < num:
                    num_list.insert(k, num)
                    word_list.insert(k, word[j])
                    check = 1
                    break
            if check == 0:
                num_list.append(num)
                word_list.append(word[j])
        for j in range(top):
            count += 1.0
            if (line.strip("\n")).split(" ")[2] == word_list[j]:
                ans_num += 1.0
            print line.strip("\n")
            print str(j) + ":" + word_list[j] + ":" + str(num_list[j])
            print str(100 * ans_num / count) + "\n"
            if print_result == True:
                ff.write(line.strip("\n") + "\n")
                ff.write(str(j) + ":" + word_list[j] + ":" + str(num_list[j]) + "\n")
                #ff.write(str(100 * ans_num / count) + "\n")
    if print_result == True:
        ff.close()
def test_gbrbm(data, data_len, rng):
    h_end = 3000
    h_now = 1750
    h_step = 50
    l_end = 0.001
    e_event = 0.01
    e_step = 100
    f_test_ans = open("rbm_ans.txt", "w")
    while True:
        epoch_list = [0 for j in range(e_step)]
        l_now = 0.1
        while True:
            epoch = 0
            gbrbm_test = gbrbm.GBRBM(input = data, n_visible = data_len, n_hidden = h_now, numpy_rng = rng)
            while True:
                gbrbm_test.contrastive_divergence(lr = l_now, k = 1)
                cost = gbrbm_test.get_reconstruction_cross_entropy()
                print str(h_now) + " " + str(l_now) + ":Training epoch " + str(epoch) + ", cost is " + str(cost)
                if epoch > 100 and (cost > epoch_list[epoch % e_step] * (1 - e_event) or cost != cost):
                    f_test_ans.write(str(h_now) + " " + str(l_now) + "\n")
                    f_test_ans.write(str(epoch) + " " + str(cost) + "\n")
                    break
                epoch_list[epoch % e_step] = cost
                epoch += 1
            if l_now <= l_end:
                break
            l_now *= 0.1
        if h_now >= h_end:
            break
        h_now += h_step
    f_test_ans.close()


def add_data(data):
    data2 = data.tolist()
    f_l = open("random_info_l")
    f_r = open("random_info_r")
    line_l = f_l.readline()
    line_r = f_r.readline()
    random_list_l = []
    random_list_r = []
    while line_l:
        random_list_l.append(line_l.strip("\n"))
        line_l = f_l.readline()
    while line_r:
        random_list_r.append(line_r.strip("\n"))
        line_r = f_r.readline()
    f_l.close()
    f_r.close()
    ff = open("relation_learn_number")
    line = ff.readline()
    for i in range(len(data2)):
        nums_l = (random_list_l[int((line.split(" "))[1])].strip("\n")).split(" ")
        nums_r = (random_list_r[int((line.split(" "))[3])].strip("\n")).split(" ")
        for j in range(len(nums_l)):
            data2[i].append(float(nums_l[j]))
        for j in range(len(nums_r)):
            data2[i].append(float(nums_r[j]))
        line = ff.readline()
    ff.close()
    return numpy.array(data2), len(data2[0])


rng = numpy.random.RandomState(123)
data1,data1_len, data2, data2_len = load_vec_data("relation_1_not_number_vec")
f_gbrbm = open("gbrbm_dump")
#f_gbrbm2 = open("gbrbm2_dump")
gbrbm = pickle.load(f_gbrbm)
#gbrbm2 = pickle.load(f_gbrbm2)
#f_gbrbm1.close()
#f_gbrbm2.close()
#data11, data11_len = make_hidden(gbrbm1, data1)
#data11, data11_len = add_data(data11)
#data22, data22_len = make_hidden(gbrbm2, data2)
dic = load_test_vec("data_number", "data_number_vec")
data33 = numpy.c_[data1, data2]
data33_len = data1_len + data2_len
pp = 0.0001
#gbrbm = gbrbm.GBRBM(input = data33, n_visible = data33_len, n_hidden = 1000, numpy_rng = rng)
#learn_rbm(gbrbm, training_epochs = 1000, learning_rate = pp)
for i in range(10000):
    print i
    #learn_rbm(gbrbm, training_epochs = 19000, learning_rate = pp)
    #f_gbrbm = open("gbrbm_dump", "w")
    #pickle.dump(gbrbm, f_gbrbm)
    #f_gbrbm.close()
    ans = get_ans_gbrbm(data33, gbrbm)
    calc_ans("relation_1_not_number", ans, dic, top = 1, print_result = True, file_num = 0)
    exit()