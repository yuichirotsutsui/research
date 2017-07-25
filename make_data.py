#coding: utf-8
import random
import os
file_name = "relation_3_1"
file_name_2 = "words_1"
end_num = 1

f = open("relation_2")
dic = {}
line = f.readline()
while line:
  word = line.strip("\n").split(" ")[1]
  if word not in dic:
    dic[word] = [line]
  else:
    dic[word].append(line)
  line = f.readline()

f_w = open(file_name, "w")
for key in dic.keys():
  li = dic[key]
  random.shuffle(li)
  num = 0
  for line in li:
    f_w.write(line)
    num += 1
    if num == end_num:
      break

f.close()
f_w.close()

f = open(file_name)
line = f.readline()
kai_list = []
while line:
  word = line.split(" ")[0]
  kai_list.append(word)
  line = f.readline()

f.close()
f = open("relation_2")
line = f.readline()
w_list = []
while line:
  word = line.split(" ")[0]
  if word not in kai_list and word not in w_list:
    w_list.append(word)
  line = f.readline()

f_w = open(file_name_2, "w")
for word in w_list:
  f_w.write(word + "\n")

f.close()
f_w.close()

os.system('python make_vec.py ' + file_name)
os.system('python make_unlearned_test_data.py ' + file_name_2)

