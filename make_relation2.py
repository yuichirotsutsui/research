#coding: utf-8
f = open("relation")
dic = {}
line = f.readline()
while line:
  words = line.strip("\n").split(" ")
  if words[1] in dic:
    dic[words[1]] += 1
  else:
    dic[words[1]] = 1
  line = f.readline()
f.close()

f = open("relation")
f_w = open("relation_2", "w")
line = f.readline()
while line:
  words = line.strip("\n").split(" ")
  if dic[words[1]] >= 20:
    f_w.write(line)
  line = f.readline()
f.close()
f_w.close()
