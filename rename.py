#coding: utf-8
f = open("relation_1")
#衣服ー＞服
#食物ー＞食べ物
#生き物ー＞生物
dic = {"衣服": "服", "食物": "食べ物", "生き物": "生物"}
f_w = open("relation", "w")
line = f.readline()
list_write = []
while line:
  z_word =  line.strip("\n").split(" ")[0] 
  word = line.strip("\n").split(" ")[1]
  if word in dic:
    word = dic[word]
  list_write.append(z_word + " " + word + "\n")
  line = f.readline()
tmp = set(list_write)
list_write = list(tmp)
for line in list_write:
  f_w.write(line)
f.close()
f_w.close()
