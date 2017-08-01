#coding: utf-8
import random
import os

dic = {}
files = os.listdir("./ans/")
for file in files:
  f = open("./ans/" + file)
  lines = f.readlines()
  if len(lines) > 0:
    print file
    dic[file] = float(lines[-1].strip("\n"))
  f.close()
for ans in sorted(dic.items(), key=lambda x:x[1])[::-1]:
  print ans