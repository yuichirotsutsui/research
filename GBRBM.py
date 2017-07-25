#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Restricted Boltzmann Machine (RBM)
    References :
    - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
    Training of Deep Networks, Advances in Neural Information Processing
    Systems 19, 2007
    - DeepLearningTutorial
    https://github.com/lisa-lab/DeepLearningTutorials
    """

import sys
import numpy
import random
import copy
import cupy
import chainer.cuda
import chainer
from chainer import computational_graph
from chainer import cuda, Variable

#numpy.seterr(all='ignore')

def sigmoid(x):
	return 1. / (1 + cupy.exp(-1 * x))


class GBRBM(object):
	def __init__(self, input=None, n_visible=2, n_hidden=30, W=None, hbias=None, vbias=None, cupy_rng=None):
		self.n_visible = n_visible  # num of units in visible (input) layer
		self.n_hidden = n_hidden    # num of units in hidden layer
		if cupy_rng is None:
			cupy_rng = cupy.random.RandomState(1234)
		numpy_rng = numpy.random.RandomState(1234)
		if W is None:
			a = 1. / n_visible
			initial_W = cupy.array(cupy_rng.uniform(low=-a, high=a, size=(n_visible, n_hidden)))
			W = initial_W
		if hbias is None:
			hbias = cupy.zeros(n_hidden)  # initialize h bias 0
		if vbias is None:
			vbias = cupy.zeros(n_visible)  # initialize v bias 0
		self.cupy_rng = cupy_rng
		self.numpy_rng = numpy_rng
		self.input = chainer.cuda.to_gpu(input)
		self.input_cpu = chainer.cuda.to_cpu(input)
		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.r_num = 0

	def binomial(self, size, p):
		ran_list = cupy.random.sample(size = size)
		tmp_list = p - ran_list
		tmp_list2 = cupy.where(tmp_list > 0, 1, tmp_list)
		return cupy.where(tmp_list2 < 0, 0, tmp_list2)

	def normal(self, size, p, var):
		ran_list = cupy.random.randn(size[0], size[1])
		return p - (ran_list * var)

	def contrastive_divergence(self, lr=0.1, k=1, input=None, batch_size = 10):
		if input is not None:
			self.input = input
		numpy.random.shuffle(self.input_cpu)
		data = chainer.cuda.to_gpu(self.input_cpu)
		i = 0
		steps = batch_size
		while 1:
			if steps * (i + 1) > len(data):
				break
			input_data = data[i * steps:(i + 1) * steps]
			ph_mean, ph_sample = self.sample_h_given_v(input_data)
			#if i == 0:
			#	print "ph_mean"
			#	print ph_mean[0][0:10]
			#	print "ph_sample"
			#	print ph_sample[0][0:10]
			chain_start = ph_sample
			for step in xrange(k):
				if step == 0:
					nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
				else:
					nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)
			i += 1
			#if i == 1:
			#	print "nv_means"
			#	print nv_means[0][0:10]
			#	print "nv_samples"
			#	print nv_samples[0][0:10]
			#	print "nh_means"
			#	print nh_means[0][0:10]
			#	print "nh_samples"
			#	print nh_samples[0][0:10]
			#	print "w"
			#	print self.W[0][0:10]
			#	print "vbias"
			#	print self.vbias[0:10]
			#	print "hbias"
			#	print self.hbias[0:10]
			self.W += lr * (cupy.dot(input_data.T, ph_mean)- cupy.dot(nv_means.T, nh_means)) / batch_size
			self.vbias += lr * cupy.mean(input_data - nv_means, axis=0)
			self.hbias += lr * cupy.mean(ph_mean - nh_means, axis=0)

	def sample_h_given_v(self, v0_sample, simple_sample = False):
		h1_mean = self.propup(v0_sample)
		h1_sample = None
		if simple_sample:
			h1_sample = cupy.where(h1_mean > 0.5, 1, h1_mean)
			h1_sample = cupy.where(h1_sample <= 0.5, 0, h1_sample)
		else:
			h1_sample = self.binomial(size=h1_mean.shape, p=h1_mean)
		return [h1_mean, h1_sample]

	def sample_v_given_h(self, h0_sample, var = 1):
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.normal(size = v1_mean.shape, p = v1_mean, var = var)
		return [v1_mean, v1_sample]

	def propup(self, v):
		pre_sigmoid_activation = cupy.dot(v, self.W) + self.hbias
		return sigmoid(pre_sigmoid_activation)

	def propdown(self, h):
		return cupy.dot(h, self.W.T) + self.vbias
		
	def gibbs_hvh(self, h0_sample):
		v1_mean, v1_sample = self.sample_v_given_h(h0_sample, var = 0)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [v1_mean, v1_sample, h1_mean, h1_sample]

	def make_memory(self, v0_sample, num, data = None, var = 0, simple_sample = True):
		v0_sample = chainer.cuda.to_gpu(v0_sample)
		if data is not None: data = chainer.cuda.to_gpu(data)
		ph_mean, ph_sample = self.sample_h_given_v(v0_sample, simple_sample = simple_sample)
		pv_mean, pv_sample = self.sample_v_given_h(ph_sample, var = var)
		if num == 1:
			return pv_sample
		else:
			for i in range(num - 1):
				ph_mean, ph_sample = self.sample_h_given_v(v0_sample, simple_sample = simple_sample) if data is None else self.sample_h_given_v(cupy.hstack((data, cupy.hsplit(pv_sample, [len(data[0])])[1])), simple_sample = simple_sample)
				pv_mean, pv_sample = self.sample_v_given_h(ph_sample, var = var)
			return pv_sample

	def get_reconstruction_cross_entropy(self):
		ph_mean, ph_sample = self.sample_h_given_v(self.input, simple_sample = True)
		v1_mean = self.propdown(ph_sample)
		v1_sample = self.normal(size = v1_mean.shape, p =  v1_mean, var = 0)
		cross_entropy = cupy.mean((self.input - v1_sample) * (self.input - v1_sample))
		return cross_entropy
