import numpy as np 
from numpy import array as a 
from multiprocessing_generator import ParallelGenerator
from random import random as r
from random import gauss as g
from random import shuffle
import matplotlib.pyplot as plt
from time import sleep
from pickle import dump, load 

class agent:
	c = a([0,0])	# Row,Col
	E = []			# Sensing Matrix
	env = ''		# Environment Generator
	f = 0			# Fitness Score

	G = []			# Gene Matrix
	P = []			# Next Step Policy
	cfg = ''

	H = []			# Move History


	def __init__(self,env,cfg):
		self.env = env
		self.E = env.getSensingMatrix(self.c)
		self.G = gene(cfg)
		self.cfg = cfg
		self.get_coords()
		self.H = np.zeros([cfg.M,1])

	def step(self):

		self.E = self.env.getSensingMatrix(self.c)

		if np.linalg.norm(self.c) < self.cfg.B + 10:

			E = self.E
			A = self.G.A 
			B = self.G.B 
			D = self.G.D

			self.P = (A@E@B)*np.eye(8)@np.ones([8,1]) + D@self.H



			for i in range(len(self.P)):
				if self.P[i] == np.array(self.P).max():
					break

			self.H[1:] = self.H[:-1]
			self.H[0] = i 

			# Up
			if i == 0:
				self.c += a([1,0])
			# Down
			elif i == 1:
				self.c += a([-1,0])
			# Left
			elif i == 2:
				self.c += a([0,-1])
			# Right
			elif i == 3:
				self.c += a([0,1])
			# Up Right
			elif i == 4:
				self.c += a([1,1])
			# Down Right
			elif i == 5:
				self.c += a([-1,1])
			# Down Left
			elif i == 6:	
				self.c += a([-1,-1])
			# Up Left
			elif i == 7:
				self.c += a([1,-1])

			self.f += self.env.consume(self.c)

			return(i)

		return(-999)

	def get_coords(self):
		c = [int(g(0,10)),int(g(0,10))]
		while np.linalg.norm(c) > self.cfg.B:
			c = [int(g(0,10)),int(g(0,10))]
		self.c = c

class genePool:

	P = []	# Gene Pool [N_A,Dir,Vision]
	cfg = ''

	def __init__(self,c):
		self.cfg = c

	def save(self,fn):
		dump(self.P,open(fn,'wb'))

	def loadGene(self,fn=''):
		if fn == '':
			self.P = load(open(self.cfg.FN,'rb'))	
		else:
			self.P = load(open(fn,'rb'))

	def getShuffleReducePool(self,agents):
		tmp = []

		fs = []
		for agent in agents:
			fs.append(agent.f)
		idx = sorted(range(len(fs)), key=lambda k: fs[k])
		idx = idx[int(len(idx)/2):]


		for i in idx:
			tmp.append(agents[i].G.W)
		tmp = a(tmp)
		for i in range(tmp.shape[1]):
			for j in range(tmp.shape[2]):
				np.random.shuffle(tmp[:,i,j])

		self.P = tmp

	def mutate(self):
		self.P += np.random.normal(0,1,self.P.shape)

	def pltAvg(self):
		plt.figure('Weights')
		plt.clf()
		plt.imshow(self.P.mean(axis=0),aspect='equal')
		plt.show(block=False)
		plt.pause(.001)

	def plt_weights(self):
		p = self.P.mean(axis=0)
		A = p[:8,:]
		B = p[8:16,:]
		D = p[16:,:]

		g = np.zeros([A.shape[1],A.shape[1]])



		fig,ax = plt.subplots(nrows=2,ncols=5,num='Weight Distribution',clear=True)

		names = ['up','down','left','right','UR','DR','DL','UL','Memory']

		for i in range(8):
			v = A[i,:]
			h = B[i,:]

			for j in range(len(v)):
				for k in range(len(h)):
					g[j,k] = v[j]*h[k]

			# ax[int(i/4),i%4].clear()
			ax[int(i/4),i%4].title.set_text(names[i])
			ax[int(i/4),i%4].imshow(g,aspect='equal')


		ax[0,4].imshow(D,aspect='equal')
		ax[1,4].axis('off')

		plt.show(block=False)
		plt.pause(.001)



	def reproduce(self,agents,init=False):

		self.save('current.p')

		self.getShuffleReducePool(agents)
		self.mutate()

		fs = []
		for agent in agents:
			fs.append(agent.f)
		idx = sorted(range(len(fs)), key=lambda k: fs[k])
		if not init:
			try:
				idx = idx[:int(len(idx)/2)]
			except Exception as e:
				print(e)
				print('Use Even Number of Agents')
		else:
			self.loadGene()

		PCTR = 0
		for i in range(len(agents)):
			if i in idx:
				agents[i].G.W = self.P[PCTR%self.P.shape[0]]
				agents[i].G.A = agents[i].G.W[:8,:]
				agents[i].G.B = agents[i].G.W[8:16,:].T
				agents[i].G.D = agents[i].G.W[16:,:]
				PCTR += 1


		i = max(idx)
		agents[i].G.W = self.P.mean(axis=0)
		agents[i].G.A = agents[i].G.W[:8,:]
		agents[i].G.B = agents[i].G.W[8:16,:].T
		agents[i].G.D = agents[i].G.W[16:,:]

		return(agents)

class gene:
	A = []	# Weight Matrix Vertical
	B = []	# Weight Matrix Horizontal
	D = [] 	# Weight Matrix Memory
	W = []	# Weigh Matrix [A B.T].T

	def __init__(self,c):
		# [up down left right UR DR DL UL]

		# P = A@E@B
		# 8X1 = 8XV VXV VX1
		self.W = np.random.normal(0,1,[24,c.V])
		self.A = self.W[:8,:]
		self.B = self.W[8:16,:].T
		self.D = self.W[16:,:]

class environment:

	master = {}
	c = ''

	def __init__(self,c):
		self.c = c

	def getSensingMatrix(self,coord):
		E = np.zeros([self.c.V,self.c.V])
		for i in range(int(coord[0]-(self.c.V-1)/2),int(coord[0]+(self.c.V-1)/2+1)):
			for j in range(int(coord[1]-(self.c.V-1)/2),int(coord[1]+(self.c.V-1)/2+1)):
				ir = int(i - coord[0] + (self.c.V-1)/2)
				jr = int(j - coord[1] + (self.c.V-1)/2)
				cstr = '['+str(i)+','+str(j)+']'
				if cstr not in self.master:
					if r()*100 <= self.c.S and np.linalg.norm([coord[0],coord[1]])<self.c.B:
						self.master[cstr] = 1
						E[ir,jr] = 1
					else:
						self.master[cstr] = 0
				else:
					E[ir,jr] = self.master[cstr]

		return(E)

	def consume(self,crd):
		cstr = '['+str(crd[0])+','+str(crd[1])+']'

		if self.master[cstr]:
			self.master[cstr] = 0
			return(1)
		else:
			return(0)

	def plot(self,A):
		master = self.master

		# Get range
		xm	= 0
		ym	= 0
		xma	= 0
		yma	= 0
		for key in master:
			x = eval(key)[0]
			y = eval(key)[1]

			if x > xma:
				xma = x 
			if x < xm:
				xm = x
			if y > yma:
				yma = y 
			if y < ym:
				ym = y

		grid = np.zeros([xma-xm+1,yma-ym+1])

		for key in master:
			xr = eval(key)[0] - xm
			yr = eval(key)[1] - ym
			grid[xr,yr] = master[key]

		for ag in A:
			xr = ag.c[0]-xm 
			yr = ag.c[1]-ym 
			grid[xr,yr] = -1

		plt.figure('Environment')
		plt.clf()
		plt.imshow(grid,aspect='equal')
		plt.show(block=False)
		plt.pause(.001)

class splice:

	A = []
	c = ''
	GP = ''
	md = []

	def __init__(self,c):

		self.c = c
		E = environment(c)
		self.A = [agent(E,c) for i in range(c.N_A)]
		self.GP = genePool(c)

	def sharedEnvironmentTrain(self):

		if self.c.FN:
			self.GP.reproduce(self.A,init=True)

		for j in range(300000):
			# initialize environment
			self.E = environment(c)
			self.E.master = {}
			for ag in self.A:
				ag.env = self.E
				ag.get_coords()

			self.md = []

			# For each agent, take step
			for i in range(self.c.L):
				for a in self.A:
					self.md.append(a.step())
				self.E.plot(self.A)

			self.performanceMetrics()

			# Reproduce
			self.A = self.GP.reproduce(self.A)

			# self.GP.pltAvg()
			self.GP.plt_weights()

			avg = 0
			maxv = 0
			for ag in self.A:
				avg += ag.f
				if ag.f > maxv:
					maxv = ag.f
				ag.f = 0

			avg /= self.c.N_A

			print('gen'+str(j)+': ')
			print(avg)
			print(maxv)

	def move_distribution(self):
		plt.figure('Move Distribution')
		K = np.zeros([3,3])
		for i in range(8):
			K[int(i/3),i%3] = len([j for j in self.md if j == i])
		K = K.flatten()
		K[5:] = K[4:-1]
		K = K.reshape([3,3])
		plt.clf()
		plt.imshow(K,aspect='equal')
		plt.show(block=False)
		plt.pause(.001)



	def performanceMetrics(self):

		self.move_distribution()

		print('Weight Variance:'+str(np.var(self.GP.P)))

		fs = []
		for ag in self.A:
			fs.append(ag.f)

		fs.sort()

		plt.figure('performance')
		plt.plot(fs)
		plt.show(block=False)
		plt.pause(.001)


class cfg:

	N_A = 10	# Num Agents
	S 	= 20	# Sparsity per 100
	V	= 10	# Vision / Size of sensing matrix
	B	= 1000	# Resource Boundary || x,y || > 1000
	L 	= 100   # Lifespan
	M 	= 10 	# Memory
	FN 	= ''	# Gene Pool Filename | Ignore if Empty

	def __init__(self,ID):
		
		# V5 M5
		if ID == 1:
			self.N_A 	= 40
			self.S 		= 60
			self.V 		= 5
			self.B 		= 30
			self.FN 	= 'well_trained_v5wM.p'
			self.L 		= 40
			self.M 		= 5

if __name__ == '__main__':


	c = cfg(1)
	
	s = splice(c)
	s.sharedEnvironmentTrain()

	plt.show()



