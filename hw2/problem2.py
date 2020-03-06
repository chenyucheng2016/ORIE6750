import numpy as np
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


def f0_prob_mass_bern(y):
	if y == 0:
		return 1.0/2
	elif y == 1:
		return 1.0/2
	else:
		print('wrong input')
		return -1
def f1_prob_mass_bern(y):
	if y == 0:
		return 1.0/3
	elif y == 1:
		return 2.0/3
	else:
		print('wrong input')
		return -1
def f0_prob_mass_Gauss(y,delta):
	return norm(0,1).cdf(y) - norm(0,1).cdf(y-delta)
def f1_prob_mass_Gauss(y,delta):
	return norm(1,1).cdf(y) - norm(1,1).cdf(y-delta)

def computeQFactor1(prob,V,M):#problem a
	Y = np.array([0,1])
	Q = 0
	for y in Y:
		prob_next = prob*f1_prob_mass_bern(y)/(prob*f1_prob_mass_bern(y) + (1.0-prob)*f0_prob_mass_bern(y))
		i = round(prob_next / (1.0/M))
		Q = Q + V[int(i)]*(prob*f1_prob_mass_bern(y) + (1.0-prob)*f0_prob_mass_bern(y))
	return Q
def computeQFactor2(prob,V,M):#problem a
	delta = 0.01
	Y = np.arange(-7,7,delta)
	Q = 0
	for y in Y:
		prob_next = prob*f1_prob_mass_Gauss(y,delta)/(prob*f1_prob_mass_Gauss(y,delta) + (1.0-prob)*f0_prob_mass_Gauss(y,delta))
		i = round(prob_next / (1.0/M))
		Q = Q + V[int(i)]*(prob*f1_prob_mass_Gauss(y,delta) + (1.0-prob)*f0_prob_mass_Gauss(y,delta))
	return Q

def ValueIter(V,N,M,c):
	delta = 1.0/M
	for i in range(N):
		print('iteration',i)
		for j in range(M):
			p = j*delta
			#print(c+computeQFactor1(p,V,M))
			V[j] = min(p, 1-p, c + computeQFactor2(p,V,M))
	Q = np.zeros(M)
	for i in range(M):
		p = i*delta
		Q[i] = c + computeQFactor1(p,V,M)
	return V, Q

if __name__ == "__main__":
	"""
	M = 100
	N = 1000
	c = 0.05
	delta = 1.0/M
	p_arr = np.arange(0.,1.0, delta)
	V0 = np.minimum(p_arr,1 - p_arr)
	V = np.random.rand(M+1)
	V,Q = ValueIter(V,N,M,c)
	a = -1
	b = -1
	flag = 0
	for i in range(M+1):
		if Q[i] < V0[i]:
			if i*delta < 0.5 and flag == 0:
				a = i*delta
				flag = 1
			else:
				b = i*delta
	print(a)
	print(b)
	"""
	c = 0.05
	a = 0.2
	b = 0.8
	sampNum = 1000
	epochs = 100
	epochCost = np.zeros(epochs)
	for e in range(epochs):
		sampleCosts = -np.ones(sampNum)
		p0 = 0.5
		for i in range(sampNum):
			u = np.random.uniform(0,1,1)#sample from p(theta)
			if u >= p0:
				theta = 0
			else:
				theta = 1
			p = p0
			goCost  = 0
			while (p > a and p < b):
				if u > p0:#sample f0
					y = np.random.normal(0,1,1)
				else:
					y = np.random.normal(1,1,1)
				p = p*f1_prob_mass_Gauss(y,0.1)/(p*f1_prob_mass_Gauss(y,0.1) + (1-p)*f0_prob_mass_Gauss(y,0.1))
				goCost = goCost + c
			if p < a:
				theta_est = 0
			else:
				theta_est = 1
			if theta == theta_est:
				sampleCosts[i] = goCost
			else:
				sampleCosts[i] = 1 + goCost
			epochCost[e] = np.average(sampleCosts)
		print(e)
	print(epochCost)
	plt.figure()
	ax = plt.hist(epochCost, bins = 20)
	plt.show()


"""
if __name__== "__main__":
	M = 1000
	N = 1000
	c = 0.05
	delta = 1.0/M
	p_arr = np.arange(0.,1.0 + delta/2, delta)
	V0 = np.minimum(p_arr,1 - p_arr)
	V = np.random.rand(M+1)
	V,Q = ValueIter(V,N,M+1,c)
	a = -1
	b = -1
	flag = 0
	for i in range(M+1):
		if Q[i] < V0[i]:
			if i*delta < 0.5 and flag == 0:
				a = i*delta
				flag = 1
			else:
				b = i*delta
	sampNum = 1000
	epochs = 20
	epochCost = np.zeros(epochs)
	for e in range(epochs):
		sampleCosts = -np.ones(sampNum)
		p0 = 0.5
		for i in range(sampNum):
			u = np.random.uniform(0,1,1)#sample from p(theta)
			if u >= p0:
				theta = 0
			else:
				theta = 1
			p = p0
			goCost  = 0
			while (p > a and p < b):
				if u > p:#sample f0
					y = bernoulli.rvs(1.0/2.0, size = 1)
				else:
					y = bernoulli.rvs(2.0/3.0, size = 1)
			#print(y)
				p = p*f1_prob_mass_bern(y)/(p*f1_prob_mass_bern(y) + (1-p)*f0_prob_mass_bern(y))
				goCost = goCost + c
			#print(p)
			if p < a:
				theta_est = 0
			else:
				theta_est = 1
			if theta == theta_est:
				sampleCosts[i] = goCost
			else:
				sampleCosts[i] = 1 + goCost
			epochCost[e] = np.average(sampleCosts)
	plt.figure()
	ax = plt.hist(epochCost, bins = 5)
	plt.show()
"""
    




