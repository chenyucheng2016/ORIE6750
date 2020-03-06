import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
def computeQFactor1(prob,horizon,V,M):#problem a
	#T(p,y)
	#likelihood
	Y = np.array([0,1])
	Q = 0
	for y in Y:
		prob_next = prob*f1_prob_mass_bern(y)/(prob*f1_prob_mass_bern(y) + (1.0-prob)*f0_prob_mass_bern(y))
		i = round(prob_next / (1.0/M))
		Q = Q + V[int(i),int(horizon-1)]*(prob*f1_prob_mass_bern(y) + (1.0-prob)*f0_prob_mass_bern(y))
	return Q
def computeQFactor2(prob,horizon,V,M):#problem a
	#T(p,y)
	#likelihood
	delta = 0.1
	Y = np.arange(-7,7,delta)
	Q = 0
	for y in Y:
		prob_next = prob*f1_prob_mass_Gauss(y,delta)/(prob*f1_prob_mass_Gauss(y,delta) + (1.0-prob)*f0_prob_mass_Gauss(y,delta))
		i = round(prob_next / (1.0/M))
		Q = Q + V[int(i),int(horizon-1)]*(prob*f1_prob_mass_Gauss(y,delta) + (1.0-prob)*f0_prob_mass_Gauss(y,delta))
	return Q
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


if __name__== "__main__":
	c = 0.05
	H = 20
	M = 20
	delta = 1.0/M
	p_arr = np.arange(0.,1. + delta/2.0, delta)
	H_arr = np.arange(0,H+1)
	V = np.zeros((p_arr.size,H+1))
	Qvalue = np.zeros((p_arr.size,H+1))
	for h in H_arr:
		for i in range(M+1):
			p = p_arr[i]
			if h == 0:
				V[i,h] = min(p,1-p)
			else:
				#q = p*f1_prob_mass_bern(1) + (1-p)*f0_prob_mass_bern(1)
				Qvalue[i,h] = c + computeQFactor2(p,h,V,M)
				#Qvalue[i,h] = c + V[i,h-1]*(1-q) + V[i+1,h-1]*q
				V[i,h] = min(p,1-p, Qvalue[i,h])
	plt.figure('')
	ax = plt.plot(p_arr,V[:,0])
	ax = plt.plot(p_arr,Qvalue[:,H])
	plt.show()


