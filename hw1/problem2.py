import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

def SampleMethod1(numSamples):
	thetas = bernoulli.rvs(1.0/3.0, size = numSamples)
	ys = []
	for theta in thetas:
		#print(theta)
		y = []
		if theta == 0:
            		for i in range(5):
                		y.append(bernoulli.rvs(1.0/2.0, size = 1))
		elif theta == 1:
            		for i in range(5):
                		y.append(bernoulli.rvs(3.0/4.0, size = 1))
		ys.append(y)
	return (thetas, ys)

def SampleMethod2(numSamples):
	ys = []
	thetas = []
	for i in range(numSamples):
		y = []
		p = 1.0/3.0
		for j in range(5):
			theta = bernoulli.rvs(p, size = 1)
			if theta == 0:
				sample = bernoulli.rvs(1.0/2.0, size = 1)
			elif theta == 1:
				sample = bernoulli.rvs(3.0/4.0, size = 1)
			y.append(sample)
			p = p*bernoulli.pmf(sample, 3.0/4.0)/(p*bernoulli.pmf(sample,3.0/4.0) +(1 - p)*bernoulli.pmf(sample,1.0/2.0))
		ys.append(y)
		thetas.append(bernoulli.rvs(p,size = 1))
	return (thetas, ys)


if __name__== "__main__":
    numSamples = 10000
    thetas1, ys1 = SampleMethod1(numSamples)
    thetas2, ys2 = SampleMethod2(numSamples)
    ys_np1 = np.sum(np.array(ys1),axis = 1)
    ys_np2 = np.sum(np.array(ys2),axis = 1)
    plt.figure()
    ax1 = plt.subplot(211)
    ax1.set_title("Method1")
    plt.hist(ys_np1,bins = 5)
   # plt.hist(np.array(thetas1), bins = 2)
    ax2 = plt.subplot(212)
    ax2.set_title("Method2")
    plt.hist(ys_np2, bins = 5)
    #plt.hist(np.array(thetas2), bins = 2)
    plt.show()

  
    
