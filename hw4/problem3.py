import numpy as np
import copy
import matplotlib.pyplot as plt
def sim(epoch,H,num_arms):
	mu = 0
	sigma = 1
	lambdaVar = 1
	v = np.zeros(epoch)
	for i in range(epoch):
		xs = np.random.normal(mu, sigma, num_arms)
		thetas = np.zeros(num_arms)
		varis = np.ones(num_arms)
		for h in range(H):
			samples = []
			for j in range(num_arms):
				samples.append(np.random.normal(thetas[j], varis[j], 1))
			arm = np.argmax(samples)
			y = np.random.normal(xs[arm], lambdaVar, 1)
			thetas[arm] = (varis[arm]*y + lambdaVar*thetas[arm])/(varis[arm] + lambdaVar)
			varis[arm] = varis[arm] * lambdaVar / (varis[arm] + lambdaVar)
		v[i] = max(thetas)
	return np.mean(v)
def simRand(epoch,H,num_arms):
	mu = 0
	sigma = 1
	lambdaVar = 1
	v = np.zeros(epoch)
	for i in range(epoch):
		xs = np.random.normal(mu, sigma, num_arms)
		thetas = np.zeros(num_arms)
		varis = np.ones(num_arms)
		for h in range(H):
			arm = np.random.randint(0,num_arms)
			y = np.random.normal(xs[arm], lambdaVar, 1)
			thetas[arm] = (varis[arm]*y + lambdaVar*thetas[arm])/(varis[arm] + lambdaVar)
			varis[arm] = varis[arm] * lambdaVar / (varis[arm] + lambdaVar)
		v[i] = max(thetas)
	return np.mean(v)
def simUCB(epoch,H,num_arms):
	mu = 0
	sigma = 1
	lambdaVar = 1
	beta = 1.96
	v = np.zeros(epoch)
	for i in range(epoch):
		xs = np.random.normal(mu, sigma, num_arms)
		thetas = np.zeros(num_arms)
		varis = np.ones(num_arms)
		ucb = np.zeros(num_arms)
		for h in range(H):
			for j in range(num_arms):
				ucb[i] = thetas[i] + beta*np.sqrt(varis[i])
			arm = np.argmax(ucb)
			y = np.random.normal(xs[arm], lambdaVar, 1)
			thetas[arm] = (varis[arm]*y + lambdaVar*thetas[arm])/(varis[arm] + lambdaVar)
			varis[arm] = varis[arm] * lambdaVar / (varis[arm] + lambdaVar)
		v[i] = max(thetas)
	return np.mean(v)


if __name__=="__main__":
	H = 100
	num_arms = 10
	epoch = 5000
	v_ths = np.zeros(H)
	v_rands = np.zeros(H)
	v_ucbs = np.zeros(H)
	for i in range(1,H+1):
		v_th = sim(epoch, H, num_arms)
		v_rand = simRand(epoch, H, num_arms)
		v_ucb = simUCB(epoch,H,num_arms)
		v_ths[i] = v_th
		v_rands[i] = v_rand
		v_ucbs[i] = v_ucb
	x = np.linspace(1,H,H)
	plt.plot(x,v_ths,'g^')
	plt.plot(x,v_rands,'ro')
	plt.plot(x,v_ucbs,'bs')
	plt.show()






