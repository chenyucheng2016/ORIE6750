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
				ucb[j] = thetas[j] + beta*np.sqrt(varis[j])
			arm = np.argmax(ucb)
			y = np.random.normal(xs[arm], lambdaVar, 1)
			thetas[arm] = (varis[arm]*y + lambdaVar*thetas[arm])/(varis[arm] + lambdaVar)
			varis[arm] = varis[arm] * lambdaVar / (varis[arm] + lambdaVar)
		v[i] = max(thetas)
	return np.mean(v)
def simKG(epoch,H,num_arms):
	mu = 0
	sigma = 1
	lambdaVar = 1
	v = np.zeros(epoch)
	for i in range(epoch):
		xs = np.random.normal(mu, sigma, num_arms)
		thetas = np.zeros(num_arms)
		varis = np.ones(num_arms)
		for h in range(H):
			thetaprimes = thetaprime(thetas,varis,lambdaVar, np.linspace(0,num_arms-1,num_arms))
			arm = np.argmax(thetaprimes)
			y = np.random.normal(xs[arm], lambdaVar, 1)
			thetas[arm] = (varis[arm]*y + lambdaVar*thetas[arm])/(varis[arm] + lambdaVar)
			varis[arm] = varis[arm] * lambdaVar / (varis[arm] + lambdaVar)
		v[i] = max(thetas)
	return np.mean(v)
def thetaprime(thetas, varis, lambdaVar, arms):
	thetaprime = np.zeros(len(arms))
	#print(arms)
	for arm in arms:
		thetas_copy = copy.deepcopy(thetas)
		arm = int(arm)
		theta = thetas[arm]
		var = varis[arm]
		nSamples = 100
		ys = np.random.normal(theta,var + lambdaVar,nSamples)
		val_arm = 0
		for y in ys:
			theta_updated = (var*y + lambdaVar*theta)/(var + lambdaVar)
			thetas_copy[arm] = theta_updated
			val_arm += max(thetas_copy)
		thetaprime[arm] = val_arm/nSamples
	return thetaprime

if __name__=="__main__":
	H = 100
	num_arms = 10
	epoch = 1000
	v_ths = np.zeros(H)
	v_rands = np.zeros(H)
	v_ucbs = np.zeros(H)
	v_kgs = np.zeros(H)
	for i in range(0,H):
		print(i)
		v_th = sim(epoch, i, num_arms)
		v_rand = simRand(epoch, i, num_arms)
		v_ucb = simUCB(epoch,i,num_arms)
		v_kg = simKG(epoch,i,num_arms)
		v_ths[i] = v_th
		v_rands[i] = v_rand
		v_ucbs[i] = v_ucb
		v_kgs[i] = v_kg

	Harray = np.arange(1, H + 1, 1)
	fig, ax = plt.subplots()
	plt.plot(Harray, v_ths, 'r--', label='Thomson')
	plt.plot(Harray, v_ucbs, 'bs', label='UCB')
	plt.plot(Harray, v_rands, 'g^', label='Rand')
	plt.plot(Harray, v_kgs, 'k+', label='KG')
	plt.legend()
	plt.show()


	#x = np.linspace(1,H,H)
	#plt.plot(x,v_ths,'g^')
	#plt.plot(x,v_rands,'ro')
	#plt.plot(x,v_ucbs,'bs')
	#plt.show()






