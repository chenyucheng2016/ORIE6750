import numpy as np
import copy
import matplotlib.pyplot as plt
def sim(epoch,H):
	mu1 = 0
	sigma1 = 1
	mu2 = 0
	sigma2 = 1
	lambdaVar = 1
	v = np.zeros(epoch)
	for i in range(epoch):
		x1 = np.random.normal(mu1, sigma1, 1)
		x2 = np.random.normal(mu2, sigma2, 1)
		theta1 = 0
		var1 = 1
		theta2 = 0
		var2 = 1
		for h in range(H):
			sample1 = np.random.normal(theta1, var1, 1)
			sample2 = np.random.normal(theta2, var2, 1)
			arm = np.argmax([sample1,sample2])
			if arm == 0:
				y = np.random.normal(x1,lambdaVar,1)
				theta1 = (var1*y + lambdaVar*theta1)/(var1 + lambdaVar)
				var1 = var1 * lambdaVar / (var1 + lambdaVar)
			elif arm == 1:
				y = np.random.normal(x2,lambdaVar,1)
				theta2 = (var2*y + lambdaVar*theta2)/(var2 + lambdaVar)
				var2 = var2 * lambdaVar / (var2 + lambdaVar)
		v[i] = max(theta1,theta2)
	return np.mean(v)

def simKG(epoch, H):
	mu1 = 0
	sigma1 = 1
	mu2 = 0
	sigma2 = 1
	lambdaVar = 1
	v = np.zeros(epoch)
	for i in range(epoch):
		x1 = np.random.normal(mu1, sigma1, 1)
		x2 = np.random.normal(mu2, sigma2, 1)
		theta1 = 0
		var1 = 1
		theta2 = 0
		var2 = 1
		for h in range(H):
			thetas = [theta1,theta2]
			thetaprimes = thetaprime([theta1,theta2], [var1,var2],lambdaVar,[0,1])
			arm = np.argmax(thetaprimes)
			if arm == 0:
				y = np.random.normal(x1,lambdaVar,1)
				theta1 = (var1*y + lambdaVar*theta1)/(var1 + lambdaVar)
				var1 = var1 * lambdaVar / (var1 + lambdaVar)
			elif arm == 1:
				y = np.random.normal(x2,lambdaVar,1)
				theta2 = (var2*y + lambdaVar*theta2)/(var2 + lambdaVar)
				var2 = var2 * lambdaVar / (var2 + lambdaVar)
		v[i] = max(theta1,theta2)
	return np.mean(v)
def thetaprime(thetas, varis, lambdaVar, arms):
	thetaprime = np.zeros(len(arms))
	for arm in arms:
		thetas_copy = copy.deepcopy(thetas)
		theta = thetas[arm]
		var = varis[arm]
		nSamples = 1000
		ys = np.random.normal(theta,var + lambdaVar,nSamples)
		val_arm = 0
		for y in ys:
			theta_updated = (var*y + lambdaVar*theta)/(var + lambdaVar)
			thetas_copy[arm] = theta_updated
			val_arm += max(thetas_copy)
		thetaprime[arm] = val_arm/nSamples




if __name__=="__main__":
	epoch = 10000
	H = 10
	v_KGs = np.zeros(H)
	for h in range(H):
		vKG = simKG(epoch, h)
		v_KGs[h] = vKG
	vals_test = np.array([0.0, 0.406, 0.5631, 0.6137, 0.6596, 0.6651, 0.6947, 0.7113, 0.7143, 0.7283])
	Harray = np.arange(1, H + 1, 1)
	fig, ax = plt.subplots()
	plt.plot(Harray, vals_test, 'r--', label='opt')
	plt.plot(Harray, v_KGs, 'bs', label='KG')
	plt.legend()
	plt.show()




