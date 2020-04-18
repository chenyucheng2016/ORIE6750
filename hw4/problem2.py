import numpy as np
import copy
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
		var2 = 0
		for h in range(H):
			arm = np.argmax([theta1,theta2])
			if arm == 0:
				y = np.random.normal(x1,lambdaVar,1)
				theta1 = (var1*y + lambdaVar*theta1)/(var1 + lambdaVar)
				var1 = var1 * lambdaVar / (var1 + lambdaVar)
			elif arm == 1:
				y = np.random.normal(x2,lambdaVar,1)
				theta1 = (var2*y + lambdaVar*theta2)/(var2 + lambdaVar)
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
		var2 = 0
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
				theta1 = (var2*y + lambdaVar*theta2)/(var2 + lambdaVar)
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
	epoch = 50000
	H = 9
	#v = sim(epoch, H)
	vKG = simKG(epoch, H)
	#print(v)
	print(vKG)



