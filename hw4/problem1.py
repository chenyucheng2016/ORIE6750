import numpy as np
from math import floor, gamma
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
def encodeState(theta, mNum, discrete_theta, thetaNum):
    return int(floor((theta - discrete_theta)/discrete_theta) + thetaNum*mNum)
def decodeState(thetaNum,discrete_theta,i):
    theta = i % thetaNum + 1
    mNum =  floor(int(i) / int(thetaNum))
    return theta*discrete_theta, mNum


def valueFunction(discrete_theta, theta_max, measure_max, H):
    thetaNum = int(floor(2*theta_max / discrete_theta))
    maxState = int(encodeState(2*theta_max, measure_max, discrete_theta, thetaNum)) + 1
    minState = int(encodeState(discrete_theta, 0, discrete_theta, thetaNum))
    V = np.zeros((maxState, maxState, H))
    i_test = encodeState(theta_max,0,discrete_theta,thetaNum)
    for h in range(H - 1, -1, -1):
        print(h)
        print("mc2")
        for i in range(minState,maxState):#theta
            for j in range(minState,maxState):#var
                theta1, mNum1 = decodeState(thetaNum, discrete_theta, i)
                theta2, mNum2 = decodeState(thetaNum, discrete_theta, j)
                #print(theta2, mNum2)
                if j != encodeState(theta2, mNum2, discrete_theta, thetaNum):
                    print(j)
                    print(encodeState(theta2, mNum2, discrete_theta, thetaNum))
                    print("\n")
                if h == (H-1):
                    V[i,j,h] = max(theta1, theta2) - theta_max
                else:
                    params = [theta1,mNum1,theta2,mNum2]
                    V[i,j,h] = max(evalIntegralMC(params, h, discrete_theta, V, thetaNum, maxState, minState, 0, theta_max),
                                   evalIntegralMC(params, h, discrete_theta, V, thetaNum, maxState, minState, 1, theta_max))
            
    val_intest = []

    for h in range(H):
        val_intest.append(V[i_test,i_test,h])
    print(val_intest)
    return V


def evalIntegralMC(params, h, discrete_theta, V, thetaNum, maxState, minState, arm, theta_max):
    theta1 = params[0]
    mNum1 = params[1]
    theta2 = params[2]
    mNum2 = params[3]
    valueInt = 0
    nSamples = 1000
    counter = 0
    #print(V[:,:,h+1])
    #time.sleep(20)
    if arm == 0:
        ys = np.random.normal(params[0] - theta_max, 1.0/(params[1] + 1) + 1, nSamples)
        for y in ys:
            params_new = trainsitionDyn(params, y, arm, discrete_theta, theta_max)#arm = {0,1}
            i = encodeState(params_new[0], params_new[1], discrete_theta, thetaNum) 
            j = encodeState(params_new[2], params_new[3], discrete_theta, thetaNum)
            if i >= maxState or j >= maxState:
                print("exceed")
                continue
            else:
                counter += 1
                valueInt = valueInt + V[i,j,h+1]
        return valueInt/counter
    elif arm == 1:
        ys = np.random.normal(params[2] - theta_max, 1.0/(params[3] + 1) + 1, nSamples)
        for y in ys:
            params_new = trainsitionDyn(params, y, arm, discrete_theta, theta_max)#arm = {0,1}
            i = encodeState(params_new[0], params_new[1], discrete_theta, thetaNum)
            j = encodeState(params_new[2], params_new[3], discrete_theta, thetaNum)
            if i >= maxState or j >= maxState:
                continue
            else:
                counter += 1
                valueInt = valueInt + V[i,j,h+1]
        return valueInt/counter


def trainsitionDyn(params, y, arm, discrete_theta, theta_max):
    if arm == 0:
        oldprecision = 1 + params[1]
        return [round((oldprecision*(params[0] - theta_max)+ y)/(oldprecision + 1)/discrete_theta)*discrete_theta + theta_max, params[1] + 1, params[2], params[3]]
    elif arm == 1:
        oldprecision = 1 + params[3]
        return [params[0], params[1],round((oldprecision*(params[2] - theta_max) + y)/(oldprecision + 1)/discrete_theta)*discrete_theta + theta_max, params[3] + 1]
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

if __name__=="__main__":
    k = 2
    discrete_theta = 0.2
    measure_max = 9
    theta_max = 10
    H = 10
    #vals = valueFunction(discrete_theta, theta_max, measure_max, H)
    vals_sim = np.zeros(H)
    epoch = 10000
    for h in range(H):
        value = sim(epoch,h)
        vals_sim[h] = value
    vals_test = np.array([0.0,0.406,0.5631,0.6137,0.6596,0.6651,0.6947,0.7113,0.7143,0.7283])
    vals_sim = np.array(vals_sim)
    Harray = np.arange(1,H+1,1)
    fig, ax = plt.subplots()
    plt.plot(Harray,vals_test,'r--',label='opt')
    plt.plot(Harray,vals_sim,'bs',label='sim')
    plt.legend()
    plt.show()
    

