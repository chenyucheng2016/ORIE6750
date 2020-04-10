import numpy as np
from math import floor, gamma
import matplotlib.pyplot as plt

def encodeState(alpha,beta,precision,alphaNum):
    return floor((alpha - precision)/precision) + alphaNum*floor(beta/precision)
def decodeState(alphaNum,precision,i):
    alpha = i % alphaNum + 1
    beta =  floor(i / alphaNum)
    #print(i)
    #print(beta)
    return alpha*precision, beta*precision


def valueFunction(precision, alpha_max, beta_max, H):
    alphaNum = int(floor(alpha_max / precision)) + 1
    maxState = encodeState(alpha_max, beta_max, precision, alphaNum) + 1
    minState = encodeState(precision, precision, precision, alphaNum)
    V = np.zeros((maxState, maxState, H))
    for h in range(H - 1, -1, -1):
        for i in range(minState,maxState):
            for j in range(minState,maxState):
                alpha1, beta1 = decodeState(alphaNum, precision, i)
                alpha2, beta2 = decodeState(alphaNum, precision, j)
                if j != encodeState(alpha2,beta2,precision,alphaNum):
                    print(j)
                    print(encodeState(alpha2,beta2,precision,alphaNum))
                    print("\n")
                if h == (H-1):
                    V[i,j,h] = max(alpha1/(alpha1 + beta1), alpha2/(alpha2 + beta2))
                else:
                    params = [alpha1,beta1,alpha2,beta2]
                    V[i,j,h] = max(evalIntegral(params,h,precision,V,alphaNum,maxState,minState,0),evalIntegral(params,h,precision,V,alphaNum,maxState,minState,1))
                    if alpha1 == 1 and beta1 == 1 and alpha2 == 1 and beta2 == 1:
                        print(evalIntegral(params,h,precision,V,alphaNum,maxState,minState,0))
                        print(evalIntegral(params,h,precision,V,alphaNum,maxState,minState,1))
    i_test = encodeState(1,1,precision,alphaNum)
    vals_tests = np.zeros(H)
    for i in range(H - 1, -1, -1):
        vals_tests[H-1-i] = V[i_test,i_test,i]
    return vals_tests


def evalIntegral(params,h,precision,V,alphaNum,maxState,minState,arm):
    alpha1 = params[0]
    beta1 = params[1]
    alpha2 = params[2]
    beta2 = params[3]
    valueInt = 0
    for y in [0,1]:
        params_new = trainsitionDyn(params,y,arm)#arm = {0,1}
        i = encodeState(params_new[0], params_new[1], precision, alphaNum)
        j = encodeState(params_new[2], params_new[3], precision, alphaNum)
        if i >= maxState or j >= maxState or i < minState or j < minState:
            return 0
        if arm == 0:
            B_up = gamma(params_new[0])*gamma(params_new[1])/gamma(params_new[0]+params_new[1])
            B_down = gamma(params[0])*gamma(params[1])/gamma(params[0]+params[1])
        elif arm == 1:
            B_up = gamma(params_new[2]) * gamma(params_new[3]) / gamma(params_new[2] + params_new[3])
            B_down = gamma(params[2]) * gamma(params[3]) / gamma(params[2] + params[3])
        valueInt = valueInt + V[i,j,h+1]*B_up/B_down
    return valueInt


def trainsitionDyn(params,y,arm):
    if arm == 0:
        return [params[0] + y, params[1] + 1 - y, params[2], params[3]]
    elif arm == 1:
        return [params[0], params[1], params[2] + y, params[3] + 1 - y]
def sim(epoch,H):
    vals = np.zeros(epoch)
    for i in range(epoch):
        params = [1,1,1,1]
        theta0 = np.random.beta(params[0], params[1], 1)
        theta1 = np.random.beta(params[2], params[3], 1)
        thetas = [theta0,theta1]
        #val = 0
        for j in range(H):
            sample0 = np.random.beta(params[0],params[1],1)
            sample1 = np.random.beta(params[2],params[3],1)
            arm = np.argmax([sample0,sample1])
            theta = thetas[arm]
            y = np.random.binomial(1,theta,1)
            if arm == 0:
                params[0] = params[0] + y
                params[1] = params[1] + 1 - y
            elif arm == 1:
                params[2] = params[2] + y
                params[3] = params[3] + 1 - y
            #val = val + y
        vals[i] = max(params[0]/(params[0] + params[1]), params[2]/(params[2] + params[3]))
    return np.mean(vals)




if __name__=="__main__":
    k = 2
    precision = 1
    alpha_max = 52
    beta_max = 52
    H = 50
    vals_test = valueFunction(precision,alpha_max,beta_max,H)
    vals_sim = np.zeros(H)
    epoch = 3000
    for h in range(H):
        value = sim(epoch,h)
        vals_sim[h] = value
    vals_test = np.array(vals_test)
    vals_sim = np.array(vals_sim)
    Harray = np.arange(1,H+1,1)
    fig, ax = plt.subplots()
    plt.plot(Harray,vals_test,'r--',label='opt')
    plt.plot(Harray,vals_sim,'bs',label='sim')
    plt.legend()
    plt.show()


