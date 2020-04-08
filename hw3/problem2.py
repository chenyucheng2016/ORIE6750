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
    alpha_range = np.arange(precision, alpha_max + precision, precision)
    beta_range = np.arange(precision, beta_max + precision, precision)
    alphaNum = int(floor(alpha_max / precision)) + 1
    maxState = encodeState(alpha_max, beta_max, precision, alphaNum) + 1
    minState = encodeState(precision, precision, precision, alphaNum)
    V = np.zeros((maxState, maxState, H))
    #A = np.zeros((maxState, maxState, H))
    for h in range(H - 1, -1, -1):
        for i in range(minState,maxState):
            for j in range(minState,maxState):
                alpha1, beta1 = decodeState(alphaNum, precision, i)
                alpha2, beta2 = decodeState(alphaNum, precision, j)
                #print(alpha2,beta2)
                if j != encodeState(alpha2,beta2,precision,alphaNum):
                    print(j)
                if h == (H-1):
                    V[i,j,h] = max(alpha1/(alpha1 + beta1), alpha2/(alpha2 + beta2))
                    #A[i,j,h] = np.argmax([alpha1/(alpha1 + beta1), alpha2/(alpha2 + beta2)])
                else:
                    params = [alpha1,beta1,alpha2,beta2]
                    V[i,j,h] = max(alpha1/(alpha1 + beta1) + evalIntegral(params,h,precision,V,alphaNum,maxState,minState,0),alpha2/(alpha2 + beta2) + evalIntegral(params,h,precision,V,alphaNum,maxState,minState,1))
                    #A[i,j,h] = np.argmax([alpha1/(alpha1 + beta1) + evalIntegral(params,h,precision,V,alphaNum,maxState,minState,0),alpha2/(alpha2 + beta2) + evalIntegral(params,h,precision,V,alphaNum,maxState,minState,1)])
    i_test = encodeState(1,1,precision,alphaNum)
    print(i_test)
    print(V[i_test,i_test,0])
    return V[minState:maxState-1,minState:maxState-1,0]


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
        if i >= maxState or j >= maxState or i <= minState or j <= minState:
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
        val = 0
        for j in range(H):
            sample0 = np.random.beta(params[0],params[1],1)
            sample1 = np.random.beta(params[2],params[3],1)
            arm = np.argmax([sample0,sample1])
            theta = max([sample0,sample1])
            y = np.random.binomial(1,theta,1)
            if arm == 0:
                params[0] = params[0] + y
                params[1] = params[1] + 1 - y
            elif arm == 1:
                params[2] = params[2] + y
                params[3] = params[3] + 1 - y
            val = val + y
        vals[i] = val
    return np.mean(vals)




if __name__=="__main__":
    k = 2
    precision = 0.5
    alpha_max = 35
    beta_max = 35
    H = 30
    V = valueFunction(precision,alpha_max,beta_max,H)
    #epoch = 5000
    #value = sim(epoch,H)
    #print((value))
    V = np.array(V)
    plt.imshow(V)
    plt.colorbar()
    plt.show()

