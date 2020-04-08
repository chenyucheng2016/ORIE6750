import numpy as np
import matplotlib.pyplot as plt
def evalInegral(p,V,h,precision):
    i = p[0]
    j = p[1]
    valueInt = 0
    for y in [0, 1]:
        p = transitionDyn([i,j],precision,y)
        valueInt = valueInt + V[p[0],p[1],h+1]*(i*precision*lik1(y) + j*precision*lik2(y) + (1 - i*precision - j*precision)*lik0(y))
    return valueInt
def transitionDyn(p,precision,y):
    i = p[0]
    j = p[1]
    p_new0 = i*precision*lik1(y)/(i*precision*lik1(y) + j*precision*lik2(y) + (1 - i*precision - j*precision)*lik0(y))
    p_new1 = j*precision*lik2(y)/(i*precision*lik1(y) + j*precision*lik2(y) + (1 - i*precision - j*precision)*lik0(y))
    i_new = int(round(p_new0/precision))
    j_new = int(round(p_new1/precision))
    return [i_new, j_new]

def lik0(y):
    if y == 1:
        return 0.5
    elif y == 0:
        return 0.5
    else:
        return -1
def lik1(y):
    if y == 1:
        return 0.333
    elif y == 0:
        return 0.667
    else:
        return -1
def lik2(y):
    if y == 1:
        return 0.667
    elif y == 0:
        return 0.333
    else:
        return -1
def valueFunction(precision, H):
    numCell = int(1/precision)
    print(numCell)
    V = np.zeros((numCell,numCell,H))
    c = 0.01
    A = np.zeros((numCell,numCell,H))
    for h in range(H-1,-1,-1):
        for i in range(numCell):
            for j in range(numCell):
                if i*precision + j*precision <= 1:
                    if h == (H-1):
                        V[i, j, h] = min(1 - i*precision, 1 - j*precision, j*precision + i*precision)
                    else:
                        V[i, j, h] = min(1 - i*precision, 1 - j*precision, j*precision + i*precision,  c + evalInegral([i, j], V, h, precision))
                        A[i, j,  h] = np.argmin([1 - i*precision, 1 - j*precision, j*precision + i*precision,  c + evalInegral([i, j], V, h, precision)])
                else:
                    V[i, j, h] = -1
                    A[i, j, h] = -1
    return V[:,:,0], A[:,:,0]
if __name__=="__main__":
    precision = 0.005
    H = 50
    V, A = valueFunction(precision, H)
    V = np.array(V)
    A = np.array(A)
    plt.imshow(A)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
