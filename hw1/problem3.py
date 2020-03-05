import numpy as np

def BackTrack(numStates, H):
	V_table = -np.ones((H,numStates))
	A_table = -np.ones((H,numStates))
	for i in range(H):
		for j in range(numStates):
			if i == 0:
				if j == 9:
					V_table[i,j] = 1
				else:
					V_table[i,j] = 0
			else:
				if j == 9:
					V_table[i,j] = 1
					A_table[i,j] = 0
				else:
					V_array = np.array([V_table[i-1,j+1],1.0/2.0*V_table[i-1,min(j+2,9)] + 1.0/2.0*V_table[i-1,max(0,j-1)]])
					A_table[i,j] = np.argmax(V_array)
					V_table[i,j] = V_array[int(A_table[i,j])]
	return V_table, A_table





if __name__=="__main__":
	V, A = BackTrack(10,9)
	print(V)
	print(A)
	
	
