import numpy as np

class InfectionGraph:#all information about a state
    def __init__(self, nodes, adjMat, p, q, L, init_belief):
        self.nodes = nodes
        self.numPeople = len(nodes)
        self.adjMat = adjMat #initial social connections for the individuals
        self.p = p #probability of infecting neighbors
        self.q = q #probability of reporting to public health
        self.L = L #number of tests each time
        self.init_belief = init_belief #initial belief
    def findChildrenNodes(self,start):
        children = []
        for i in range(self.numPeople):
            if self.adjMat[start][i] > 0:
                children.append(i)
        return children
    def isolateNode(self,infected):
        for i in range(self.numPeople):
            self.adjMat[infected][i] = 0
            self.adjMat[i][infected] = 0
    def isInfected(self,target):
        for infect in self.infectedPeople:
            if abs(infect - target) < 0.01:
                return 1
        return 0
def findEdges(adjMat,numPeople):
    edges = []
    for i in range(numPeople):
        for j in range(i,numPeople):
            if adjMat[i,j] > 0:
                edges.append([i,j])
    return edges

def valueFunction(init_belief, adjMat, p ,q, L):
    numPeople = len(init_belief)
    belief_discretization = [0,0.25,0.5,0.75,1]
    edges = findEdges(adjMat,numPeople)
    numEdges = len(edges)
    


if __name__=="__main__":
    p = 0.5
    q = 0.5
    L = 1
    #case 1
    numPeople = 3
    nodes = np.linspace(0,numPeople-1,numPeople)
    nodes = np.int_(nodes)
    init_belief = np.array([2.0/3.0,1.0/2.0,1.0/2.0])
    adjMat = np.zeros((numPeople,numPeople))
    adjMat[0,1] = 1
    adjMat[0,2] = 1
    adjMat[1,0] = 1
    adjMat[2,0] = 1
    V = valueFunction(init_belief, adjMat, p, q, L)












