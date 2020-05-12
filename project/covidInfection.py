import numpy as np
from copy import deepcopy
from itertools import combinations
class InfectionPOMDP:
    def __init__(self,init_belief, adjMat, p, q, L, H):
        self.init_belif = init_belief
        self.adjMat = adjMat
        self.p = p
        self.q = q
        self.L = L
        self.H = H
        self.numPeople = len(init_belief)
        self.belief_discretization = np.array([0, 0.25, 0.5, 0.75, 1])
        self.belief_precision = 0.25
        self.numDiscretization = len(self.belief_discretization)
        self.edges = self.findOriginalEdges()
        self.numEdges = len(self.edges)
        valueSize = self.numDiscretization*np.ones((1,self.numPeople))
        valueSize = np.append(valueSize, 2*np.ones((1,self.numEdges)))
        valueSize = np.append(valueSize, H)
        self.valueSize = np.int_(valueSize)
        self.V = np.zeros(self.valueSize)# first numPeople dimentions are for nodes and the next numEdges dimentions are for social connections

    def findOriginalEdges(self):
        edges = []
        for i in range(self.numPeople):
            for j in range(i,self.numPeople):
                if self.adjMat[i,j] > 0:
                    edges.append([i,j])
        return edges

    def EvalTerminalStateVal(self, state):
        belief = self.belief_discretization[np.array(state[0:self.numPeople])]
        unInfected = self.findUninfected(belief)
        return len(unInfected)

    def findNeighbors(self, state, person):
        connections = state[-self.numEdges:]
        neighbors = []
        for i in range(len(connections)):
            conn = connections[i]
            if conn > 0:
                edge = self.edges[i]
                if abs(person - edge[0]) < 0.00001:
                    neighbors.append(edge[1])
                elif abs(person - edge[1]) < 0.00001:
                    neighbors.append(edge[0])
        return neighbors

    def findUninfected(self, belief):
        Uninfected = []
        for i in range(len(belief)):
            person = belief[i]
            if abs(person) < 0.00001:
                Uninfected.append(i)
        return Uninfected

    def findInfected(self, belief):
        Infected = []
        for i in range(len(belief)):
            person = belief[i]
            if abs(person - 1) < 0.00001:
                Infected.append(i)
        return Infected

    def findUncertain(self, belief):
        Uncertain = []
        for i in range(len(belief)):
            person = belief[i]
            if (abs(person - 1) > 0.00001 and abs(person) > 0.00001):
                Uncertain.append(i)
        return Uncertain

    def stateSearch(self, state, h, l, V):
        if l == len(self.valueSize[0:-1]):
            if h == (self.H-1):
                V[h] = self.EvalTerminalStateVal(np.int_(state))

        else:
            dim = self.valueSize[l]
            for i in range(dim):
                state[l] = i
                self.stateSearch(state, h, l+1, V[i])

    def valueFunction(self):
        for h in range(self.H-1, -1, -1):
            self.stateSearch(-np.ones(self.numPeople + self.numEdges), h, 0, self.V)
    def transitionDynamics(self, state):#no self report
        belief = self.belief_discretization[np.array(state[0:self.numPeople])]
        #print(belief)
        #print(state)
        new_state = deepcopy(state)
        for i in range(self.numPeople):
            neighbors = self.findNeighbors(state, i)
            if len(neighbors) == 0:
                new_state[i] = round(belief[i] / self.belief_precision)
            else:
                product_neighbor = 1
                for ne in neighbors:
                    product_neighbor = product_neighbor * (1 - self.p*belief[ne])
                new_state[i] = round((belief[i] + (1 - belief[i])*(1 - product_neighbor))/self.belief_precision)
        return np.int_(new_state)
    def evalIntegral(self, state, Lt, h):
        belief = self.belief_discretization[np.array(state[0:self.numPeople])]
        unCertain = self.findUncertain(belief)
        self.ExpectVal(state,1,unCertain,belief)
    def controlSet(self, unCertain):
        return list(combinations(unCertain, self.L))

    def ExpectVal(self, state, Lt, observ, unCertain, belief):#expected value over dynamics given an observation
        #Lt is the index of the individual that is tested
        #observ is the test result
        p_unCertain = self.generatePowerSet(unCertain)
        prob = np.zeros(len(p_unCertain))
        for i in range(len(p_unCertain)):
            if i <= 0:
                continue
            else:
                subset = p_unCertain[i]
                product_self_report = 1
                for person in subset:
                    product_self_report = product_self_report * self.q * belief[person]
                prob[i] = product_self_report
        prob[0] = 1 - np.sum(prob[1:])
        print(prob)
        print(np.sum(prob))

    def generatePowerSet(self, set):
        setSize = len(set)
        pSet = [[]]
        for i in range(1,setSize+1):
            comb = combinations(set , i)
            for c in list(comb):
                pSet.append(list(c))
        return pSet


if __name__=="__main__":
    p = 0.3
    q = 0.5
    L = 1
    #case 1
    numPeople = 3
    H = 4
    nodes = np.linspace(0,numPeople-1,numPeople)
    nodes = np.int_(nodes)
    init_belief = np.array([2.0/3.0,1.0/2.0,1.0/2.0])
    adjMat = np.zeros((numPeople,numPeople))
    adjMat[0,1] = 1
    adjMat[0,2] = 1
    adjMat[1,0] = 1
    adjMat[2,0] = 1
    iPOMDP = InfectionPOMDP(init_belief, adjMat, p, q, L, H)
    state = [1,1,1,1,1]
    iPOMDP.evalIntegral(state,1,1)















