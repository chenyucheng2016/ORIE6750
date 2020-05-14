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

    def genInitState(self):
        initState = []
        for p in self.init_belif:
            initState.append(round(p/self.belief_precision))
        for i in range(self.numEdges):
            initState.append(1)
        initState = np.int_(initState)
        return initState

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
                state = np.int_(state)
                controlSet = self.controlSet(state)
                maxVal = -99
                if len(controlSet) > 0:
                    for l in controlSet:
                        val = self.evalIntegral(state, l, h)
                        if val > maxVal:
                            maxVal = val
                    V[h] = maxVal
                else:
                    V[h] = self.EvalTerminalStateVal(state)
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
        prob_ob, ob_set = self.observationFnc(Lt,belief)
        val_int = 0
        for i in range(len(ob_set)):
            ob = ob_set[i]
            expectedVal = self.ExpectVal(state, Lt, ob, unCertain, belief, h+1)
            val_int = val_int + expectedVal * prob_ob[i]
        return val_int

    def observationFnc(self, Lt, belief):#generate all observations associated with one control input
        prob = []
        ob_set = []
        self.generateObservations([], ob_set)
        for i in range(len(ob_set)):
            ob_case = ob_set[i]
            product_ob = 1
            for j in range(len(ob_case)):
                ob = ob_case[j]
                person = Lt[j]
                product_ob = product_ob * (belief[person]**ob)*((1 - belief[j])**(1-ob))
            prob.append(product_ob)
        return prob, ob_set

    def generateObservations(self, ob_case, ob_set):
        if len(ob_case) == self.L:
            ob_set.append(ob_case)
            return
        else:
            for i in [0,1]:
                new_ob_case = deepcopy(ob_case)
                new_ob_case.append(i)
                self.generateObservations(new_ob_case, ob_set)


    def controlSet(self, state):
        belief = self.belief_discretization[np.array(state[0:self.numPeople])]
        unCertain = self.findUncertain(belief)
        return list(combinations(unCertain, self.L))

    def ExpectVal(self, state, Lt, observe, unCertain, belief, h):#expected value over dynamics given an observation
        #Lt is the index of the individual that is tested
        #observe is the test result {0,1}
        # updated by observation
        for i in range(len(Lt)):
            l = Lt[i]
            if l in unCertain:
                belief[l] = observe[i]
                unCertain.remove(l)
        infected = self.findInfected(belief)
        uninfected = self.findUninfected(belief)
        for person in infected:
            state[person] = round(1/self.belief_precision)
        for person in uninfected:
            state[person] = 0
        #uncertainty from self report
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
        expectedVal = 0
        for i in range(len(p_unCertain)):
            selfReport = p_unCertain[i]
            if not selfReport: #no self report
                new_state = self.transitionDynamics(state)
            else:
                for sr in selfReport:
                    belief[sr] = 1
                new_state = deepcopy(state)
                for j in range(len(belief)):
                    new_state[j] = round(belief[j]/self.belief_precision)
            new_state = self.isolateInfected(new_state)
            v = self.evalStateValue(new_state, h, 0, self.V)
            expectedVal = expectedVal + v * prob[i]
        return expectedVal

    def evalStateValue(self, state, h, l, V):
        #use recursion to evaluate state value
        if l == len(self.valueSize[0:-1]):
            return V[h]
        else:
            i = state[l]
            return self.evalStateValue(state, h , l+1, V[i])

    def isolateInfected(self, state):
        belief = self.belief_discretization[np.array(state[0:self.numPeople])]
        infected = self.findInfected(belief)
        edges2Remove = []
        edges2RemoveIdx = []
        for person in infected:
            neighbors = self.findNeighbors(state,person)
            for ne in neighbors:
                if person > ne:
                    edges2Remove.append([ne, person])
                else:
                    edges2Remove.append([person, ne])
        for e2m in edges2Remove:
            for i in range(len(self.edges)):
                eg = self.edges[i]
                if abs(e2m[0] - eg[0]) < 0.00001 and abs(e2m[1] - eg[1]) < 0.00001:
                    edges2RemoveIdx.append(i)
        for i in edges2RemoveIdx:
            state[int(i + self.numPeople)] = 0
        return state

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
    q = 0.1
    L = 1
    #case 1
    numPeople = 3
    H = 3
    nodes = np.linspace(0,numPeople-1,numPeople)
    nodes = np.int_(nodes)
    init_belief = np.array([2.0/3.0,1.0/2.0,1.0/2.0])
    adjMat = np.zeros((numPeople,numPeople))
    adjMat[0,1] = 1
    adjMat[0,2] = 1
    adjMat[1,0] = 1
    adjMat[2,0] = 1
    iPOMDP = InfectionPOMDP(init_belief, adjMat, p, q, L, H)
    iPOMDP.valueFunction()
    initState = iPOMDP.genInitState()
    v = iPOMDP.evalStateValue(initState, 0, 0, iPOMDP.V)
    print(v)

















