import numpy as np

class InfectionGraph:
    def __init__(self, nodes, adjMat, p,q,L):
        self.nodes = nodes
        self.adjMat = adjMat
        self.p = p #probability of infecting neighbors
        self.q = q #probability of reporting to public health
        self.L = L #number of tests each time
        self.numPeople = len(nodes)
        self.peopleInGraph = len(nodes)
        init_infected = np.random.randint(self.numPeople)
        self.infectedPeople = [init_infected]
    def findChildrenNodes(self,start):
        children = []
        for i in range(self.numPeople):
            if self.adjMat[start][i] > 0:
                children.append(i)
        return children
    def removeNode(self,infected):
        self.nodes.remove(infected)
        self.peopleInGraph = len(self.nodes)
        for i in range(self.numPeople):
            self.adjMat[infected][i] = 0
            self.adjMat[i][infected] = 0
    def isInfected(self,target):
        for infect in self.infectedPeople:
            if abs(infect - target) < 0.01:
                return 1
        return 0
    def evolveOneTimeStep(self, testPeople):
        for infect in self.infectedPeople:
            children = self.findChildrenNodes(infect)
            for child in children:
                if np.random.uniform(0,1,1) <= self.p and self.isInfected(child) == 0:
                    self.infectedPeople.append(child)
        for infect in self.infectedPeople:#remove the infected people who are self reported
            if np.random.uniform(0,1) <= self.q:
                self.infectedPeople.remove(infect)
        for person in testPeople:
            if self.isInfected(person):
                self.infectedPeople.remove(person)
if __name__=="__main__":
    numPeople = 50
    maxConnection = 6
    nodes = np.linspace(0,numPeople-1,numPeople)
    nodes = np.int_(nodes)
    adjMat = np.zeros((numPeople,numPeople))
    p = 0.5
    q = 0.5
    L = 10
    for i in nodes:
        connection = np.random.randint(int(maxConnection/2),maxConnection)
        children = np.random.choice(numPeople,connection,replace=False)
        for child in children:
            adjMat[i][child] = 1
    for i in nodes:
        for j in nodes:
            if adjMat[i][j] > 0:
                adjMat[j][i] = 1
    iGraph = InfectionGraph(nodes, adjMat, p, q, L)












