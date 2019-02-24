import numpy as np  
from collections import defaultdict 
  
#Class to represent a graph 
class Graph: 
    # undirected graph. Do not use edge weights other than 1.
    
    def __init__(self,vertices): 
        self.V = vertices #No. of vertices 
        self.graph = [] # default dictionary  
                                # to store graph 
          
   
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph.append([u,v]) 
  
    # A utility function to find set of an element i 
    # (uses path compression technique) 
    def find(self, parent, i): 
        if parent[i] == i: 
            return i 
        return self.find(parent, parent[i]) 
  
    # A function that does union of two sets of x and y 
    # (uses union by rank) 
    def union(self, parent, rank, x, y): 
        xroot = self.find(parent, x) 
        yroot = self.find(parent, y) 
  
        # Attach smaller rank tree under root of  
        # high rank tree (Union by Rank) 
        if rank[xroot] < rank[yroot]: 
            parent[xroot] = yroot 
        elif rank[xroot] > rank[yroot]: 
            parent[yroot] = xroot 
  
        # If ranks are same, then make one as root  
        # and increment its rank by one 
        else : 
            parent[yroot] = xroot 
            rank[xroot] += 1
            
    def asEdgeList(self):
        # returns edge list, ignores weights
        edge_list = defaultdict(list)
        for u, v in self.graph:
            edge_list[u].append(v)
            edge_list[v].append(u)
        return edge_list
  
    # The main function to construct MST using Kruskal's  
        # algorithm 
    def KruskalMST(self): 
  
        result = [] #This will store the resultant MST 
  
        i = 0 # An index variable, used for sorted edges 
        e = 0 # An index variable, used for result[] 
  
            # Step 1:  Sort all the edges in non-decreasing  
                # order of their 
                # weight.  If we are not allowed to change the  
                # given graph, we can create a copy of graph 
        np.random.shuffle(self.graph)
  
        parent = [] ; rank = [] 
  
        # Create V subsets with single elements 
        for node in range(self.V): 
            parent.append(node) 
            rank.append(0) 
      
        # Number of edges to be taken is equal to V-1 
        while e < self.V -1 : 
  
            # Step 2: Pick the smallest edge and increment  
                    # the index for next iteration 
            u,v =  self.graph[i] 
            i = i + 1
            x = self.find(parent, u) 
            y = self.find(parent ,v) 
  
            # If including this edge does't cause cycle,  
                        # include it in result and increment the index 
                        # of result for next edge 
            if x != y: 
                e = e + 1     
                result.append([u,v]) 
                self.union(parent, rank, x, y)             
            # Else discard the edge 
  
        result_graph = Graph(self.V)
        for u, v in result:
            result_graph.addEdge(u, v)
        return result_graph

class ActionSpace(object):
    def __init__(self):
        # N, E, S, W in that order
        self.n = 4

    def possible_actions(self):
        pa = np.arange(0, self.n)
        np.random.shuffle(pa)
        return pa

    def sample(self):
        return np.random.randint(0, self.n - 1)

'''
class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        self.state_0 = np.random.randint(0, 50, shape, dtype=np.uint16)
        self.state_1 = np.random.randint(100, 150, shape, dtype=np.uint16)
        self.state_2 = np.random.randint(200, 250, shape, dtype=np.uint16)
        self.state_3 = np.random.randint(300, 350, shape, dtype=np.uint16)
        self.states = [self.state_0, self.state_1, self.state_2, self.state_3] 
'''
class ObservationSpace(object):
    def __init__(self, shape, n, v):
        self.shape = shape
        self.n = n
        self.verbose = v
        self.graph = self.initialize_graph().KruskalMST().asEdgeList()
        if self.verbose:
            print(self.graph)
        self.states = [self.generate_features(i) for i in range(int(self.n**2))]
        
    
    def generate_features(self, i):
        n = self.n
        features = np.zeros((3, 3))
        features[1,1] = 1.
        
        north = i - self.n
        if i == 0:
            north = None
        east = i + 1
        if i % self.n == n - 1:
            east = None
        south = i + self.n
        if i + self.n > int(self.n ** 2):
            south = None
        west = i - 1
        if i % self.n == 0:
            west = None
        
        if north is not None and north in self.graph[i]:
            features[0,1] = 1.
        if east is not None and east in self.graph[i]:
            features[1,2] = 1.
        if south is not None and south in self.graph[i]:
            features[2,1] = 1.
        if west is not None and west in self.graph[i]:
            features[1,0] = 1.
        
        '''
        ne = None
        if north is not None and east is not None:
            ne = north + 1
        se = None
        if south is not None and east is not None:
            se = south + 1
        sw = None
        if south is not None and west is not None:
            sw = south - 1
        nw = None
        if north is not None and west is not None:
            nw = north - 1
        
        if ne is not None and (ne in self.graph[north] or ne in self.graph[east]):
            features[0,2] = 1.
        if se is not None and (se in self.graph[south] or se in self.graph[east]):
            features[2,2] = 1.
        if sw is not None and (sw in self.graph[south] or sw in self.graph[west]):
            features[2,0] = 1.
        if nw is not None and (nw in self.graph[north] or nw in self.graph[west]):
            features[0,0] = 1.
        '''  
        
        if self.verbose:
            print(i)
            print(features)

        features = np.repeat(features, 3, axis=0)
        features = np.repeat(features, 3, axis=1)

        return np.expand_dims(features, -1)
        
        
    def initialize_graph(self):
        graph = Graph(int(self.n**2))
        mat = np.arange(int(self.n**2)).reshape((self.n, self.n))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i < self.n - 1:
                    graph.addEdge(mat[i,j], mat[i+1,j])
                if j < self.n - 1:
                    graph.addEdge(mat[i,j], mat[i,j+1])            
        if self.verbose:
            print(graph.asEdgeList())
        return graph
        


class EnvMaze(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified 
    """
    # note total nodes will be n squared
    def __init__(self, shape=(9, 9, 1), n=10, v=False):
        #4 states
        self.cur_state = 0
        self.num_iters = 0
        self.n = n
        self.visited = []
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(shape, n, v)

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.visited = []
        return self.observation_space.states[self.cur_state]

    def try_step(self, action):
        assert (0 <= action and action < 4 and action is not None)
        self.num_iters += 1
        if action == 0:
            ns = self.cur_state - self.n if self.cur_state - self.n >= 0 else self.cur_state
        elif action == 1:
            ns = self.cur_state + 1 if self.cur_state % self.n != self.n - 1 else self.cur_state
        elif action == 2:
            ns = self.cur_state + self.n if self.cur_state + self.n < int(self.n ** 2) else self.cur_state
        else:
            ns = self.cur_state - 1 if self.cur_state % self.n != 0 else self.cur_state
        if ns not in self.observation_space.graph[self.cur_state]:
            ns = self.cur_state
        return ns

    def step(self, action):
        ns = self.try_step(action)
        done = (ns == int(self.n**2) - 1)
        reward = 10. * float(done)
        self.cur_state = ns
        self.visited.append(self.cur_state)
        if len(self.visited) > int(self.n ** 3):
            done = True
            reward = 0.0
        return self.observation_space.states[self.cur_state], reward, done, {}


    def render(self):
        print(self.cur_state)
        
  
def test1():
    env = EnvMaze(n=3, v=True)
    
        
if __name__ == '__main__':
    test1()
    