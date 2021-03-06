import numpy as np  
from collections import defaultdict
import matplotlib.pyplot as plt
#from graphviz import Graph as GraphVizGraph

from tqdm import tqdm
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from queue import Queue

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

    def bfs_paths(self, start, goal):
        g = self.asEdgeList()
        queue = [(start, [start])]
        while queue:
            (vertex, path) = queue.pop(0)
            for next in set(g[vertex]) - set(path):
                if next == goal:
                    yield path + [next]
                else:
                    queue.append((next, path + [next]))

    def shortest_path(self, start, goal):
        try:
            return next(self.bfs_paths(start, goal))
        except StopIteration:
            return None

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
def generate_hard_maze(n):
    nodes = np.arange(int(n ** 2))
    g = defaultdict(list)

    for node in nodes:
        if node == int(n ** 2) - 1:
            break
        if node * 2 + 1 <= n ** 2 - 1:
            g[node].append(node * 2 + 1)
        else:
            g[node].append(0)
        if node * 2 + 2 <= n ** 2 - 1:
            g[node].append(node * 2 + 2)
        else:
            g[node].append(0)
        g[node] += [0, 0]
        np.random.shuffle(g[node])

    return g


class ObservationSpace(object):
    def __init__(self, shape, n, v, hard):
        self.shape = shape
        self.n = n
        self.verbose = v
        self.hard = hard
        if hard:
            self.graph = generate_hard_maze(n)
        else:
            self.graph = self.initialize_graph().KruskalMST().asEdgeList()
        #self.save_graph()
        if self.verbose:
            print(self.graph)
        self.states = [self.generate_features(i) for i in range(int(self.n**2))]
        
    def save_graph(self):
        g = GraphVizGraph('Maze')
        for state in self.graph.keys():
            for n in self.graph[state]:
                g.edge(str(state), str(n))
        #g.view()

    def generate_features(self, i):
        n = self.n
        features = np.zeros((3, 3))
        features[1,1] = 1.
        
        if self.hard:
            features[0, 1] = 1.
            features[2, 1] = 1.
            features[1, 0] = 1.
            features[1, 2] = 1.
        else:
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
    def __init__(self, shape=(9, 9, 1), n=10, v=False, hard=False):
        #4 states
        self.cur_state = 0
        self.num_iters = 0
        self.n = n
        self.reward_scale = 1.0
        self.hard = hard
        self.visited = []
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(shape, n, v, hard)

    def get_bfs_length(self):
        return 0 #len(self.observation_space.graph_obj.shortest_path(0, int(self.n ** 2) - 1))

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.visited = []
        return self.observation_space.states[self.cur_state]

    def check_next_step(self, action):
        assert (0 <= action and action < 4 and action is not None)
        if self.hard:
            ns = self.observation_space.graph[self.cur_state][action]
        else:
            if action == 0:
                ns = self.cur_state - self.n if self.cur_state - self.n >= 0 else self.cur_state
            elif action == 1:
                ns = self.cur_state + 1 if self.cur_state % self.n != self.n - 1 else self.cur_state
            elif action == 2:
                ns = self.cur_state + self.n if self.cur_state + self.n < int(self.n ** 2) else self.cur_state
            else:
                ns = self.cur_state - 1 if self.cur_state % self.n != 0 else self.cur_state
        return ns

    def try_step(self, action):
        ns = self.check_next_step(action)
        if ns not in self.observation_space.graph[self.cur_state]:
            ns = self.cur_state
        return ns

    def try_and_penalize_step(self, action):
        ns = self.check_next_step(action)
        reward = 0.
        if ns not in self.observation_space.graph[self.cur_state]:
            ns = self.cur_state
            reward = -0.01
        return ns, reward

    def step(self, action):
        self.num_iters += 1
        ns, reward = self.try_and_penalize_step(action)
        done = (ns == int(self.n**2) - 1)
        if reward == 0.:
            reward = self.reward_scale * float(done)
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

def test2():
    env = EnvMaze(n=10, hard=True)
    print('Hard Maze')
    N_TRIALS = 10000
    rewards = []
    completions = []
    for _ in tqdm(range(N_TRIALS)):
        reward = 0
        done = False
        env.reset()
        for i in range(int(env.n ** 3)):
            a = env.action_space.sample()
            _, r, done, _ = env.step(a)
            reward += r
            if done:
                break
        rewards += [reward]
        completions += [int(done)]

    print('Reward = {} +/- {}'.format(np.round(np.mean(rewards), 4), np.round(np.std(rewards), 4)))
    print('Completion Rate = {} +/- {}'.format(np.round(np.mean(completions), 4), np.round(np.std(completions), 4)))
    print()


def test3():
    n = 10
    print('Kruskal Maze')
    N_TRIALS = 10000
    rewards = []
    completions = []
    for _ in range(N_TRIALS):
        env = EnvMaze(n=n)
        reward = 0
        done = False
        for i in range(int(env.n ** 3)):
            a = env.action_space.sample()
            _, r, done, _ = env.step(a)
            reward += r
            if done:
                break
        rewards += [reward]
        completions += [int(done)]

    print('Reward = {} +/- {}'.format(np.round(np.mean(rewards), 4), np.round(np.std(rewards), 4)))
    print('Completion Rate = {} +/- {}'.format(np.round(np.mean(completions), 4), np.round(np.std(completions), 4)))
    print()



    
        
if __name__ == '__main__':
    test1()
    test2()
    test3()
