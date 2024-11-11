from collections import defaultdict
from collections import deque
from heapq import heappop, heappush
from typing import List,Dict,Tuple,Any


graph=defaultdict(list)
heuristics=defaultdict(list)
weights=defaultdict(list)
graph={
    'S':['A','B'],
    'A':['S','B','D'],
    'B':['S','A','C'],
    'C':['B','E'],
    'D':['A','G'],
    'E':['C'],
    'G':['D']
}

heuristics = {
    'S': 8,
    'A': 6,
    'B': 6,
    'C': 4,
    'D': 4,
    'E': 2,
    'G': 0
}

weights={
    ('S','A'):3,
    ('S','B'):5,
    ('A','B'):4,
    ('B','C'):3,
    ('A','D'):3,
    ('C','E'):6,
    ('D','G'):5
}

and_nodes={
    'B'
}

oracle_path=['S','A','D','G']
oracle_cost=11

start_node='S'
goal_node='G'
w=2

MAX,MIN=1000,-1000


def bms(graph:Dict[str,List],start:str,goal:str)->List[str]:
    
    #Keeps track of all possible paths
    all_paths=[]
    current_path=[start]

    def find_all_paths(current_path:List[str])->None:
        current_node=current_path[-1]
        if current_node==goal:
            all_paths.append(current_path.copy())
            return        
        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                current_path.append(neighbour)
                find_all_paths(current_path)
                current_path.pop() #helps search all possible paths (backtracking)

    find_all_paths(current_path)

    #     #shortest path
    # if all_paths:
    #     shortest_path=min(all_paths,key=len) #min(length of list in all_paths)
    #     return shortest_path
    # else:
    #     return None
    print(all_paths)

def bfs(graph:Dict[str,List],start:str,goal:str)->List[str]:
    queue=deque([[start]])

    while queue:
        current_path=queue.popleft()
        current_node=current_path[-1]

        if current_node==goal:
            return current_path
        
        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                queue.append(new_path)
    return None

def dfs(graph:Dict[str,List],start:str,goal:str)-> List[str]:

    ds=deque([[start]])
    #stack=[[start]]

    while ds:
        current_path=ds.pop() #popleft bfs
        current_node=current_path[-1]

        if current_node==goal:
            return current_path
        
        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                ds.append(new_path)
    
    return None

def hill_climb(graph:Dict[str,List],start:str,goal:str,heuristic:Dict[str,int])->List[str]:

    current_node=start
    path=[current_node]
    # print("Path:",path)

    while current_node!=goal:
        neighbours=graph.get(current_node,[])
        # print("Curr:",current_node)
        # print("Neighbours:",neighbours)

        if not neighbours:
            return None

        next_node=min(neighbours,key= lambda x:heuristic.get(x,float('inf')))
        # print("Next Node:",next_node)
        
        if heuristic.get(next_node,float('inf')) >= heuristic.get(current_node,[]):
            break

        current_node=next_node
        # print("Curr after",current_node)
        path.append(current_node)
        # print("Path after:",path)
    
    return path

def beam(graph:Dict[str,List],start:str,goal:str,heuristics:Dict[str,int],w:int)->List[str]:

    current_nodes=[(heuristics[start],[start])]

    while current_nodes:
        new_nodes=[]

        for h, current_path in sorted(current_nodes)[:w]:
            current_node=current_path[-1]
            # print("Current Node:",current_node)

            if current_node==goal:
                return current_path
            
            for neighbour in graph.get(current_node):
                if neighbour not in current_path:
                    new_path=current_path+[neighbour]
                    new_h=sum(heuristics.get(node,0)for node in new_path)-heuristics.get(start,0)
                    heappush(new_nodes,(new_h,new_path))
        current_nodes=new_nodes
        # print("Current Node:",current_nodes)

    return None

def a_star(graph:Dict[str,List],start:str,goal:str,heuristic:Dict[str,int],weights:Dict[Tuple[str],int])->List[str]:
    pq=[(heuristic[start],0,[start])]

    while pq:
        a,cost,current_path=heappop(pq)
        current_node=current_path[-1]
        visited=set()

        if current_node==goal:
            return current_path
        
        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                new_cost=cost+weights.get((current_node,neighbour),0)
                new_h=new_cost+heuristic.get(neighbour)
                heappush(pq,(new_h,new_cost,new_path))
    
    return None

def bb(graph:Dict[str,List],start:str,goal:str,weights:Dict[Tuple[str],int])->List[str]:
    pq=[(0,[start])]

    while pq:
        cost,current_path=heappop(pq)
        current_node=current_path[-1]

        if current_node==goal:
            return current_path
        
        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                new_cost=cost+weights.get((current_node,neighbour),0)
                heappush(pq,(new_cost,new_path))
    return None

def best_first(graph:Dict[str,List],start:str,goal:str,heuristic:Dict[str,int],weights:Dict[Tuple[str],int])->List[str]:

    pq=[(heuristic[start],[start])]
    visited=set()

    while pq:
        cost,current_path=heappop(pq)
        current_node=current_path[-1]

        if current_node==goal:
            return current_path
        
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                new_cost=heuristic.get(neighbour)
                heappush(pq,(new_cost,new_path))

    return None

def oracle_search_with_h(graph:Dict[str,List],start:str,goal:str,weights:Dict[Tuple[str],int],oracle_path,oracle_cost)->List[str]:
    pq=[(0,[start])]
    best_path=oracle_path
    min_cost=oracle_cost

    while pq:
        cost,current_path=heappop(pq)
        current_node=current_path[-1]

        if current_node==goal:
            if cost< min_cost:
                best_path,min_cost=current_path,cost
            continue
        for neighbour in graph.get(current_node,[]):
            if neighbour not in current_path:
                new_path=current_path+[neighbour]
                new_cost=cost+weights.get((current_node,neighbour),1)
                if new_cost < min_cost:
                    heappush(pq,(new_cost,new_path))
    return best_path

def ao_star(graph: Dict[str, List[str]], start: str, goal: str, weights: Dict[Tuple[str, str], int], and_nodes: Set[str]) -> List[str]:
    def calculate_cost(node: str, visited: Set[str]) -> Tuple[float, List[str]]:
        if node == goal:
            return 0, [node]
        if node in visited:
            return float('inf'), []
        visited.add(node)
        if node in and_nodes:
            total_cost = 0
            total_path = [node]
            for neighbor in graph.get(node, []):
                cost, path = calculate_cost(neighbor, visited.copy())
                total_cost += cost + weights.get((node, neighbor), 1)
                total_path.extend(path)
            return total_cost, total_path
        else:
            min_cost = float('inf')
            best_path = []
            for neighbor in graph.get(node, []):
                cost, path = calculate_cost(neighbor, visited.copy())
                total_cost = cost + weights.get((node, neighbor), 1)
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = [node] + path
            return min_cost, best_path
    _, path = calculate_cost(start, set())
    return path if path else []