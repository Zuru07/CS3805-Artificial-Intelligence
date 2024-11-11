game_tree={
    'A':['B','C'],
    'B':['D','E'],
    'C':['F','G'],
    'D': 3,
    'E': 5,
    'F': 2,
    'G': 9
}

ALPHA,BETA=-1000,1000

def minmax(node:str,depth:int,maximising_player:bool)->int:
    
    # Terminal node -> Checks whether the node is a terminal node and returns a boolean value
    if isinstance(game_tree[node],int):
        return game_tree[node]
    
    # Non-terminal node -> Recursively calls min-max
    # call of max
    if maximising_player:
        max_eval=-1000 #or -infinity
        for child in game_tree[node]:
            eval=minmax(child,depth+1,False)
            max_eval=max(max_eval,eval)
        return max_eval
    
    # call of min
    else:
        min_eval=1000 #or infinity
        for child in game_tree[node]:
            eval=minmax(child,depth+1,True)
            min_eval=min(min_eval,eval)
        return min_eval


def alpha_beta(node: str, depth: int, maximizing_player: bool, alpha: int, beta: int) -> int:

    # Terminal node
    if isinstance(game_tree[node], int):
        return game_tree[node]

    # Non-Terminal node
    if maximizing_player:
        max_eval = -1000
        for child in game_tree[node]:
            eval = alpha_beta(child, depth + 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            # Prune the branch if beta <= alpha
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 1000
        for child in game_tree[node]:
            eval = alpha_beta(child, depth + 1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            # Prune the branch if beta <= alpha
            if beta <= alpha:
                break
        return min_eval

# Start the minimax algorithm with alpha-beta pruning from the root of the game tree
optimal_score = alpha_beta('A', 0, True, ALPHA, BETA)
print(optimal_score)



# Start from root - P.S I am sorry for stupid doubt, I was tired
optimal_score=minmax('A',0,True)
# print(optimal_score)

