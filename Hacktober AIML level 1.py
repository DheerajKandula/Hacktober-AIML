# TASK 1
import math
# 1. ACTIVATION FUNCTIONS

# Sigmoid function

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Relu

def relu(x):
    return  max(0, x)

# Tanh

def tanh(x):
    return math.tanh(x)

# 2. DOT PRODUCT AND COSINE SIMILARITY

# Dot product

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Incorrect input: Vectors must have the same length !")
    return sum(a*b for a,b in zip(v1, v2))

# Cosine Similarity

def vector_magnitude(x):
    return math.sqrt(sum(n**2 for n in x))

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Incorrect input: Vectors must have the same length")
    num = dot_product(v1, v2)
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 == 0 or mag2  == 0:
        raise ValueError("Error!: vector magnitude cannot be zero")
    return num / (mag1 * mag2)
# 3. NORMALIZATION FUNCTION

# L1

def l1_normalize(vector):
    l1_norm = sum(abs(x) for x in vector)
    if l1_norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / l1_norm for x in vector]

# L2 

def l2_normalize(vector):
    l2_norm = math.sqrt(sum(x**2 for x in vector))
    if l2_norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / l2_norm for x in vector]

# Min - Max

def min_max_normalize(vector, new_min=0, new_max=1):
    old_min = min(vector)
    old_max = max(vector)
    
    if old_min == old_max:
        raise ValueError("Cannot normalize constant vector")
    
    range_old = old_max - old_min
    range_new = new_max - new_min
    
    return [(x - old_min) / range_old * range_new + new_min for x in vector]
# 4. Heuristic function (greedy move for tic-tac-toe)

def greedy_heuristic(board, player):

    opponent = 'O' if player == 'X' else 'X'

    win_move = find_winning_move(board, player)
    if win_move:
        return win_move
    
    block_move = find_winning_move(board, opponent)
    if block_move:
        return block_move
    
    if board[1][1] == ' ':
        return (1, 1)
    
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for corner in corners:
        if board[corner[0]][corner[1]] == ' ':
            return corner
    
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for edge in edges:
        if board[edge[0]][edge[1]] == ' ':
            return edge
    
    return None

def find_winning_move(board, player):
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player
                if check_winner(board, player):
                    board[i][j] = ' '
                    return (i, j)
                board[i][j] = ' '
    return None

def check_winner(board, player):
    for i in range(3):
        if all(board[i][j] == player for j in range(3)):
            return True
        if all(board[j][i] == player for j in range(3)):
            return True
    
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2-i] == player for i in range(3)):
        return True
    
    return False