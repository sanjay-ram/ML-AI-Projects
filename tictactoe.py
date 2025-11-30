import numpy as np
import random


Q = {}

def initialize_broad():
    return np.zeros((3,3), dtype=int)

def check_win(board, player):
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

def check_draw(board):
    return not np.any(board == 0)

def get_available_actions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

def choose_action(state, board, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_available_actions(board))
    else:
        if state in Q and Q[state]:
            return max(Q[state], key=Q[state].get)
        else:
            return random.choice(get_available_actions(board))
        
def update_q_value(state, action, reward, next_state, alpha, gamma):
    max_future_q = max(Q.get(next_state, {}).values(), default=0)
    current_q = Q.get(state, {}).get(action, 0)
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    if state not in Q:
        Q[state] = {}
    Q[state][action] = new_q

def board_to_tuple(board):
    return tuple(map(tuple, board))

def train(episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    win_history = []
    for episode in range(episodes):
        board = initialize_broad()
        state = board_to_tuple(board)
        done = False
        result = None  
        while not done:
            action = choose_action(state, board, epsilon)
            board[action[0], action[1]] = 1
            next_state = board_to_tuple(board)
            if check_win(board, 1):
                update_q_value(state, action, 1, next_state, alpha, gamma)
                result = 1  
                done = True
            elif check_draw(board):
                update_q_value(state, action, 0.5, next_state, alpha, gamma)
                result = 0  
                done = True
            else:
                opponent_action = random.choice(get_available_actions(board))
                board[opponent_action[0], opponent_action[1]] = -1
                next_state = board_to_tuple(board)
                if check_win(board, -1):
                    update_q_value(state, action, -1, next_state, alpha, gamma)
                    result = -1  
                    done = True
                elif check_draw(board):
                    update_q_value(state, action, 0.5, next_state, alpha, gamma)
                    result = 0  
                    done = True
                else:
                    update_q_value(state, action, 0, next_state, alpha, gamma)
            state = next_state
        if result == 1:
            win_history.append(1)
        else:
            win_history.append(0)
    return win_history

win_history = train(100000)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window_size = 100

win_rate = moving_average(win_history, window_size)
episodes = np.arange(window_size, len(win_history) + 1)


def print_board(board):
    symbols = {0: " ", 1: "X", -1: "O"}
    print("\n   0   1   2")
    for i in range(3):
        row = " | ".join(symbols[board[i, j]] for j in range(3))
        print(f"{i}  {row}")
        if i < 2:
            print("  ---+---+---")
    print()

def play_against_bot(epsilon=0.0):
    board = initialize_broad()
    state = board_to_tuple(board)
    
    print("Du spielst O (-1), der Bot spielt X (1).")
    print("Koordinaten eingeben als:  Zeile Spalte  (z.B. 0 2)\n")
    
    while True:
        print_board(board)

        # --- Mensch spielt (-1) ---
        while True:
            try:
                r, c = map(int, input("Dein Zug (z,b): ").split())
                if board[r, c] == 0:
                    break
                else:
                    print("Feld ist schon belegt.")
            except:
                print("Ungültige Eingabe, bitte z.b. '1 2' eingeben.")

        board[r, c] = -1

        if check_win(board, -1):
            print_board(board)
            print("DU hast gewonnen!")
            return
        if check_draw(board):
            print_board(board)
            print("Unentschieden!")
            return

        state = board_to_tuple(board)
        action = choose_action(state, board, epsilon=epsilon)  
        board[action[0], action[1]] = 1
        print(f"\nBot spielt: {action}")

        if check_win(board, 1):
            print_board(board)
            print(" hat gewonnen!")
            return
        if check_draw(board):
            print_board(board)
            print("Unentschieden!")
            
if "__main__" == __name__:
    play_against_bot(epsilon=0.0)