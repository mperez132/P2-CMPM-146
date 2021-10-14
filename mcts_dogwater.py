from mcts_node import MCTSNode
from random import choice, random
from math import sqrt, log, inf
# from multiprocessing import Lock, Pool
from threading import Lock, Thread
from time import sleep

# num_nodes = 1000
num_nodes = 1000
explore_faction = 2.
# thread_count = 4
returned_nodes = []
node_count = 0
lock = Lock()

def traverse_nodes(node, board, state, identity):
    # print("o")
    curr_node = node
    return best_ucb(curr_node, True)

# def expand_leaf_multi(node, board, state):
#     returned_nodes = []
#     untrieds = len(node.untried_actions)
#     if untrieds > 4:
#         untrieds = 4
#     with Pool(processes = untrieds) as pool:
#         reses = []
#         for i in range(untrieds):
#             r = pool.apply_async(expand_leaf, (node, board, state))
#             reses.append(r)
#         for res in reses:
#             returned_nodes.append(res.get())
#     # print([n.visits for n in returned_nodes])
#     return returned_nodes 

def expand_leaf_multi(node, board, state):
    # print("elm")
    global returned_nodes
    returned_nodes = []
    untrieds = len(node.untried_actions)
    if untrieds > 4:
        untrieds = 4
    # print(untrieds)
    tlist = []
    for i in range(untrieds):
        # print(i)
        t = Thread(target=expand_leaf, args=(node,board,state,))
        t.start()
        tlist.append(t)
    for thread in tlist:
        thread.join()
    return returned_nodes
    

def expand_leaf(node, board, state):
# def expand_leaf(node, board, state, returned_nodes):
    # choose a random action in our node's list of untried actions
    # do we have any untried actions
    # print("lock")
    # print("Expanding")
    lock.acquire()
    # print("Acquired lock")
    next_action = node.untried_actions[0]
    # remove the node from parent untried
    node.untried_actions.pop(0)
    # print(len(node.untried_actions))
    # sleep(0.5)
    # print(len(node.untried_actions))
    lock.release()
    # get a new state based on that move
    next_state = board.next_state(state, next_action)
    # create a new node using this info
    new_node = MCTSNode(node, next_action, board.legal_actions(next_state))
    # node_count+=1
    # print(node_count, end=', ')
    # add it as a child to our provided node
    node.child_nodes[next_action] = new_node
    # return the node and the new state
    returned_nodes.append(new_node)
    # print("All done with")
    return new_node

def rollout(board, state):
    temp_state = state
    # randomly choose an action from the board
    while not board.is_ended(temp_state):
        rand_action = choice(board.legal_actions(temp_state))
        temp_state = board.next_state(temp_state, rand_action)
    return temp_state

def backpropagate(node, won):
    while node.parent:
        node.wins += won
        node.visits += 1
        node = node.parent
    node.wins += won
    node.visits += 1    

def think(board, state):
    identity_of_bot = board.current_player(state)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(state))
    i_val = 0
    while i_val < num_nodes:
        # Copy the game for sampling a playthrough
        sampled_game = state
        # Start at root
        node = root_node
        global returned_nodes
        
        # MCTS start
        # use traverse_nodes to find the next leaf position as a result of OUR action
        node = traverse_nodes(node, board, sampled_game, identity_of_bot)
        # use expand_leaf to add the node from our action
        sampled_game, board = action_tracker(node, board, sampled_game)
        
        wins = []
        if not node.untried_actions:
            win = board.points_values(sampled_game)[identity_of_bot]
            wins = [(node, win)]
        else:
            lock = Lock()
            # print("z", end='')
            expand_leaf_multi(node, board, sampled_game)
            i_val += len(returned_nodes)
            # print(len(returned_nodes))
            for rn in returned_nodes:
                # print(len(returned_nodes))
                sampled_game = board.next_state(sampled_game, rn.parent_action)
                # use the state provided by rollout to simulate that game
                # sampled_game = rollout(board, sampled_game)
                sampled_game = big_brain_rollout(board, sampled_game, identity_of_bot)
                # use the leaf from expand_leaf AND the result from rollout to set our visits and wins
                wins.append((rn, board.points_values(sampled_game)[identity_of_bot]))
        # print(wins)
        for w in wins:
            backpropagate(w[0], w[1])
    return best_child_action(root_node, identity_of_bot)

def best_child_action(node, identity):
    best_winrate = float('-inf')
    best_action = None
    winrates = []
    for action, child in node.child_nodes.items():
        winrate = child.wins
        winrates.append(winrate)
        if winrate > best_winrate:
            best_action = action
            best_winrate = winrate
    return best_action

# Updating the state with each action in node_actions
def action_tracker(node, board, state):
    node_actions = []
    while node.parent:
        node_actions.append(node.parent_action)
        node = node.parent
    node_actions.reverse()
    for each_action in node_actions:
        state = board.next_state(state, each_action)
    return state, board

def ucb(child, ours):
    winrate = (child.wins/child.visits)
    if not ours:
        winrate = 1 - winrate
    # print(winrate)
    return winrate + explore_faction * sqrt(log(child.parent.visits)/child.visits)

def best_ucb(node, ours):
    while not node.untried_actions and node.child_nodes:
        bestOutcome = float('-inf')
        temp = None
        for action, child in node.child_nodes.items():
            child_ucb = ucb(child, ours)
            if child_ucb > bestOutcome:
                temp = child
                bestOutcome = child_ucb
        node = temp
        ours = not ours
    return node

def big_brain_rollout(board, state, identity):
    rollouts = 1
    max_depth = 2
    
    if board.is_ended(state):
        return state
    
    moves = board.legal_actions(state)

    best_move = moves[0]
    best_expectation = float('-inf')

    all_scores = dict()

    wl_list = [None, None]
    # Define a helper function to calculate the difference between the bot's score and the opponent's.
    def outcome(owned_boxes, game_points):
        if game_points is not None:
            # Try to normalize it up?  Not so sure about this code anyhow.
            red_score = game_points[1]*9
            blue_score = game_points[2]*9
        else:
            red_score = len([v for v in owned_boxes.values() if v == 1])
            blue_score = len([v for v in owned_boxes.values() if v == 2])
        return red_score - blue_score if identity == 1 else blue_score - red_score

    def rollout_single(move, all_scores, wl_list):
        total_score = 0.0

        # Sample a set number of games where the target move is immediately applied.
        for r in range(rollouts):
            rollout_state = board.next_state(state, move)

            while not board.is_ended(rollout_state):
                rollout_move = choice(board.legal_actions(rollout_state))
                rollout_state = board.next_state(rollout_state, rollout_move)

            total_score += outcome(board.owned_boxes(rollout_state), board.points_values(rollout_state))
            
            
            if total_score <= 0:
                wl_list[1] = rollout_state
            else:
                wl_list[0] = rollout_state

        all_scores[move] = total_score
        # print(all_scores)
    tlist = []
    
    for move in moves:
        t = Thread(target=rollout_single, args=(move, all_scores, wl_list))
        t.start()
        tlist.append(t)

    avg_score = 0
    for t in tlist:
        t.join()
        
    for move in all_scores:
        avg_score += all_scores[move]

    avg_score /= len(all_scores)
    
    # if avg_score > 0:
    #     return wl_list[0]
    # return wl_list[1]
    return wl_list[avg_score <= 0]
    
    # return big_brain_rollout(board, board.next_state(state, best_move), identity)
    # return board.next_state(state, best_move)

# def roulette(node):
#     while not node.untried_actions and node.child_nodes:
#         move_selector = random() # random float 0 < x < 1
#         accumulator = 0
#         for child in node.child_nodes.values():
#             winrate = child.wins/child.visits
#             accumulator += winrate
#             if accumulator > move_selector:
#                 node = child        
#     return node

# def roulette(node):
#     Compute the total score from all scores
#     total_score = 0
#     for 
#     for score in scores do
#         scorePercent = the percent score is from total score
#     MoveSelector = Randomly generated number between 0 and 1.0
#     Accumulator = 0
#     for scorePercent in scorePercents do
#         Accumulator += scorePercent
#         if Accumulator ≥MoveSelector then
#             return the move associated with the current scorePercent

# def roulette(node, board, state, identity):
#     wins = False
#     cnodes = []
#     for child in node.child_nodes.values():
#         cnodes.append(child)
#         if child.wins:
#             wins = True
#             break
#     if not wins:
#         return choice(cnodes)
    
#     for child in cnodes:
#         next_state = board.next_state(state, child.parent_action)
#         if board.is_ended(next_state) and board.win_values(next_state)[identity] == 1:
#             return child
#     Check for any one move wins
#     Compute the total score from all scores
#     for score in scores do
#         scorePercent = the percent score is from total score
#     MoveSelector = Randomly generated number between 0 and 1.0
#     Accumulator = 0
#     for scorePercent in scorePercents do
#         Accumulator += scorePercent
#         if Accumulator ≥MoveSelector then
#             return the move associated with the current scorePercent
