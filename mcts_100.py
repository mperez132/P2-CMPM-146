from mcts_node import MCTSNode
from random import choice
from math import sqrt, log, inf

num_nodes = 100
explore_faction = 2.

def traverse_nodes(node, board, state, identity):
    curr_node = node
    return best_ucb(curr_node, True)

def expand_leaf(node, board, state):
    # choose a random action in our node's list of untried actions
    # lock
    # do we have any untried actions
    next_action = choice(node.untried_actions)
    # remove the node from parent untried
    node.untried_actions.remove(next_action)
    # release lock
    # get a new state based on that move
    next_state = board.next_state(state, next_action)
    # create a new node using this info
    new_node = MCTSNode(node, next_action, board.legal_actions(next_state))
    # add it as a child to our provided node
    node.child_nodes[next_action] = new_node
    # return the node and the new state
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

    for step in range(num_nodes):
        # Copy the game for sampling a playthrough
        sampled_game = state
        # Start at root
        node = root_node
        
        # MCTS start
        # use traverse_nodes to find the next leaf position as a result of OUR action
        node = traverse_nodes(node, board, sampled_game, identity_of_bot)
        # use expand_leaf to add the node from our action
        sampled_game, board = action_tracker(node, board, sampled_game)
        if not node.untried_actions:
            win = board.points_values(sampled_game)[identity_of_bot]
        else:
            node = expand_leaf(node, board, sampled_game)
            sampled_game = board.next_state(sampled_game, node.parent_action)
            # use the state provided by rollout to simulate that game
            sampled_game = rollout(board, sampled_game)
            # use the leaf from expand_leaf AND the result from rollout to set our visits and wins
            win = board.points_values(sampled_game)[identity_of_bot]
            
        backpropagate(node, win)
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
