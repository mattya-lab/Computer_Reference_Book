# -*- coding: utf-8 -*-

import math
from simpleai.search import SearchProblem, astar

MAP = """
##############################
#         #              #   #
# ####    ########       #   #
#  o #    #              #   #
#    ###     #####  ######   #
#      #   ###   #           #
#      #     #   #  #  #   ###
#     #####    #    #  #x    #
#              #       #     #
##############################
"""

print(MAP)
#MAP's type change <class 'str' > type into list type #<shape = ( 10, 30 )>
MAP = [list(x) for x in MAP.split("\n") if x]

cost_regular = 1.0
cost_diagonal = 1.7
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
    "up left":cost_diagonal,
    "up right":cost_diagonal,
    "down left":cost_diagonal,
    "down right":cost_diagonal,
}

class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)
        
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)
                
        super(MazeSolver, self).__init__(initial_state=self.initial)
    
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)
            
        return actions
    
    def result(self, state, action):
        x, y = state
        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1
        
        now_state = (x, y)
        return now_state
    
    def is_goal(self, state):
        return state == self.goal
    
    def cost(self, state, action, state2):
        return COSTS[action]
    
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        
        return math.sqrt((x - gx)**2 + (y - gy)**2)

probrem = MazeSolver(MAP)
result = astar(probrem, graph_search=True)
#Get Route to start path and goal path
path = [x[1] for x in result.path()]

print()
for y in range(len(MAP)):
    for x in range(len(MAP[y])):
        if (x, y) == probrem.initial:
            print('o', end='')
        elif (x, y) == probrem.goal:
            print('x', end='')
        elif (x, y) in path:
            print(".", end='')
        else:
            print(MAP[y][x], end='')
    print()