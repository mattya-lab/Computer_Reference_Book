# -*- coding: utf-8 -*-

import copy
from functools import partial

class RobotController(object):
    def __init__(self, max_moves):
        self.max_moves = max_moves
        self.direction = ["north", "east", "south", "west"]
        self.direction_row = [1, 0, -1, 0]
        self.direction_col = [0, 1, 0, -1]
    
    def _reset(self):
        self.row = self.row_start 
        self.col = self.col_start 
        self.direction = 1
        self.moves = 0  
        self.consumed = 0
        self.matrix_exc = copy.deepcopy(self.matrix)
        self.routine = None


    def turn_left(self): 
        if self.moves < self.max_moves:
            self.moves += 1
            self.direction = (self.direction - 1) % 4

    def turn_right(self):
        if self.moves < self.max_moves:
            self.moves += 1    
            self.direction = (self.direction + 1) % 4
        
    def move_forward(self):
        if self.moves < self.max_moves:
            self.moves += 1
            self.row = (self.row + self.direction_row[self.direction]) % self.matrix_row
            self.col = (self.col + self.direction_col[self.direction]) % self.matrix_col

            if self.matrix_exc[self.row][self.col] == "target":
                self.consumed += 1

            self.matrix_exc[self.row][self.col] = "passed"

    def _conditional(self, condition, out1, out2):
        if condition():
            out1()
        else:
            out2()            

    def sense_target(self):
        ahead_row = (self.row + self.direction_row[self.direction]) % self.matrix_row
        ahead_col = (self.col + self.direction_col[self.direction]) % self.matrix_col        
        return self.matrix_exc[ahead_row][ahead_col] == "target"
   
    def if_target_ahead(self, out1, out2):
        return partial(self._conditional, self.sense_target, out1, out2)

    def _progn(self, *args):
        for arg in args:
            arg()

    def prog2(self, out1, out2): 
        return partial(self._progn, out1, out2)

    def prog3(self, out1, out2, out3):     
        return partial(self._progn, out1, out2, out3)
           
    def run(self,routine):
        self._reset()
        while self.moves < self.max_moves:
            routine()
    
    def traverse_map(self, matrix):
        self.matrix = list()
        for i, line in enumerate(matrix):
            self.matrix.append(list())

            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("target")

                elif col == ".":
                    self.matrix[-1].append("empty")

                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.direction = 1

        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])