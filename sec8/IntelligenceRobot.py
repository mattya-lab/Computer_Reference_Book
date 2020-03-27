# -*- coding: utf-8 -*-
import numpy as np
import random
from RobotController import RobotController
from deap import algorithms, base, creator, tools, gp

random.seed(7)

max_moves = 750
robot = RobotController(max_moves)

with open('target_map.txt', 'r') as f:
    robot.traverse_map(f)        
        
def eval_func(individual):
    routine = gp.compile(individual, pset)
    robot.run(routine)
    return (robot.consumed,)

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(robot.if_target_ahead, 2)
pset.addPrimitive(robot.prog2, 2)
pset.addPrimitive(robot.prog3, 3)
pset.addTerminal(robot.move_forward)
pset.addTerminal(robot.turn_left)
pset.addTerminal(robot.turn_right)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

population = toolbox.population(n=400)
hall_of_fame = tools.HallOfFame(1)

stats = tools.Statistics(lambda x: x.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

probab_crossover = 0.4
probab_mutate = 0.3
num_generations = 50

population, log = algorithms.eaSimple(population, toolbox, probab_crossover, 
                                      probab_mutate, num_generations, stats, 
                                      halloffame=hall_of_fame)
#print(hall_of_fame[0])
#---Out---
#prog2(prog2(move_forward, if_target_ahead(prog2(move_forward, 
#if_target_ahead(move_forward, turn_left)), prog3(prog3(turn_left, 
#move_forward, move_forward), move_forward, move_forward))), 
#if_target_ahead(move_forward, turn_right))


#This means that
 
#move_forward, 
#if target_ahead():
#    move_forward, 
#    if_target_ahead():
#        move_forward()
#    else :
#        turn_left()
#else :
#    turn_left, 
#    move_forward(), 
#    move_forward(), 
#    move_forward(), 
#    move_forward(), 
#if_target_ahead():
#    move_forward()
#else : 
#    turn_right)
