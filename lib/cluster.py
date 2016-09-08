import itertools
import math

from random import random
from random import randint

import numpy as np


def _distance(component1, component2):
    """
    """

    image1 = component1.sensor_topo
    image2 = component2.sensor_topo

    # distance of images
    distance =  math.pow(np.linalg.norm(np.abs(image1 - image2)), 2)

    return distance


def _objective(collections):
    """
    """
    
    length = len(collections[0])
    combinations = itertools.combinations(range(len(collections)), 2)
    
    sum_ = 0
    for combination in combinations:
        for i in range(length):
            sum_ += _distance(collections[combination[0]][i],
                              collections[combination[1]][i])
    return sum_


def _nonesorter(value):
    if not value[0]:
        return float("inf")
    return value[0]


def _order_similarly(model, unordered, amount):
    """ Returns `amount` best solutions
    """ 

    permutations = [list(perm) for perm in 
                    itertools.permutations(range(model.shape[0]))]

    # replace last one and sort

    min_permutations = [(None, None)]*amount

    for i, permutation in enumerate(permutations):
        objective = _objective([model, unordered[permutation]])

        last_obj, last_perm = min_permutations[amount-1]

        if not last_obj or objective < last_obj:
            min_permutations[amount-1] = (objective, permutation)
         
            # sort if new entry
            min_permutations = sorted(min_permutations, key=_nonesorter)

    return [perm for obj, perm in min_permutations]


def _get_initial_state(data):
    """
    """

    # take `amount` best solutions
    amount = 20

    # create an index array
    solution = np.zeros((data.shape[0], amount, data.shape[1]))
    solution = solution.astype(np.int8)

    # take first as it is
    solution[0, :, :] = np.array([range(data.shape[1]) for i in range(amount)])

    solution_count = solution.shape[0]

    for i in range(data.shape[0] - 1):

        model = data[0, :][solution[0, 0, :]]
        unordered = data[i+1, :]

        sol = _order_similarly(model, unordered, amount)

        solution[i+1, :] = np.array(sol)

        print (str(i+2) + " out of " + str(solution_count) + 
               " initial solutions found.")

    return solution


def _cost(components, solution):
    ordered = [component[solution[i, 0]] 
               for i, component in enumerate(components)]
    return _objective(ordered)


def _neighbor(solution):

    new_solution = solution.copy()

    # select three trials to swap
    trial_idxs = [randint(0, solution.shape[0] - 1) for i in range(3)]
    for trial_idx in trial_idxs:

        # swap random solution with first one
        solution_idx = randint(0, solution.shape[1] - 1)
        new_solution[trial_idx, 0] = solution[trial_idx, solution_idx]
        new_solution[trial_idx, solution_idx] = solution[trial_idx, 0]

    return new_solution


def _acceptance_probability(old_cost, new_cost, T):

    if old_cost > new_cost:
        return 1.0

    return pow(math.e, float(old_cost-new_cost)/T)


def _anneal(components, solution):
    """ depicted from http://katrinaeg.com/simulated-annealing.html
    """
    old_cost = _cost(components, solution)
    T = 1.0
    T_min = 0.00001
    alpha = 0.95
    while T > T_min:
        print str(T)
        idx = 1
        while idx <= 100:
            new_solution = _neighbor(solution.copy())
            new_cost = _cost(components, new_solution)
            ap = _acceptance_probability(old_cost, new_cost, T)
            if ap > random():
                if old_cost > new_cost: 
                    print "Improves!"
                if old_cost < new_cost:
                    print "Gets worse!"
                if not old_cost == new_cost:
                    print "old_cost: ", old_cost, ", new_cost: ", new_cost

                solution = new_solution
                old_cost = new_cost
            idx += 1
        T = T*alpha
    return solution


def cluster_components(components):
    """
    sort elements of nxm matrix so that components so that elements 
    on same row are similar to each other. use simulated annealing
    """
    print "Clustering.."

    np_components = np.array(components)

    print "Get initial state"
    initial_state = _get_initial_state(np_components)

    print "Do simulated annealing.."
    solution = _anneal(np_components, initial_state)

    for i in range(len(components)):
        components[i] = list(np_components[i, :][solution[i, 0]])

    return components
