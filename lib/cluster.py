import itertools
from random import random

import numpy as np


def _distance(component1, component2):
    """
    """

    image1 = component1.sensor_topo
    image2 = component2.sensor_topo

    # distance of images
    distance =  np.linalg.norm(np.abs(image1 - image2))

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
    amount = 5

    data = np.array(data)

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


def _cost(solution):
    return 1.0


def _neighbor(solution):
    return solution


def _acceptance_probability(old_cost, new_cost, T):
    return 1.0


def _anneal(solution):
    """ depicted from http://katrinaeg.com/simulated-annealing.html
    """
    old_cost = _cost(solution)
    T = 1.0
    T_min = 0.00001
    alpha = 0.9
    while T > T_min:
        idx = 1
        while idx <= 100:
            new_solution = _neighbor(solution)
            new_cost = _cost(new_solution)
            ap = _acceptance_probability(old_cost, new_cost, T)
            if ap > random():
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
    solution = _anneal(initial_state)

    for i in range(len(components)):
        components[i] = list(np_components[i, :][solution[i, 0]])

    return components
