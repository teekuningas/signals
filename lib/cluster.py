import itertools

import numpy as np


def _distance(component1, component2):
    """
    """
    return 1.0


def _objective(components):
    """
    """
    
    length = len(components[0])
    combinations = itertools.combinations(range(len(components)), 2)
    
    sum_ = 0
    for combination in combinations:
        for i in range(length):
            sum_ += _distance(components[combination[0]][i],
                              components[combination[1]][i])
    return sum_


def _order_similarly(model, unordered):
    """
    """
    permutations = [list(perm) for perm in 
                    itertools.permutations(range(model.shape[0]))]

    min_objective = None
    min_permutation = None
    for permutation in permutations:
        objective = _objective([model, unordered[permutation]])
        if not min_objective or objective < min_objective:
            min_objective = objective
            min_permutation = permutation
    
    return min_permutation


def _get_initial_state(components):
    """
    """
    components = np.array(components)

    # create an index array
    solution = np.zeros((components.shape[0], components.shape[1]))
    solution = solution.astype(np.int8)

    # take first as it is
    solution[0, :] = np.arange(components.shape[1])

    for i in range(solution.shape[0] - 1):
        solution[i+1, :] = _order_similarly(
            components[i, :][solution[i, :]], 
            components[i+1, :]
        )

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
    solution = initial_state

    for i in range(len(components)):
        components[i] = list(np_components[i, :][solution[i]])

    return components
