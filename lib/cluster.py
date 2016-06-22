import itertools
import multiprocessing

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


def _order_similarly(data):
    """
    """
    model, unordered = data

    permutations = [list(perm) for perm in 
                    itertools.permutations(range(model.shape[0]))]

    min_objective = None
    min_permutation = None
    for i, permutation in enumerate(permutations):
        if i%1000 == 0:
            print (str(i) + "'s permutation of thread " + 
                   str(multiprocessing.current_process()))
        objective = _objective([model, unordered[permutation]])
        if not min_objective or objective < min_objective:
            min_objective = objective
            min_permutation = permutation
    
    return min_permutation


def _get_initial_state(data):
    """
    """
    data = np.array(data)

    # create an index array
    solution = np.zeros((data.shape[0], data.shape[1]))
    solution = solution.astype(np.int8)

    # take first as it is
    solution[0, :] = np.arange(solution.shape[1])

    # order rest with respect to first one
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    solution_count = solution.shape[0]

    idx = 0
    while True:
        print (str(idx+1) + " out of " + str(solution_count) + 
               " initial solutions found.")

        parallel_count = (cpu_count if (solution_count-1) - idx >= cpu_count 
                          else (solution_count-1) - idx)

        components = [(data[0, :][solution[0, :]], data[i+1, :]) 
                      for i in range(idx, idx + parallel_count)]

        intmed = pool.map(
            _order_similarly,
            components
        )

        for i in range(parallel_count):
            solution[idx+i+1, :] = intmed[i]

        idx += parallel_count
        if idx >= solution_count - 1:
            break

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
