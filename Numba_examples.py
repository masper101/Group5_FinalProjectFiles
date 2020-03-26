from numba import jit
import numpy as np
import time

"""
Example 1: Simple performance test

Obtained directly from https://numba.pydata.org/numba-examples/

"""

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a): # Function is compiled and runs in machine code
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

"""
Example 2: Corona Virus Simulation 

This example models the time it takes for the corona virus to affect 100% of a population of 
1 Million people based on various behavioral rules. 
Documentation: see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3372334/

Techniques include:
1) Control, no changes in behaviors and habits of a population
2) Moderate, social distancing rules enacted
3) Significant, "shelter in place" policies passed

"""

pop = 1e6

@jit
def infect_time(connections):
    # this function calculates the time it takes to infect a population of ppl with
    # no behavioral control measures in place

    infected = {'child': 0, 'adult': 1, 'senior': 0}  # an adult is most likely to be infected first
    total = [np.sum([num for num in infected.values()])]

    #loop everyday until 100% infected
    day = 0
    while total[-1] <= pop:
        infected_1 = infected.copy()
        for key in infected_1:
            new_infected = np.multiply(infected[key], connections[key])
            infected['child'] += new_infected[0]
            infected['adult'] += new_infected[1]
            infected['senior'] += new_infected[2]
        total.append(np.sum([num for num in infected.values()]))
        day += 1
    return day, total

# Connections that a person will have each day [child, adult, senior]
control = {'child': [36, 9, 2], 'adult': [4, 16, 2], 'senior': [3, 7, 9]}
moderate = {'child': [8, 9, 2], 'adult': [4, 10, 2], 'senior': [3, 7, 5]}
significant = {'child': [0, 2, 0], 'adult': [1, 1, 0], 'senior': [0, 0, 1]}

# Compilation time
start1 = time.time()
[control_days, control_totals] = infect_time(control)
[moderate_days, moderate_totals] = infect_time(moderate)
[significant_days, significant_totals] = infect_time(significant)
end1 = time.time()
print('It takes %d days with NO regulations to infect 1 million people.' % control_days)
print('It takes %d days with moderate regulations to infect 1 million people.' % moderate_days)
print('It takes %d days with significant regulations to infect 1 million people.' % significant_days)
print("Elapsed (with compilation) = %s" % (end1 - start1))

# Executing from cache
start2 = time.time()
[control_days, control_totals] = infect_time(control)
[moderate_days, moderate_totals] = infect_time(moderate)
[significant_days, significant_totals] = infect_time(significant)
end2 = time.time()
print('It takes %d days with NO regulations to infect 1 million people.' % control_days)
print('It takes %d days with moderate regulations to infect 1 million people.' % moderate_days)
print('It takes %d days with significant regulations to infect 1 million people.' % significant_days)
print("Elapsed (after compilation) = %s" % (end2 - start2))
