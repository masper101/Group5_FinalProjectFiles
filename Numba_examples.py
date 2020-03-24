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

Techniques include:
1) Control, no changes in behaviors and habits of a population
2) Moderate, large social gatherings are prohibited 
3) Limited, social distancing rules enacted
4) Significant, "shelter in place" policies passed

"""
