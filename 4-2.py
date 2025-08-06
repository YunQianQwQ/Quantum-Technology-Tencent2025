import numpy as np
import tensorcircuit as tc

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

count = {}
for _ in range(10000):
    vec, _ = c.measure(0,1)
    vec = tuple(vec)
    if vec in count:
        count[vec] += 1
    else:
        count[vec] = 1

print(count)