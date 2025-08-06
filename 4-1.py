import numpy as np
import tensorcircuit as tc

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

print(c.expectation([tc.gates.z(), [0]], [tc.gates.z(), [1]]))