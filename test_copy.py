import sys
import run_model
import hp_prediction
import numpy as np
import run_model
#p = np.load('ss')
#file = Parameters.load("ss")
#p = Parameters()
p, do_task = run_model.make_model(input_size=8,
    mem_size=128,
    mem_width=20,
    output_size=8,
    hidden_sizes=[100]
    )
with p:
    s = p.load('cpy')
print p.values()
i = np.array([[0, 1, 1, 1, 0, 0, 0, 0]], dtype='float32')
weights, output = do_task(i)
print output
