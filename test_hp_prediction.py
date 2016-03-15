import theano
import theano.tensor as T
import numpy as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U
from theano_toolkit import hinton
import controller
import model
import tasks
import sys
import run_model
import hp_prediction

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import run_model
#p = np.load('ss')
#file = Parameters.load("ss")
#p = Parameters()



p, do_task = run_model.make_model(input_size=9,
    mem_size=128,
    mem_width=20,
    output_size=27,
    hidden_sizes=[500, 500]
    )
with p:
    s = p.load('hp_predict')
#print p.values()
from utils.gen_coords import read_pdb

i,o = read_pdb("training/" + "2mwy.pdb")

weights, outputs = do_task(i)
plt.figure(1, figsize=(20, 7))
plt.subplot(311)
plt.imshow(i.T, interpolation='nearest')
plt.subplot(312)
plt.imshow(o.T, interpolation='nearest')
plt.subplot(313)
plt.imshow(outputs.T, interpolation='nearest')
plt.show()
#print len(outputs.tolist()[1])
#.make_train.predict(i)
# = s.values()
#print p.values()

plt.figure(1, figsize=(20, 20))
plt.imshow(weights.T[100:123], interpolation='nearest', cmap=cm.gray)
plt.show()
