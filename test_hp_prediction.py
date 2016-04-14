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



p, do_task = run_model.make_model(input_size= 9 * 5,
    mem_size=128,
    mem_width=20,
    output_size=3,
    hidden_sizes=[512, 512]
    )
with p:
    s = p.load('ss')
#print p.values()
from utils.gen_coords import *
i= []
o=[]
predicted = []
for input ,output in gen_training_data("training/" + "2mwy.pdb"):
    i.extend(np.array(input).tolist())
    o.extend(np.array(output).tolist())

    weights, outputs = do_task(input)
    predicted.extend(np.array(outputs).tolist())
    #i = np.array(i, dtype='float32')

i = np.array(i, dtype='float32')
o = np.array(o, dtype='float32')
predicted = np.array(predicted, dtype='float32')

plt.figure(1, figsize=(20, 7))
plt.subplot(311)
plt.imshow(i.T, interpolation='nearest')
plt.subplot(312)
plt.imshow(o.T, interpolation='nearest')
plt.subplot(313)
plt.imshow(predicted.T, interpolation='nearest')
plt.show()

print "inputs:", i
print "actual outputs:", o
print "predicted: ", predicted
#print len(outputs.tolist()[1])
#.make_train.predict(i)
# = s.values()
#print p.values()

plt.figure(1, figsize=(20, 20))
plt.imshow(weights.T[0:123], interpolation='nearest')# cmap=cm.gray)
plt.show()
