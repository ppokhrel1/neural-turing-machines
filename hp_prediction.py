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

np.random.seed(1234)

from utils.gen_coords import read_pdb

#read the residue sequence and structure from file
#get residue sequence( 7-9 length ) and the predicted structure
#return it as input_sequence and expected output for the neural network

#split the array into multiple halves
def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

read_sequence = []
gen_dirs = []


#print file.readline()
def read_sentence_n_gen_lists():
    inputs = []
    outputs = []
    file = open("sequence.txt", "r")
    sentence = file.readline()
    while(sentence != ''):
        sequence, directions = sentence.split()[0], sentence.split()[1]
        input = sequence
        #print input
        in_seq = []
        for res in input:
            in_seq.append(float("{0:.2f}".format(float(ord(res)))))
        #print in_seq

        #in_seq = split(in_seq, 9)

        #directions = file.readline()
        output = directions
        out_seq = []
        #create arrays of size 9 each and append them to out_seq

        for chr in output:
            out_seq.append(float("{0:.2f}".format(float(ord(chr)))))
        #print out_seq
        #out_seq = split(in_seq, 9)


        #convert the list to numpy array
        from numpy import array, reshape
        in_seq = array(in_seq, dtype='float32')
        out_seq = array(out_seq, dtype='float32')


        #print len(out_seq)
        in_seq = in_seq.reshape(len(in_seq)/9, 9)
        #in_seq = reshape(in_seq, (len(in_seq)/9, 9))
        out_seq = out_seq.reshape(len(out_seq)/9, 9)
        #out_seq = reshape(out_seq, (len(out_seq)/9, 9))
        inputs.append((in_seq, out_seq))
        #print in_seq
        sentence = file.readline()

    file.close()
    #add padding alter on
    return inputs



def make_train(input_size, output_size, mem_size, mem_width, hidden_sizes=[100]):
    P = Parameters()
    ctrl = controller.build(P, input_size, output_size,
                            mem_size, mem_width, hidden_sizes)
    predict = model.build(P, mem_size, mem_width, hidden_sizes[-1], ctrl)

    input_seq = T.matrix('input_sequence')
    output_seq = T.matrix('output_sequence')
    #input_seq = T.matrix('0101010101')
    #output_seq = T.matrix('0101010101')
    #print input_seq
    seqs = predict(input_seq)
    output_seq_pred = seqs[-1]
    cross_entropy = T.sum(T.nnet.binary_crossentropy(
        5e-6 + (1 - 2 * 5e-6) * output_seq_pred, output_seq), axis=1)
    params = P.values()
    l2 = T.sum(0)
    for p in params:
        l2 = l2 + (p ** 2).sum()
    cost = T.sum(cross_entropy) + 1e-3 * l2
    grads = [T.clip(g, -100, 100) for g in T.grad(cost, wrt=params)]
    #print "ads"
    train = theano.function(
        inputs=[input_seq, output_seq],
        outputs=cost,
        updates=updates.adadelta(params, grads)
    )

    #print str(train)
    return P, train




if __name__ == "__main__":
    model_out = sys.argv[1]

    P, train = make_train(
        input_size=9,
        mem_size=128,
        mem_width=20,
        output_size=27,
        hidden_sizes=[100, 100]
    )
    #print "xxx"
    max_sequences = 1000
    patience = 20000
    patience_increase = 3
    improvement_threshold = 0.995
    best_score = np.inf
    test_score = 0.
    score = None
    alpha = 0.95

        #length = np.random.randint(
        #    int(20 * (min(counter, 50000) / float(50000))**2) + 1) + 1
        #print str(length)
        #i, o = tasks.copy(8, length)
    #inputs = read_sentence_n_gen_lists()
    import os
        #print inputs
    for counter in xrange(max_sequences):
        for file in os.listdir("training"):
            if file.endswith(".pdb"):
                print file
                i,o = read_pdb("training/" + file)
                if score == None:
                    score = train(i, o)
                    print score
                else:
                    score = alpha * score + (1 - alpha) * train(i, o)
                    print "round:", counter, "score:", score
                    if score < best_score:
                        # improve patience if loss improvement is good enough
                        if score < best_score * improvement_threshold:
                            patience = max(patience, counter * patience_increase)
                if patience <= counter:
                    break
        P.save(model_out)
        best_score = score
