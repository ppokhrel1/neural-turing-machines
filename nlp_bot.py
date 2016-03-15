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

import random
from theano.compile.nanguardmode import NanGuardMode
from theano.compile.debugmode import DebugMode

from theano import compile
compile.debugmode.DebugMode()

np.random.seed(1234)

theano.config.exception_verbosity = 'high' # Use 'warn' to activate this feature

#np.seterr( over='ignore' )

#hacky correction for theano bug that converts 0 or 1 to NaN
#does not remove the nan but allows the function to continue
#so that the number get replaced by some other number in next run
#Note to future self: I dont really understand how it manages to do that
#maybe it is realted to checking of values which corrects the weights in the matrices
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and
            np.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            np.clip(output, -10, 10, out=output)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break


# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }




def make_train(input_size, output_size, mem_size, mem_width, hidden_sizes=[100]):
    P = Parameters()
    ctrl = controller.build(P, input_size, output_size,
                            mem_size, mem_width, hidden_sizes)
    predict = model.build(P, mem_size, mem_width, hidden_sizes[-1], ctrl)

    input_seq = T.matrix('input_sequence')
    output_seq = T.matrix('output_sequence')
    seqs = predict(input_seq)
    output_seq_pred = seqs[-1]
    #print seqs
    cross_entropy = T.sum(T.nnet.binary_crossentropy(
        5e-6 + (1 - 2 * 5e-6) * output_seq_pred, output_seq), axis=1)
    cross_entropy = T.maximum(cross_entropy, 0.000001)
    params = P.values()
    l2 = T.sum(0)
    for p in params:
        l2 = l2 + (p ** 2).sum()
    cost = T.sum(cross_entropy) + 1e-4 * l2
    #print "cost", cost
    grads = [T.clip(g, -10, 10) for g in T.grad(cost, wrt=params)]
    #print grads
    #print cost
    mode = theano.compile.MonitorMode(post_func=detect_nan).excluding(
    'local_elemwise_fusion', 'inplace')
    train = theano.function(
        inputs=[input_seq, output_seq],
        outputs=cost,
        #mode=mode,
        mode=theano.compile.MonitorMode(
                        post_func=detect_nan),
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
        updates=updates.rmsprop(params, grads, learning_rate = 1e-4)
    )

    return P, train


def gen_input(length):
    sentence = ''
    out_sentence = ''
    output = []
    output_sentence = []
    for i in xrange(length):
        num = random.randint(36, 100)
        output.append(num)
        output_sentence.append(num + 2)

    output = np.array(output, dtype='float32')
    output_sentence = np.array(output_sentence, dtype='float32')
    return output.reshape(len(output), 1), output_sentence.reshape(len(output_sentence), 1)

if __name__ == "__main__":
    model_out = sys.argv[1]

    P, train = make_train(
        input_size=1,
        mem_size=128,
        mem_width=20,
        output_size=1,
        hidden_sizes=[500, 500]
    )

    max_sequences = 100000
    patience = 20000
    patience_increase = 3
    improvement_threshold = 0.995
    best_score = np.inf
    test_score = 0.
    score = None
    alpha = 0.95
    for counter in xrange(max_sequences):
        '''length = np.random.randint(
           int(20 * (min(counter, 50000) / float(50000))**2) + 1) + 1
        '''
        #print length
        length = np.random.randint(1,21)
        #print str(length)
        #i, o = tasks.copy(8, length)
        i, o = gen_input(length)
        if score == None:
            score = train(i, o)
        else:
            score = alpha * score + (1 - alpha) * train(i, o)
        print "round:", counter, "score:", score
        #print train
        if score < best_score:
            # improve patience if loss improvement is good enough
            if score < best_score * improvement_threshold:
                patience = max(patience, counter * patience_increase)
            P.save(model_out)
            best_score = score

        if patience <= counter:
            break
