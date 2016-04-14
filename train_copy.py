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
            #theano.printing.debugprint(node)
            #np.clip(output, -10, 10, out=output)
            print([T.sgn(input[0]) for input in fn.inputs])

            #print('Inputs : %s' % [input[0] for input in fn.inputs])
            #print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

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
    #cross_entropy = T.maximum(cross_entropy, 0.000001)
    params = P.values()
    l2 = T.sum(0)
    for p in params:
        l2 = l2 + (p ** 2).sum()
    cost = T.sum(cross_entropy) + 1e-4 * l2
    #print "cost", cost
    grads = [T.clip(g, -100, 100) for g in T.grad(cost, wrt=params)] #if 0 is in the grads, we set it to 1e-10 to avoid exploding gradients
    #print grads
    #grads = T.clip(grads, -10, 10)
    #mode = theano.compile.MonitorMode(post_func=detect_nan).excluding(
    #'local_elemwise_fusion', 'inplace')
    train = theano.function(
        inputs=[input_seq, output_seq],
        outputs=cost,
        #mode=mode,
        mode=theano.compile.MonitorMode(
                        post_func=detect_nan),
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
        updates=updates.rmsprop(params, grads, learning_rate = 1e-8)
    )

    return P, train

if __name__ == "__main__":
    model_out = sys.argv[1]

    P, train = make_train(
        input_size=8,
        mem_size=128,
        mem_width=20,
        output_size=8,
        hidden_sizes=[100]
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

        #print length
        length = np.random.randint(1,21)
        #print str(length)
        i, o = tasks.copy(8, length)

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
