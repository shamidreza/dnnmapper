"""
The initial version of the code is downloaded from https://raw.githubusercontent.com/mdenil/dropout/master/mlp.py on 2015-02-14.
The code is further modified to match dnnmapper software needs.
You have to follow the LICENSE provided on deeplearning.net website (also included below), in addition to 
the LICENSE provided as part of the dnnmapper software.
"""
"""
    This file is part of dnnmapper.

    dnnmapper is free software: you can redistribute it and/or modify
    it under the tersms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    dnnmapper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with dnnmapper.  If not, see <http://www.gnu.org/licenses/>.
"""
"""
https://github.com/mdenil/dropout/blob/master/LICENSE:
Copyright (C) 2012 Misha Denil

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression


##################################
## Various activation functions ##
##################################
from experiment import CUR_ACIVATION_FUNCTION as af
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def mse(self, y):
        return T.mean((self.output-y)**2)
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)
        
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.
    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True,
            pretrained=None):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input        
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            if pretrained:
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                        input=next_dropout_layer_input,
                        activation=activations[layer_counter],
                        n_in=n_in, n_out=n_out, use_bias=use_bias,
                        dropout_rate=dropout_rates[layer_counter + 1],
                        W=theano.shared(pretrained.layers[layer_counter].W.eval()),
                        b=theano.shared(pretrained.layers[layer_counter].b.eval()))
            else:
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                        input=next_dropout_layer_input,
                        activation=activations[layer_counter],
                        n_in=n_in, n_out=n_out, use_bias=use_bias,
                        dropout_rate=dropout_rates[layer_counter + 1]
                        )
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W, ##$* (1 - dropout_rates[layer_counter]), ##$
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = DropoutHiddenLayer(
                rng, next_dropout_layer_input,
                n_in, n_out, None, 0.0, use_bias)##$make sure if we need dropout here
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = HiddenLayer(
            rng,
            next_layer_input, n_in, n_out, None,##$
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W, ##$ * (1 - dropout_rates[-1]),##$
            b=dropout_output_layer.b
            )
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        
        

        # Grab all the parameters together.
        if False:##$
            self.params = [ param for layer in self.dropout_layers for param in layer.params ]
        else:
            self.params = [ param for layer in self.layers for param in layer.params ]

    def dropout_mse(self, y):
        return self.dropout_layers[-1].mse(y)
    def dropout_errors(self, y):
        return self.dropout_layers[-1].mse(y)
    def mse(self, y):
        return self.layers[-1].mse(y)
    def errors(self, y):
        return self.layers[-1].mse(y)
    
    #self.dropout_negative_log_likelihood = self.dropout_layers[-1].mse
    #self.dropout_errors = self.dropout_layers[-1].mse

    #self.negative_log_likelihood = self.layers[-1].mse
    #self.errors = self.layers[-1].mse

def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        dataset,
        use_bias,
        random_seed=1234):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    from utils import load_vc
    #datasets = load_mnist(dataset)
    print '... loading the data'
    dataset = 'c2s.npy'
    datasets, x_mean, y_mean, x_std, y_std = load_vc(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(random_seed)

    # construct the MLP class
        
    if 1: # load
        f = open('c2s_pre.npy.dnn.pkl','r')
        pretrained = cPickle.load(f)
        f.close()
    else:
        pretrained=None
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias,
                     pretrained=pretrained)

    # Build the expresson for the cost function.
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    test_fprop = theano.function(inputs=[],
            outputs=classifier.layers[-1].output,
            givens={
                x: test_set_x
                })
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # Misha Denil's original version
        #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        # Misha Denil's original version
        #stepped_param = param - learning_rate * updates[gparam_mom]
        
        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            #updates[param] = stepped_param * scale
            
            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param


    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')
    X2=test_set_y.eval()
    X2 *= y_std
    X2 += y_mean
    X1=test_set_x.eval()
    X1 *= x_std
    X1 += x_mean
    last_reg = 10000.0
    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)

        # Compute loss on validation set
        validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.mean(validation_losses)

        # Report and save progress.
        print "epoch {}, test error {}, learning_rate={}{}".format(
                epoch_counter, this_validation_errors,
                learning_rate.get_value(borrow=True),
                " **" if this_validation_errors < best_validation_errors else "")

        best_validation_errors = min(best_validation_errors,
                this_validation_errors)
        results_file.write("{0}\n".format(this_validation_errors))
        results_file.flush()

        new_learning_rate = decay_learning_rate()
        YH=test_fprop()
        YH *= y_std
        YH += y_mean
        print 'Regression ', np.mean(np.mean((YH-X2)**2,1))
        print 'Baseline! ', np.mean(np.mean((X1-X2)**2,1))
        if np.mean(np.mean((YH-X2)**2,1)) < last_reg:
            print 'This is better. Saving the model to ' + dataset+'.dnn.pkl'
            f = open(dataset+'.dnn.pkl','w+')
            cPickle.dump(classifier, f)
            f.flush()
            f.close()
            last_reg = np.mean(np.mean((YH-X2)**2,1))
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    import sys
    
    # set the random seed to enable reproduciable results
    # It is used for initializing the weight matrices
    # and generating the dropout masks for each mini-batch
    random_seed = 1234

    initial_learning_rate = 0.020
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 3000
    batch_size = 10
    layer_sizes = [ 1500, 1000, 1000, 1500 ]
    
    # dropout rate for each layer
    dropout_rates = [ 0.2, 0.5, 0.5 ]
    # activation functions for each layer
    # For this demo, we don't need to set the activation functions for the 
    # on top layer, since it is always 10-way Softmax
    activations = [ T.tanh, T.tanh ]
    
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                  
    dataset = 'data/mnist_batches.npz'
    #dataset = 'data/mnist.pkl.gz'

    #if len(sys.argv) < 2:
        #print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        #exit(1)
    
    #elif sys.argv[1] == "dropout":
        #dropout = True
        #results_file_name = "results_dropout.txt"

    #elif sys.argv[1] == "backprop":
        #dropout = False
    results_file_name = "results_backprop.txt"

    #else:
        #print "I don't know how to '{0}'".format(sys.argv[1])
        #exit(1)
    dropout=True
    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             squared_filter_length_limit=squared_filter_length_limit,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             mom_params=mom_params,
             activations=activations,
             dropout=dropout,
             dropout_rates=dropout_rates,
             dataset=dataset,
             results_file_name=results_file_name,
             use_bias=False,
             random_seed=random_seed)

