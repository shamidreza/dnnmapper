"""
The initial version of the code is downloaded from http://deeplearning.net/tutorial/code/dA.py on 2015-02-14.
The code is further modified to match dnnmapper software needs.
You have to follow the LICENSE provided on deeplearning.net website (also included below), in addition to 
the LICENSE provided as part of the dnnmapper software.
"""
"""
    This file is part of dnnmapper.

    dnnmapper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
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
http://deeplearning.net/tutorial/LICENSE.html:
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

#from logistic_sgd import load_data
from utils import tile_raster_images, load_vc

try:
    import PIL.Image as Image
except ImportError:
    import Image

try:
    from matplotlib import pyplot as pp
except ImportError:
    print 'matplotlib is could not be imported'

from experiment import CUR_ACIVATION_FUNCTION as af


# start-snippet-1
class dA_joint(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input1=None,
        input2=None,
        cor_reg=None,
        n_visible1=784/2,
        n_visible2=784/2,
        n_hidden=500,
        W1=None,
        bhid1=None,
        bvis1=None,
        W2=None,
        bhid2=None,
        bvis2=None
    ):
        self.n_visible1 = n_visible1
        self.n_visible2 = n_visible2

        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W1:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible1)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible1)),
                    size=(n_visible1, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W1 = theano.shared(value=initial_W1, name='W1', borrow=True)
        if not W2:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible2)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible2)),
                    size=(n_visible2, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W2 = theano.shared(value=initial_W2, name='W2', borrow=True)

        if not bvis1:
            bvis1 = theano.shared(
                value=numpy.zeros(
                    n_visible1,
                    dtype=theano.config.floatX
                ),
                name='b1p',
                borrow=True
            )
        if not bvis2:
            bvis2 = theano.shared(
                value=numpy.zeros(
                    n_visible2,
                    dtype=theano.config.floatX
                ),
                name='b2p',
                borrow=True
            )

        if not bhid1:
            bhid1 = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b1',
                borrow=True
            )
        if not bhid2:
            bhid2 = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b2',
                borrow=True
            )

        self.W1 = W1
        self.W2 = W2

        # b corresponds to the bias of the hidden
        self.b1 = bhid1
        self.b2 = bhid2

        # b_prime corresponds to the bias of the visible
        self.b1_prime = bvis1
        self.b2_prime = bvis2

        # tied weights, therefore W_prime is W transpose
        self.W1_prime = self.W1.T
        self.W2_prime = self.W2.T

        self.theano_rng = theano_rng
        self.L1 = (
            abs(self.W1).sum()+abs(self.W2).sum()#+abs(self.b1).sum()+abs(self.b2).sum()+abs(self.b1_prime).sum()+abs(self.b2_prime).sum()
        )
    
        self.L2_sqr = (
            (self.W1**2).sum()#+(self.W2**2).sum()#+abs(self.b1**2).sum()+abs(self.b2**2).sum()+abs(self.b1_prime**2).sum()+abs(self.b2_prime**2).sum()

        )
        # if no input is given, generate a variable representing the input
        if input1 is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x1 = T.dmatrix(name='input1')
            self.x2 = T.dmatrix(name='input2')

        else:
            self.x1 = input1
            self.x2 = input2


        self.params = [self.W1, self.b1, self.b1_prime,
                       self.W2, self.b2, self.b2_prime
        ]
        # end-snippet-1
        self.output1 = af(T.dot(self.x1, self.W1) + self.b1)
        self.output2 = af(T.dot(self.x2, self.W2) + self.b2)
        self.rec1 = (T.dot(self.output1, self.W1_prime) + self.b1_prime)
        self.rec2 = (T.dot(self.output2, self.W2_prime) + self.b2_prime)
        self.reg = (T.dot(self.output1, self.W2_prime) + self.b2_prime)
        self.cor_reg = theano.shared(numpy.float32(1.0),name='reg')
    def get_corrupted_input(self, input1, input2, corruption_level):
        a=self.theano_rng.binomial(size=input1.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input1
        b=self.theano_rng.binomial(size=input2.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input2
        return a,b

    def get_hidden_values(self, input1, input2):
        """ Computes the values of the hidden layer """
        return af(T.dot(input1, self.W1) + self.b1), af(T.dot(input2, self.W2) + self.b2)

    def get_reconstructed_input(self, hidden1, hidden2):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        #a = af(T.dot(hidden1, self.W1_prime) + self.b1_prime)
        #b = af(T.dot(hidden2, self.W2_prime) + self.b2_prime)
        a = (T.dot(hidden1, self.W1_prime) + self.b1_prime)
        b = (T.dot(hidden2, self.W2_prime) + self.b2_prime)
        return a, b

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x1, tilde_x2 = self.get_corrupted_input(self.x1, self.x2, corruption_level)
        y1, y2 = self.get_hidden_values(tilde_x1, tilde_x2)

        z1, z2 = self.get_reconstructed_input(y1, y2)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L_x1 = - T.sum(self.x1 * T.log(z1) + (1 - self.x1) * T.log(1 - z1), axis=1)
        L_x2 = - T.sum(self.x2 * T.log(z2) + (1 - self.x2) * T.log(1 - z2), axis=1)
        L_X1_x2 = - T.sum(y1 * T.log(y2) + (1 - y1) * T.log(1 - y2), axis=1)
        L_X2_x1 = - T.sum(y2 * T.log(y1) + (1 - y2) * T.log(1 - y1), axis=1)
        #L_X1_x2 = T.mean(T.mean((y1-y2)**2,1))
        L_x1 = ((z1-self.x1)**2) #+ (1 - self.x1) * T.log(1 - z1), axis=1)
        L_x2 = ((z2-self.x2)**2)
        L_X1_x2 = ((y1-y2)**2)
        ##cost = T.mean(L_x1) + T.mean(L_x2) + self.cor_reg*T.mean(L_X1_x2)+0.001*self.L1+001*self.L2_sqr# + 0.2*T.mean(L_X2_x1)
        cost = T.mean(L_x1) + T.mean(L_x2) + 1.0*T.mean(L_X1_x2) #+ .001*self.L2_sqr# + 0.2*T.mean(L_X2_x1)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    
def test_dA_joint(learning_rate=0.01, training_epochs=15000,
            dataset='mnist.pkl.gz',
            batch_size=5, output_folder='dA_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    ##datasets = load_data(dataset)
    #from SdA_mapping import load_data_half
    #datasets = load_data_half(dataset)
    print 'loading data'
    datasets, x_mean, y_mean, x_std, y_std = load_vc()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]  
    test_set_x, test_set_y = datasets[2]
    print 'loaded data'

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x1 = T.matrix('x1')  # the data is presented as rasterized images
    x2 = T.matrix('x2')  # the data is presented as rasterized images
    cor_reg = T.scalar('cor_reg')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #da = dA_joint(
        #numpy_rng=rng,
        #theano_rng=theano_rng,
        #input1=x1,
        #input2=x2,

        #n_visible1=28 * 28/2,
        #n_visible2=28 * 28/2,

        #n_hidden=500
    #)
    print 'initialize functions'

    da = dA_joint(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input1=x1,
        input2=x2,
        cor_reg=cor_reg,

        #n_visible1=28 * 28/2,
        #n_visible2=28 * 28/2,
        n_visible1=24,
        n_visible2=24,
        n_hidden=50
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )
    cor_reg_val = numpy.float32(5.0)
    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x1: train_set_x[index * batch_size: (index + 1) * batch_size],
            x2: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    fprop_x1 = theano.function(
               [],
               outputs=da.output1,
               givens={
                   x1: test_set_x
               },
               name='fprop_x1'
    )
    fprop_x2 = theano.function(
               [],
               outputs=da.output2,
               givens={
                   x2: test_set_y
               },
               name='fprop_x2'
    )
    fprop_x1t = theano.function(
               [],
               outputs=da.output1,
               givens={
                   x1: train_set_x
               },
               name='fprop_x1'
    )
    fprop_x2t = theano.function(
               [],
               outputs=da.output2,
               givens={
                   x2: train_set_y
               },
               name='fprop_x2'
    )
    rec_x1 = theano.function(
               [],
               outputs=da.rec1,
               givens={
                   x1: test_set_x
               },
               name='rec_x1'
    )
    rec_x2 = theano.function(
               [],
               outputs=da.rec2,
               givens={
                   x2: test_set_y
               },
               name='rec_x2'
    )
    fprop_x1_to_x2 = theano.function(
               [],
               outputs=da.reg,
               givens={
                   x1: test_set_x
               },
               name='fprop_x12x2'
    )
    updates_reg = [
            (da.cor_reg, da.cor_reg+theano.shared(numpy.float32(0.1)))
    ]
    update_reg = theano.function(
        [],
        updates=updates_reg
    )
    print 'initialize functions ended'

    
    start_time = time.clock()

    ############
    # TRAINING #
    ############
    print 'training started'
    X1=test_set_x.eval()
    X1 *= x_std
    X1 += x_mean
    X2=test_set_y.eval()
    X2 *= y_std
    X2 += y_mean
    from dcca_numpy import cor_cost
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        
        #cor_reg_val += 1
        #da.cor_reg = theano.shared(cor_reg_val) 
        update_reg()
        
        X1H=rec_x1()
        X2H=rec_x2()
        X1H *= x_std
        X1H += x_mean
        X2H *= y_std
        X2H += y_mean
        H1=fprop_x1()
        H2=fprop_x2()
        print 'Training epoch'
        print 'Reconstruction ', numpy.mean(numpy.mean((X1H-X1)**2,1)),\
              numpy.mean(numpy.mean((X2H-X2)**2,1))
        
        if epoch%5 == 2 : # pretrain middle layer
            print '... pre-training MIDDLE layer'
            H1t=fprop_x1t()
            H2t=fprop_x2t()
            h1 = T.matrix('x')  # the data is presented as rasterized images
            h2 = T.matrix('y')  # the labels are presented as 1D vector of
            from mlp import HiddenLayer
            numpy_rng = numpy.random.RandomState(89677)
            log_reg = HiddenLayer(numpy_rng, h1, 50, 50, activation=T.tanh)

            if 1: # for middle layer
                learning_rate = 0.1
            
                #H1=theano.shared(H1)
                #H2=theano.shared(H2)
                # compute the gradients with respect to the model parameters
                logreg_cost = log_reg.mse(h2)
    
                gparams = T.grad(logreg_cost, log_reg.params)
        
                # compute list of fine-tuning updates
                updates = [
                    (param, param - gparam * learning_rate)
                    for param, gparam in zip(log_reg.params, gparams)
                ]
    
                train_fn_middle = theano.function(
                    inputs=[],
                    outputs=logreg_cost,
                    updates=updates,
                    givens={
                        h1: theano.shared(H1t),
                        h2: theano.shared(H2t)
                    },
                    name='train_middle'
                )
            epoch = 0
            while epoch < 100:
                print epoch, train_fn_middle()
                epoch += 1
            
            ##X2H=fprop_x1_to_x2()
            X2H=numpy.tanh(H1.dot(log_reg.W.eval())+log_reg.b.eval())
            X2H=numpy.tanh(X2H.dot(da.W2_prime.eval())+da.b2_prime.eval())

            X2H *= y_std
            X2H += y_mean
            print 'Regression ', numpy.mean(numpy.mean((X2H-X2)**2,1))
        
        print 'Correlation ', cor_cost(H1, H2)
    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W1.get_value(borrow=True).T,
                           img_shape=(28, 14), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')
    
    from matplotlib import pyplot as pp
    pp.plot(H1[:10,:2],'b');pp.plot(H2[:10,:2],'r');pp.show()
    
    print cor
        
  


if __name__ == '__main__':
    test_dA_joint()
