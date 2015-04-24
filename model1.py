"""
This file is part of deepcca.

deepcca is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

deepcca is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with deepcca.  If not, see <http://www.gnu.org/licenses/>.
"""
"""


"""

import os
import sys
import time
import gzip
import cPickle

import numpy

import theano
import theano.tensor as T

from utils import load_vc_siamese, melCD, load_vc_all_speakers_24
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        self.input = input
        self.activation = activation
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.params = [self.W, self.b]
        
    def mse(self, y):
        return T.mean((self.output-y)**2)
    def crossentropy(self, y):
        L = - T.sum(self.output * T.log(y) + (1 - self.output) * T.log(1 - y), axis=1)
        return T.mean(L)

class siames(object):
    def __init__(self, rng, x, y, spk_same, phon_same, layers_size=[24, 200, 200, 200, 24], CS_size=100):
        self.x = x
        self.y = y
        self.spk_same = spk_same
        self.spk_same_not = T.matrix('sn')
        self.phon_same = phon_same
        self.phon_same_not = T.matrix('pn')

        if 1: # initialize W1 and b1
            n_in = layers_size[0]
            n_out = layers_size[1]
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W1 = theano.shared(value=W_values, name='W1', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b1 = theano.shared(value=b_values, name='b1', borrow=True)
        if 1: # initialize W2 and b2
            n_in = layers_size[1]
            n_out = layers_size[2]-CS_size
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W2 = theano.shared(value=W_values, name='W2', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b2 = theano.shared(value=b_values, name='b2', borrow=True)

        if 1: # initialize W2_CS and b2_CS
            n_in = layers_size[1]
            n_out = CS_size
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W2_CS = theano.shared(value=W_values, name='W2_CS', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b2_CS = theano.shared(value=b_values, name='b2_CS', borrow=True)

        
        if 1: # initialize W3_CS and b3_CS
            n_in = CS_size
            n_out = layers_size[3]
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W3_CS = theano.shared(value=W_values, name='W3_CS', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b3_CS = theano.shared(value=b_values, name='b3_CS', borrow=True)

        
        
        if 1: # initialize W3 and b3
            n_in = layers_size[2]-CS_size
            n_out = layers_size[3]
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W3 = theano.shared(value=W_values, name='W3', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b3 = theano.shared(value=b_values, name='b3', borrow=True)

        
        
        
        if 1: # initialize W4 and b4
            n_in = layers_size[3]
            n_out = layers_size[4]
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4
            self.W4 = theano.shared(value=W_values, name='W4', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b4 = theano.shared(value=b_values, name='b4', borrow=True)

        
        self.activation = T.nnet.sigmoid
        self.output1x = self.activation( T.dot(self.x, self.W1) + self.b1 )
        self.output1y = self.activation( T.dot(self.y, self.W1) + self.b1 )
        self.output2x_CS = self.activation( T.dot(self.output1x, self.W2_CS) + self.b2_CS )
        self.output2y_CS = self.activation( T.dot(self.output1y, self.W2_CS) + self.b2_CS )
        self.output2x = self.activation( T.dot(self.output1x, self.W2) + self.b2 )
        self.output2y = self.activation( T.dot(self.output1y, self.W2) + self.b2 )
        self.output3x = self.activation( T.dot(self.output2x, self.W3) + self.b3 + \
                                         T.dot(self.output2x_CS, self.W3_CS) + self.b3_CS )
        self.output3y = self.activation( T.dot(self.output2y, self.W3) + self.b3 + \
                                         T.dot(self.output2y_CS, self.W3_CS) + self.b3_CS )
        self.output4x = T.dot(self.output3x, self.W4) + self.b4
        self.output4y = T.dot(self.output3y, self.W4) + self.b4

        self.params = [self.W1, self.b1,
                       self.W2, self.b2,
                       self.W2_CS, self.b2_CS,
                       self.W3, self.b3,
                       self.W3_CS, self.b3_CS,
                       self.W4, self.b4]
        
        D = T.mean((self.output2x-self.output2y)**2)
        D_CS = T.mean((self.output2x_CS-self.output2y_CS)**2)

        self.cost = \
            T.mean((self.output4x-self.x)**2) + \
            T.mean((self.output4y-self.y)**2) + \
            T.mean(self.spk_same*D_CS) #+ \
            #T.mean(-(1.0-self.spk_same)*D_CS) #+ \
            #T.mean(self.phon_same*D) + \
            #T.mean(-(1.0-self.phon_same)*D)

            #-(1.0-self.spk_same)*T.exp(-T.sum((self.output2x_CS-self.output2y_CS)**2)) + \
            #self.phon_same*T.mean((self.output2x-self.output2y)**2) + \
            #-(1.0-self.phon_same)*T.exp(-T.sum((self.output2x-self.output2y)**2))
            
            #T.switch(self.spk_same,
                     #T.mean((self.output2x_CS-self.output2y_CS)**2),
                     ##T.exp(-T.sum((self.output2x_CS-self.output2y_CS)**2))) + \
                     #T.mean((self.output2x_CS-self.output2y_CS)**2)) + \
            #T.switch(self.phon_same,
                     #T.mean((self.output2x-self.output2y)**2),
                     ##T.exp(-T.sum((self.output2x-self.output2y)**2)))
                     #T.mean((self.output2x-self.output2y)**2))
        self.rec = \
            T.mean((self.output4x-self.x)**2) + \
            T.mean((self.output4y-self.y)**2)
        
def ae_all(out_file, hidden_layers_sizes=None,
           corruption_levels=None,
           pretrain_lr=None,
           batch_size=None,
           training_epochs=None): # ae all on TIMIT
    print '... loading the data'
    new_data=load_vc_all_speakers_24()   
   
    numpy_rng = numpy.random.RandomState(89677)
    
    import theano
    n_train_batches = int(0.9*new_data.shape[0])
    n_train_batches /= batch_size
    
    train_set = theano.shared(new_data[:int(0.9*new_data.shape[0]), :])
    test_set = theano.shared(new_data[int(0.9*new_data.shape[0]):, :])
    test_set_numpy = new_data[int(0.9*new_data.shape[0]):, :]

    print '... building the model'
    from ae_stacked import SdA
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=new_data.shape[1],
        hidden_layers_sizes=hidden_layers_sizes,#[1000, 1000],
    )
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set,
                                                batch_size=batch_size)
    
    print '...training the model'
    import time
    import pickle
    start_time = time.clock()
    reconstruct = theano.function(
                inputs=[                   
                ],
                outputs=sda.dA_layers[0].xrec,
                givens={
                    sda.dA_layers[0].x: test_set
                }
            )
    #corruption_levels = [0.2, 0.3]
    lr = pretrain_lr
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(training_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            if i==0:
                XH = reconstruct()
                #XH = unnormalize_data(XH, mins, ranges)[:,24*7:24*7+24]
                print 'melCD', melCD(XH, test_set_numpy)
            
            lr *= 0.99
            if lr < 0.01:
                lr = 0.01
            import pickle
            f=open(out_file,'w+')
            pickle.dump(sda, f)
            #pickle.dump(mins, f)
            #pickle.dump(ranges, f)
            f.flush()
            f.close()
        
def test_siames(dataset='final_corpus.npy', 
                learning_rate=0.02,
                n_epochs=1000,
                batch_size=1):
  
    datasets = load_vc_siamese(dataset)

    train_set_x, train_set_y, train_set_spk, train_set_phon = datasets[0]
    valid_set_x, valid_set_y, valid_set_spk, valid_set_phon = datasets[1]
    test_set_x, test_set_y, test_set_spk, test_set_phon = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the x1
    y = T.matrix('y')  # the x2
    s = T.matrix('s') # the same-speaker label (0/1)
    p = T.matrix('p') # the same-phoneme label (0/1)
    rng = numpy.random.RandomState(1234)
    if 0:
        # construct the siamese class
        regressor = siames(
            rng=rng,
            x=x,
            y=y,
            spk_same=s,
            phon_same=p,
            layers_size=[24, 200, 200, 200, 24],
            CS_size=50,
        )
    else:
        import pickle
        f=open('siamese4.pkl', 'r')
        regressor = pickle.load(f)
        f.close()

    cost = regressor.cost

    test_model = theano.function(
        inputs=[],
        outputs=regressor.rec,
        givens={
            regressor.x: test_set_x,
            regressor.y: test_set_y
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=regressor.rec,
        givens={
            regressor.x: test_set_x[index * batch_size: (index + 1) * batch_size],
            regressor.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    fprop_model = theano.function(
        inputs=[],
        outputs=[regressor.output4x, regressor.output4y,
                 regressor.output2x,regressor.output2x_CS,
                 regressor.output2y,regressor.output2y_CS],
        givens={
            regressor.x: test_set_x,
            regressor.y: test_set_y
        }
    )
    
    tx_np = test_set_x.eval()
    ty_np = test_set_y.eval()
    gparams = [T.grad(cost, param) for param in regressor.params]
  
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(regressor.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            regressor.x: train_set_x[index * batch_size: (index + 1) * batch_size],
            regressor.y: train_set_y[index * batch_size: (index + 1) * batch_size],
            regressor.spk_same: train_set_spk[index * batch_size: (index + 1) * batch_size]#,
            #regressor.phon_same: train_set_phon[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    if 1:
        import pickle
        f=open('wavs_timit.pkl','r')
        wav_slt=pickle.load(f)
        wav_clb=pickle.load(f)
        wav_rms=pickle.load(f)
        wav_mdl=pickle.load(f)
        f.close()
        fprop2_model = theano.function(
            inputs=[],
            outputs=[regressor.output4x, regressor.output4y,
                     regressor.output2x,regressor.output2x_CS,
                     regressor.output2y,regressor.output2y_CS],
            givens={
                regressor.x: theano.shared(wav_clb.astype(numpy.float32)),
                regressor.y: theano.shared(wav_slt.astype(numpy.float32))
            }
        )
        fprop3_model = theano.function(
            inputs=[],
            outputs=[regressor.output4x, regressor.output4y,
                     regressor.output2x,regressor.output2x_CS,
                     regressor.output2y,regressor.output2y_CS],
            givens={
                regressor.x: theano.shared(wav_rms.astype(numpy.float32)),
                regressor.y: theano.shared(wav_mdl.astype(numpy.float32))
            }
        )
        
        out2 = fprop2_model()
        out3 = fprop3_model()
        import tsne
        vis1=tsne.bh_sne(numpy.r_[out2[3],out2[5],out3[3], out3[5]].astype(numpy.float64))
        vis2=tsne.bh_sne(numpy.r_[out2[2],out2[4],out3[2], out3[4]].astype(numpy.float64))
        vis0=tsne.bh_sne(numpy.r_[wav_clb,wav_slt,wav_rms, wav_mdl].astype(numpy.float64))

        from matplotlib import pyplot as pp
        if 1: # vis 0
            st = 0
            pp.plot(vis0[st:st+wav_clb.shape[0],0], vis0[st:st+wav_clb.shape[0],1],'b*')
            st+=wav_clb.shape[0]
            pp.plot(vis0[st:st+wav_slt.shape[0],0], vis0[st:st+wav_slt.shape[0],1],'r+')
            st+=wav_slt.shape[0]
            pp.plot(vis0[st:st+wav_rms.shape[0],0], vis0[st:st+wav_rms.shape[0],1],'go')
            st+=wav_rms.shape[0]
            pp.plot(vis0[st:st+wav_mdl.shape[0],0], vis0[st:st+wav_mdl.shape[0],1],'ko')
            pp.legend(['clb', 'slt','rms','mdl'])

            pp.show()
        if 1: # vis 1
            st = 0
            pp.plot(vis1[st:st+wav_clb.shape[0],0], vis1[st:st+wav_clb.shape[0],1],'b*')
            st+=wav_clb.shape[0]
            pp.plot(vis1[st:st+wav_slt.shape[0],0], vis1[st:st+wav_slt.shape[0],1],'r+')
            st+=wav_slt.shape[0]
            pp.plot(vis1[st:st+wav_rms.shape[0],0], vis1[st:st+wav_rms.shape[0],1],'go')
            st+=wav_rms.shape[0]
            pp.plot(vis1[st:st+wav_mdl.shape[0],0], vis1[st:st+wav_mdl.shape[0],1],'ko')
            pp.legend(['clb', 'slt','rms','mdl'])

            pp.show()
            
        if 1: # vis 2
            st = 0
            pp.plot(vis2[st:st+wav_clb.shape[0],0], vis2[st:st+wav_clb.shape[0],1],'b*')
            st+=wav_clb.shape[0]
            pp.plot(vis2[st:st+wav_slt.shape[0],0], vis2[st:st+wav_slt.shape[0],1],'r+')
            st+=wav_slt.shape[0]
            pp.plot(vis2[st:st+wav_rms.shape[0],0], vis2[st:st+wav_rms.shape[0],1],'go')
            st+=wav_rms.shape[0]
            pp.plot(vis2[st:st+wav_mdl.shape[0],0], vis2[st:st+wav_mdl.shape[0],1],'ko')
            pp.legend(['clb', 'slt','rms','mdl'])
            pp.show()
        a=0
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 100000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            #print validate_model(minibatch_index)
            #print regressor.b4.eval().mean()
            #print regressor.b4.eval().std()
            #print regressor.b4.eval().max()
            #print regressor.b4.eval().min()
            #print '******'
            #print regressor.W4.eval().mean()
            #print regressor.W4.eval().std()
            #print regressor.W4.eval().max()
            #print regressor.W4.eval().min()
            #print '------'
            
            if minibatch_index % 100000 == 0:
                if numpy.isnan(validate_model(0)):
                    print 'here'
                out = fprop_model()
                print melCD(out[0],tx_np), melCD(out[1],ty_np)
                
            #print test_model()
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                this_validation_loss = test_model()
                print(
                    'epoch %i, minibatch %i/%i, validation error %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss
                    )
                )
    
                   
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    if 1: # ae all
        ae_name = 'ae_1000_500_linear_male.pkl'
        ae_all(ae_name, hidden_layers_sizes=[1000,200],               
                   corruption_levels=[0.1,0.1],
                   pretrain_lr=0.05,
                   batch_size=20,
                   training_epochs=20)
    if 0: # train siamese
        test_siames()
