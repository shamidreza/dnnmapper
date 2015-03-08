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
from utils import *

CUR_ACIVATION_FUNCTION = Tanh

def ae_all(out_file, hidden_layers_sizes=None,
           corruption_levels=None,
           pretrain_lr=None,
           batch_size=None,
           training_epochs=None): # ae all on TIMIT
    print '... loading the data'
    data=load_vc_all_speakers()
    print '... loaded data with dimensions', str(data.shape[0]),'x', str(data.shape[1])
    print '... normalizing the data'
    mins, ranges = compute_normalization_factors(data)
    import pickle
    f=open('norm.pkl','w+')
    pickle.dump(mins, f)#[24*7:24*7+24]##$
    pickle.dump(ranges, f)#[24*7:24*7+24]##$
    f.flush()
    f.close()
    new_data = normalize_data(data, mins, ranges)
    numpy_rng = np.random.RandomState(89677)
    
    import theano
    n_train_batches = int(0.9*new_data.shape[0])
    n_train_batches /= batch_size
    #new_data = new_data.astype(np.float32)[:,24*7:24*7+24]
    #mins = mins[24*7:24*7+24]
    #ranges = ranges[24*7:24*7+24]
    train_set = theano.shared(new_data[:int(0.9*new_data.shape[0]), :])
    test_set = theano.shared(new_data[int(0.9*new_data.shape[0]):, :])
    test_set_unnormalized = unnormalize_data(new_data[int(0.9*new_data.shape[0]):, :], mins, ranges)[:,24*7:24*7+24]
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
                XH = unnormalize_data(XH, mins, ranges)[:,24*7:24*7+24]
                print 'melCD', melCD(XH, test_set_unnormalized)
            lr *= 0.99
            if lr < 0.01:
                lr = 0.01
    import pickle
    f=open(out_file,'w+')
    pickle.dump(sda, f)
    pickle.dump(mins, f)
    pickle.dump(ranges, f)
    f.flush()
    f.close()
    print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
    print np.mean(c)
            

    end_time = time.clock()

    print 'The pretraining code for file ' +\
                          os.path.split(__file__)[1] +\
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)
    
    
def model2_pre(x, y, xv, yv, xt, yt, mins, ranges,
               hidden_layers_sizes=None,
               corruption_levels=None,
               lr=None,
               batch_size=None,
               training_epochs=None): # 1.(A, Bb_backprop, C_backprop),MCEP15
    initial_learning_rate = lr
    learning_rate_decay = 0.998
    dropout=False
    squared_filter_length_limit = 15.0
    mom_start = 0.5
    mom_end = 0.99   
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
    results_file_name = "results_backprop.txt"
    dataset = 'test_xy'
    train_set_x = x
    train_set_y = y
    valid_set_x = xv
    valid_set_y = xv
    test_set_x = xt
    test_set_y = yt
    
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
    random_seed = 1234
    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    from dnn_dropout import MLP

    if 0: # no pretraining
        pretrained=None
    elif 1: # ae pretraining + middle layer
        f=open('ae_100.pkl','r')
        sda=pickle.load(f)
        mins=pickle.load(f)
        ranges=pickle.load(f)
        f.close()
        pretrained = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=corruption_levels,
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True
                     )
        for i in range(len(sda.dA_layers)):
            pretrained.layers[i].W = sda.dA_layers[i].W
            pretrained.layers[i].b = sda.dA_layers[i].b
        for i in range(len(sda.dA_layers)):
            pretrained.layers[len(sda.dA_layers)+i].W = sda.dA_layers[len(sda.dA_layers)-i-1].W_prime
            pretrained.layers[len(sda.dA_layers)+i].b = sda.dA_layers[len(sda.dA_layers)-i-1].b_prime
    elif 0: # mlp pretraining
        pass
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=corruption_levels,
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True,
                     pretrained=pretrained)
    #from mlp import MLP
    #classifier = MLP(
        #rng=rng,
        #input=x,
        #n_in=24,
        #n_hidden=75,
        #n_out=24
    #)

    # Build the expresson for the cost function.
    cost = classifier.mse(y)
    dropout_cost = classifier.dropout_mse(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
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
    from theano.ifelse import ifelse
    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)
    from collections import OrderedDict
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
    import time
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')
    X2=test_set_y.eval()
    X2 = unnormalize_data(X2, mins, ranges)
    X1=test_set_x.eval()
    X1 = unnormalize_data(X1, mins, ranges)
    last_reg = 10000.0
    while epoch_counter < training_epochs:
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
        YH = unnormalize_data(YH, mins, ranges)

        print 'Regression ', melCD(X2[:,24*7:24*7+24],YH[:,24*7:24*7+24])#np.mean(np.mean((YH-X2)**2,1))
        print 'Baseline! ', melCD(X1[:,24*7:24*7+24],X2[:,24*7:24*7+24])#np.mean(np.mean((X1-X2)**2,1))
        #print 'Regression ', melCD(X2,YH)#np.mean(np.mean((YH-X2)**2,1))
        #print 'Baseline! ', melCD(X1,X2)#np.mean(np.mean((X1-X2)**2,1))
        if np.mean(np.mean((YH-X2)**2,1)) < last_reg:
            print 'This is better. Saving the model to ' + dataset+'.dnn.pkl'
            f = open(dataset+'.dnn.pkl','w+')
            pickle.dump(classifier, f)
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

   
def model0_pre(x, y, xv, yv, xt, yt, mins, ranges,
               hidden_layers_sizes=None,
               corruption_levels=None,
               lr=None,
               batch_size=None,
               training_epochs=None): # GMM,MCEP
    import sys
    sys.path.append('../gitlab/pysig/src/python/')
    from pysig.learning.mapping import JDGMM
    model=JDGMM(Q=32,regularizer=0.0001)
    x=x.eval().astype(np.float64)
    y=y.eval().astype(np.float64)
    xt=xt.eval().astype(np.float64)
    yt=yt.eval().astype(np.float64)
    model.train(x, y)
    YH=model.map(xt)
    YH = unnormalize_data(YH, mins, ranges)
    X2 = unnormalize_data(yt, mins, ranges)
    X1 = unnormalize_data(xt, mins, ranges)
    print 'Regression ', melCD(X2,YH)#np.mean(np.mean((YH-X2)**2,1))
    print 'Baseline! ', melCD(X1,X2)#np.mean(np.mean((X1-X2)**2,1))
    

def model1_pre(x, y, xv, yv, xt, yt, mins, ranges,
               hidden_layers_sizes=None,
               corruption_levels=None,
               lr=None,
               batch_size=None,
               training_epochs=None): # 4,(C_backprop),MCEP.
    initial_learning_rate = lr
    learning_rate_decay = 0.998
    dropout=False
    squared_filter_length_limit = 15.0
    mom_start = 0.5
    mom_end = 0.99   
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
    results_file_name = "results_backprop.txt"
    dataset = 'test_xy'
    train_set_x = x
    train_set_y = y
    valid_set_x = xv
    valid_set_y = xv
    test_set_x = xt
    test_set_y = yt
    
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
    random_seed = 1234
    rng = np.random.RandomState(random_seed)

    # construct the MLP class
        
    if 0: # no pretraining
        pretrained=None
    elif 0: # ae pretraining + middle layer
        f = open('test.dnn.pkl','r')
        pretrained = pickle.load(f)
        f.close()
    elif 0: # mlp pretraining
        pass
    from dnn_dropout import MLP
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=corruption_levels,
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True,
                     pretrained=pretrained)
    #from mlp import MLP
    #classifier = MLP(
        #rng=rng,
        #input=x,
        #n_in=24,
        #n_hidden=75,
        #n_out=24
    #)

    # Build the expresson for the cost function.
    cost = classifier.mse(y)
    dropout_cost = classifier.dropout_mse(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
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
    from theano.ifelse import ifelse
    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)
    from collections import OrderedDict
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
    import time
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')
    X2=test_set_y.eval()
    X2 = unnormalize_data(X2, mins, ranges)
    X1=test_set_x.eval()
    X1 = unnormalize_data(X1, mins, ranges)
    last_reg = 10000.0
    while epoch_counter < training_epochs:
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
        YH = unnormalize_data(YH, mins, ranges)

        #print 'Regression ', melCD(X2[:,24*7:24*7+24],YH[:,24*7:24*7+24])#np.mean(np.mean((YH-X2)**2,1))
        #print 'Baseline! ', melCD(X1[:,24*7:24*7+24],X2[:,24*7:24*7+24])#np.mean(np.mean((X1-X2)**2,1))
        print 'Regression ', melCD(X2,YH)#np.mean(np.mean((YH-X2)**2,1))
        print 'Baseline! ', melCD(X1,X2)#np.mean(np.mean((X1-X2)**2,1))
        if np.mean(np.mean((YH-X2)**2,1)) < last_reg:
            print 'This is better. Saving the model to ' + dataset+'.dnn.pkl'
            f = open(dataset+'.dnn.pkl','w+')
            pickle.dump(classifier, f)
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

def model2_pre_from_siae(inp_file):
    f=open(inp_file,'r')
    sda=pickle.load(f)
    f.close()
    hidden_layers_sizes = [sda.dA_layers[0].n_visible]
    for i in range(len(sda.dA_layers)):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    for i in range(len(sda.dA_layers)-1, -1, -1):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    hidden_layers_sizes.append(sda.dA_layers[0].n_visible)
    random_seed = 1234
    rng = np.random.RandomState(random_seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    from dnn_dropout import MLP

    pretrained = MLP(rng=rng, input=x,
                 layer_sizes=hidden_layers_sizes,
                 dropout_rates=[0.0]*len(hidden_layers_sizes),
                 activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                 use_bias=True
                 )
    for i in range(len(sda.dA_layers)):
        pretrained.layers[i].W = sda.dA_layers[i].W
        pretrained.layers[i].b = sda.dA_layers[i].b
    for i in range(len(sda.dA_layers)):
        pretrained.layers[len(sda.dA_layers)+i].W = sda.dA_layers[len(sda.dA_layers)-i-1].W_prime
        pretrained.layers[len(sda.dA_layers)+i].b = sda.dA_layers[len(sda.dA_layers)-i-1].b_prime
    return pretrained

def model2_pre_from_speaker20(inp_file):
    f=open(inp_file,'r')
    pretrained=pickle.load(f)
    return pretrained

def dnn_train(pretrain_func, inp_file, out_file, x, y, xv, yv, xt, yt, mins, ranges,
               hidden_layers_sizes=None,
               middle_layers_sizes=None,
               corruption_levels=None,
               lr=None,
               batch_size=None,
               training_epochs=None): # 1.(A, Bb_backprop, C_backprop),MCEP15
    initial_learning_rate = lr
    learning_rate_decay = 0.998
    dropout=False
    squared_filter_length_limit = 15.0
    mom_start = 0.5
    mom_end = 0.99   
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
    results_file_name = "results_backprop.txt"
    dataset = 'test_xy'
    train_set_x = x
    train_set_y = y
    valid_set_x = xv
    valid_set_y = xv
    test_set_x = xt
    test_set_y = yt
    
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
    random_seed = 1234
    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    from dnn_dropout import MLP

    pretrained = pretrain_func(inp_file, middle_layers_sizes, train_set_x, train_set_y)
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=corruption_levels,
                     activations=[CUR_ACIVATION_FUNCTION]*(len(hidden_layers_sizes)+2),
                     use_bias=True,
                     pretrained=pretrained)
    #from mlp import MLP
    #classifier = MLP(
        #rng=rng,
        #input=x,
        #n_in=24,
        #n_hidden=75,
        #n_out=24
    #)

    # Build the expresson for the cost function.
    cost = classifier.mse(y)
    dropout_cost = classifier.dropout_mse(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compile theano function for validation.
    validate_model = theano.function(inputs=[index],
            outputs=classifier.mse(y),
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
    from theano.ifelse import ifelse
    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
            mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
            mom_end)
    from collections import OrderedDict
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
    import time
    start_time = time.clock()

    results_file = open(results_file_name, 'wb')
    X2=test_set_y.eval()
    X2 = unnormalize_data(X2, mins, ranges)
    X1=test_set_x.eval()
    X1 = unnormalize_data(X1, mins, ranges)
    last_reg = 10000.0
    while epoch_counter < training_epochs:
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
        YH = unnormalize_data(YH, mins, ranges)

        print 'Regression ', melCD(X2[:,24*7:24*7+24],YH[:,24*7:24*7+24])#np.mean(np.mean((YH-X2)**2,1))
        print 'Baseline! ', melCD(X1[:,24*7:24*7+24],X2[:,24*7:24*7+24])#np.mean(np.mean((X1-X2)**2,1))
        #print 'Regression ', melCD(X2,YH)#np.mean(np.mean((YH-X2)**2,1))
        #print 'Baseline! ', melCD(X1,X2)#np.mean(np.mean((X1-X2)**2,1))
        if np.mean(np.mean((YH-X2)**2,1)) < last_reg:
            print 'This is better. Saving the model to ' + dataset+'.dnn.pkl'
            f = open(out_file,'w+')
            pickle.dump(classifier, f)
            f.flush()
            f.close()
            last_reg = np.mean(np.mean((YH-X2)**2,1))
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    print ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

   
            


def model4_pre_from_siae(inp_file, midlayer, train_x, train_y): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    f=open(inp_file,'r')
    sda=pickle.load(f)
    f.close()
    hidden_layers_sizes = [sda.dA_layers[0].n_visible]
    for i in range(len(sda.dA_layers)):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    for i in range(len(midlayer)):
        hidden_layers_sizes.append(midlayer[i])
    for i in range(len(sda.dA_layers)-1, -1, -1):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    hidden_layers_sizes.append(sda.dA_layers[0].n_visible)
    random_seed = 1234
    rng = np.random.RandomState(random_seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    from dnn_dropout import MLP

    pretrained = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=[0.0]*len(hidden_layers_sizes),
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True
                     )
    for i in range(len(sda.dA_layers)):
        pretrained.layers[i].W = sda.dA_layers[i].W
        pretrained.layers[i].b = sda.dA_layers[i].b
    for i in range(len(sda.dA_layers)):
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].W = sda.dA_layers[len(sda.dA_layers)-i-1].W_prime
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].b = sda.dA_layers[len(sda.dA_layers)-i-1].b_prime
    middle_layer_size = [sda.dA_layers[-1].n_hidden]
    for i in range(len(midlayer)):
        middle_layer_size.append(midlayer[i])
    middle_layer_size.append(sda.dA_layers[-1].n_hidden)
    xhid= T.matrix('x')
    yhid= T.matrix('x')

    inbetween = MLP(rng=rng, input=xhid,
                     layer_sizes=middle_layer_size,
                     dropout_rates=[0.0]*len(middle_layer_size),
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True,
                     pretrained=None)
    
    encodex = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_x
                }
            )
    encodey = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_y
                }
            )
    xhid_value = encodex()
    yhid_value = encodey()
    xhid_value = theano.shared(xhid_value)
    yhid_value = theano.shared(yhid_value)

    cost=inbetween.mse(yhid)
    gparams = []
    for param in inbetween.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)
        
    updates = [
            (param, param - 0.1 * gparam)
            for param, gparam in zip(inbetween.params, gparams)
        ]

    train_model = theano.function(inputs=[], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value,
                                      yhid: yhid_value})
    data_len = train_x.eval().shape[0]
    batch_size = 10
    index = T.lscalar()
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value[index * batch_size:(index + 1) * batch_size],
                                      yhid: yhid_value[index * batch_size:(index + 1) * batch_size]})
    for i in xrange(100):##$
        cs = 0.0
        for j in xrange(data_len/batch_size):
            cs += train_model(j)
        print i, cs
    for i in range(len(midlayer)+1):
        pretrained.layers[len(sda.dA_layers)+i].W = inbetween.layers[i].W
        pretrained.layers[len(sda.dA_layers)+i].b = inbetween.layers[i].b
    return pretrained

def model4_pre_from_speaker20(inp_file, midlayer=None, train_x=None, train_y=None): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    f=open(inp_file,'r')
    pretrained=pickle.load(f)
    return pretrained

def model3_pre_from_siae(inp_file, midlayer, train_x, train_y): # 2.(A, Ba, Bb_backprop, C_backprop),MCEP15
    f=open(inp_file,'r')
    sda=pickle.load(f)
    f.close()
    hidden_layers_sizes = [sda.dA_layers[0].n_visible]
    for i in range(len(sda.dA_layers)):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    for i in range(len(midlayer)):
        hidden_layers_sizes.append(midlayer[i])
    for i in range(len(sda.dA_layers)-1, -1, -1):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    hidden_layers_sizes.append(sda.dA_layers[0].n_visible)
    random_seed = 1234
    rng = np.random.RandomState(random_seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    from dnn_dropout import MLP

    pretrained = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=[0.0]*len(hidden_layers_sizes),
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True
                     )
    for i in range(len(sda.dA_layers)):
        pretrained.layers[i].W = sda.dA_layers[i].W
        pretrained.layers[i].b = sda.dA_layers[i].b
    for i in range(len(sda.dA_layers)):
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].W = sda.dA_layers[len(sda.dA_layers)-i-1].W_prime
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].b = sda.dA_layers[len(sda.dA_layers)-i-1].b_prime
    middle_layer_size = [sda.dA_layers[-1].n_hidden]
    for i in range(len(midlayer)):
        middle_layer_size.append(midlayer[i])
    middle_layer_size.append(sda.dA_layers[-1].n_hidden)
    xhid= T.matrix('x')
    yhid= T.matrix('x')
    from dnn_dropout import HiddenLayer
    inbetween = HiddenLayer(rng=rng, input=xhid,
                     n_in=sda.dA_layers[-1].n_hidden,
                     n_out=sda.dA_layers[-1].n_hidden,
                     activation=CUR_ACIVATION_FUNCTION,
                     use_bias=True
                     )
    
    encodex = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_x
                }
            )
    encodey = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_y
                }
            )
    xhid_value = encodex()
    yhid_value = encodey()
    xhid_value = theano.shared(xhid_value)
    yhid_value = theano.shared(yhid_value)

    cost=inbetween.mse(yhid)
    gparams = []
    for param in inbetween.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)
        
    updates = [
            (param, param - 0.1 * gparam)
            for param, gparam in zip(inbetween.params, gparams)
        ]

    train_model = theano.function(inputs=[], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value,
                                      yhid: yhid_value})
    data_len = train_x.eval().shape[0]
    batch_size = 10
    index = T.lscalar()
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value[index * batch_size:(index + 1) * batch_size],
                                      yhid: yhid_value[index * batch_size:(index + 1) * batch_size]})
    for i in xrange(100):##$
        cs = 0.0
        for j in xrange(data_len/batch_size):
            cs += train_model(j)
        print i, cs
    for i in range(len(midlayer)+1):
        pretrained.layers[len(sda.dA_layers)+i].W = inbetween.layers[i].W
        pretrained.layers[len(sda.dA_layers)+i].b = inbetween.layers[i].b
    return pretrained

def model3_pre_from_speaker20(inp_file, midlayer=None, train_x=None, train_y=None): # 2.(A, Ba, Bb_backprop, C_backprop),MCEP15
    f=open(inp_file,'r')
    pretrained=pickle.load(f)
    return pretrained




def model4_pre_from_siae(inp_file, midlayer, train_x, train_y): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    f=open(inp_file,'r')
    sda=pickle.load(f)
    f.close()
    hidden_layers_sizes = [sda.dA_layers[0].n_visible]
    for i in range(len(sda.dA_layers)):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    for i in range(len(midlayer)):
        hidden_layers_sizes.append(midlayer[i])
    for i in range(len(sda.dA_layers)-1, -1, -1):
        hidden_layers_sizes.append(sda.dA_layers[i].n_hidden)
    hidden_layers_sizes.append(sda.dA_layers[0].n_visible)
    random_seed = 1234
    rng = np.random.RandomState(random_seed)
    x = T.matrix('x')  # the data is presented as rasterized images
    from dnn_dropout import MLP

    pretrained = MLP(rng=rng, input=x,
                     layer_sizes=hidden_layers_sizes,
                     dropout_rates=[0.0]*len(hidden_layers_sizes),
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True
                     )
    for i in range(len(sda.dA_layers)):
        pretrained.layers[i].W = sda.dA_layers[i].W
        pretrained.layers[i].b = sda.dA_layers[i].b
    for i in range(len(sda.dA_layers)):
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].W = sda.dA_layers[len(sda.dA_layers)-i-1].W_prime
        pretrained.layers[len(sda.dA_layers)+len(midlayer)+i+1].b = sda.dA_layers[len(sda.dA_layers)-i-1].b_prime
    middle_layer_size = [sda.dA_layers[-1].n_hidden]
    for i in range(len(midlayer)):
        middle_layer_size.append(midlayer[i])
    middle_layer_size.append(sda.dA_layers[-1].n_hidden)
    xhid= T.matrix('x')
    yhid= T.matrix('x')

    inbetween = MLP(rng=rng, input=xhid,
                     layer_sizes=middle_layer_size,
                     dropout_rates=[0.0]*len(middle_layer_size),
                     activations=[CUR_ACIVATION_FUNCTION]*len(hidden_layers_sizes),
                     use_bias=True,
                     pretrained=None)
    
    encodex = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_x
                }
            )
    encodey = theano.function(
                inputs=[],
                outputs=sda.dA_layers[-1].xhid,
                givens={
                    sda.dA_layers[0].x: train_y
                }
            )
    xhid_value = encodex()
    yhid_value = encodey()
    xhid_value = theano.shared(xhid_value)
    yhid_value = theano.shared(yhid_value)

    cost=inbetween.mse(yhid)
    gparams = []
    for param in inbetween.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)
        
    updates = [
            (param, param - 0.1 * gparam)
            for param, gparam in zip(inbetween.params, gparams)
        ]

    train_model = theano.function(inputs=[], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value,
                                      yhid: yhid_value})
    data_len = train_x.eval().shape[0]
    batch_size = 10
    index = T.lscalar()
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={
                                      xhid: xhid_value[index * batch_size:(index + 1) * batch_size],
                                      yhid: yhid_value[index * batch_size:(index + 1) * batch_size]})
    for i in xrange(100):##$
        cs = 0.0
        for j in xrange(data_len/batch_size):
            cs += train_model(j)
        print i, cs
    for i in range(len(midlayer)+1):
        pretrained.layers[len(sda.dA_layers)+i].W = inbetween.layers[i].W
        pretrained.layers[len(sda.dA_layers)+i].b = inbetween.layers[i].b
    return pretrained

def model4_pre_from_speaker20(inp_file, midlayer=None, train_x=None, train_y=None): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    f=open(inp_file,'r')
    pretrained=pickle.load(f)
    return pretrained


def model3_pre(): # 2.(A, Ba, Bb_backprop, C_backprop),MCEP15
    pass

def model4_pre(): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    pass

def model5_pre(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP15
    pass

def model6_pre(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP
    pass

def experiment():
    ae_name = 'ae_1000.pkl'
    if 1: # train all TIMIT AE
        ae_all(ae_name, hidden_layers_sizes=[1000],               
               corruption_levels=[0.1,0.2],
               pretrain_lr=0.1,
               batch_size=20,
               training_epochs=10)
    if 1: # load norm
        f=open('norm.pkl','r')
        mins=pickle.load(f)#[24*7:24*7+24]##$
        ranges=pickle.load(f)#[24*7:24*7+24]##$
        f.close()
    if 1: # load xy20
        x20, y20, xv20, yv20, xt20, yt20 = load_xy('clb2slt_pre.npy', num_sentences=100, mins=mins, ranges=ranges)
    if 1: # load xy
        x, y, xv, yv, xt, yt = load_xy('clb2slt.npy', num_sentences=100, mins=mins, ranges=ranges)
    
    if 1:
        dnn_train(model3_pre_from_siae, ae_name, 'dnn3_spk20.pkl',
                  x20, y20, xv20, yv20, xt20, yt20, mins, ranges,
                  hidden_layers_sizes=[x.eval().shape[1], 1000, 100, 1000, y.eval().shape[1]],
                  middle_layers_sizes=[100],
                  corruption_levels=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  lr=0.1,
                  batch_size=10,
                  training_epochs=100)
    dnn_train(model3_pre_from_speaker20, 'dnn3_spk20.pkl', 'dnn3.pkl', x, y, xv, yv, xt, yt, mins, ranges,
              hidden_layers_sizes=[x.eval().shape[1], 1000, 100, 1000, y.eval().shape[1]],
              middle_layers_sizes=[100],
               corruption_levels=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               lr=0.1,
               batch_size=10,
               training_epochs=1000)
  
    
if __name__ == "__main__":
    experiment()