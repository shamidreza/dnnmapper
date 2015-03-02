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

CUR_ACIVATION_FUNCTION = Sigmoid

def ae_all(hidden_layers_sizes=None,
           n_outs=None, 
           corruption_levels=None,
           pretrain_lr=None,
           batch_size=None,
           training_epochs=None): # ae all on TIMIT
    print '... loading the data'
    data=load_vc_all_speakers()
    print '... loaded data with dimensions', str(data.shape[0]),'x', str(data.shape[1])
    print '... normalizing the data'
    new_data, mins, ranges = normalize_data(data)
    numpy_rng = np.random.RandomState(89677)
    print '... building the model'
    from ae_stacked import SdA
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=new_data.shape[1],
        hidden_layers_sizes=hidden_layers_sizes,#[1000, 1000],
    )
    import theano
    n_train_batches = int(0.9*new_data.shape[0])
    n_train_batches /= batch_size
    new_data = new_data.astype(np.float32)
    train_set = theano.shared(new_data[:int(0.9*new_data.shape[0]), :])
    test_set = theano.shared(new_data[int(0.9*new_data.shape[0]):, :])
    test_set_unnormalized = unnormalize_data(new_data[int(0.9*new_data.shape[0]):, :], mins, ranges)
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
                outputs=sda.dA_layers[0].xh,
                givens={
                    sda.dA_layers[0].x: test_set
                }
            )
    ## Pre-train layer-wise
    #corruption_levels = [0.2, 0.3]
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
                XH = unnormalize_data(XH, mins, ranges)
                print 'melCD', melCD(XH, test_set_unnormalized)
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)
            

    end_time = time.clock()

    print 'The pretraining code for file ' +\
                          os.path.split(__file__)[1] +\
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)
    
    
def model0_pre(): # GMM,MCEP
    pass

def model1_pre(): # 4,(C_backprop),MCEP
    pass

def model2_pre(): # 1.(A, Bb_backprop, C_backprop),MCEP15
    pass

def model3_pre(): # 2.(A, Ba, Bb_backprop, C_backprop),MCEP15
    pass

def model4_pre(): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    pass

def model5_pre(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP15
    pass

def model6_pre(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP
    pass

def experiment():    
    if 0: # load xy20
        x20, y20 = load_xy20()
    if 0: # load xy
        x, y = load_xy20()
        
    if 1: # train all TIMIT AE
        ae_all(hidden_layers_sizes=[1000,1000],
               n_outs=500,
               corruption_levels=[0.2,0.3],
               pretrain_lr=0.05,
               batch_size=20,
               training_epochs=10)
    
    models = [model0, model1, model2, model3, model4, model5, model6]
    
    for model in models:
        model(x20, y20, x, y)
    
if __name__ == "__main__":
    experiment()