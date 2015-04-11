"""
The initial version of the code is downloaded from http://deeplearning.net/tutorial/code/utils.py on 2015-02-14.
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
""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
import pickle
import gzip
import os
import numpy as np
import theano
import theano.tensor as T
try:
    from matplotlib import pyplot as pp
except:
    print 'can not import matplotlib'
    
#### rectified linear unit
def ReLU(x):
    return T.maximum(0.0, x)
#### sigmoid
def Sigmoid(x):
    return T.nnet.sigmoid(x)
#### tanh
def Tanh(x):
    return T.tanh(x)
#### softmax
def SoftMax(x):
    return T.nnet.softmax(x)

def melCD(m1,m2):
    if m1.ndim == 1 and m1.ndim == 1:
	return (10.0/np.log(10.0))*(np.sqrt(2.0*np.sum((m1-m2)**2)))
    else:
	sum_distance = 0.0
	for i in range(m1.shape[0]):
	    sum_distance += (10.0/np.log(10.0))*(np.sqrt(2.0*np.sum((m1[i,:]-m2[i,:])**2)))
	return sum_distance/m1.shape[0]

def compute_normalization_factors(data):
    import numpy as np
    mins = np.zeros(data.shape[1],dtype=np.float32)
    ranges = np.zeros(data.shape[1],dtype=np.float32)

    for i in range(data.shape[1]):
	mins[i] = (data[:, i].mean())
	ranges[i] = (data[:, i].std())
    return mins, ranges

def compute_normalization_factors_neg1_1(data):
    import numpy as np
    mins = np.zeros(data.shape[1],dtype=np.float32)
    ranges = np.zeros(data.shape[1],dtype=np.float32)

    for i in range(data.shape[1]):
	mins[i] = (data[:, i].min())
	ranges[i] = ((data[:, i].max()) - mins[i])
    return mins, ranges

def normalize_data_neg1_1(data, mins, ranges):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
   
    for i in range(new_data.shape[1]):	
	new_data[:, i] -= mins[i]
	new_data[:, i] /= ranges[i]
	new_data[:, i] *= 2.0
	new_data[:, i] -= (1.0)
	#assert np.all(new_data[:, i] >= -1.0) and np.all(new_data[:, i] <= 1.0)
	new_data[new_data[:, i]>1.0, i] = 1.0
	new_data[new_data[:, i]<-1.0, i] = -1.0

    return data##new_data

def unnormalize_data_neg1_1(data, mins, ranges):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
    for i in range(new_data.shape[1]):
	new_data[:, i] += 1.0
	new_data[:, i] /= 2.0
	new_data[:, i] *= ranges[i]
	new_data[:, i] += mins[i]
    return data##new_data


def normalize_data_0_1(data):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
    mins = np.zeros(data.shape[1])
    ranges = np.zeros(data.shape[1])

    for i in range(new_data.shape[1]):
	mins[i] = (new_data[:, i].min())
	ranges[i] = (new_data[:, i].max()) - mins[i]
	new_data[:, i] -= mins[i]
	new_data[:, i] /= ranges[i]
	#assert np.all(new_data[:, i] >= 0.0) and np.all(new_data[:, i] <= 1.0)
    return new_data, mins, ranges
def unnormalize_data_0_1(data, mins, ranges):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
    for i in range(new_data.shape[1]):	
	new_data[:, i] *= ranges[i]
	new_data[:, i] += mins[i]
    return new_data

def normalize_data(data, mins, ranges):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
  
    for i in range(new_data.shape[1]):
	new_data[:, i] -= mins[i]
	new_data[:, i] /= (ranges[i]*3.0)
	#assert np.all(new_data[:, i] >= 0.0) and np.all(new_data[:, i] <= 1.0)
    return new_data
def unnormalize_data(data, mins, ranges):
    import numpy as np
    import copy
    new_data = copy.deepcopy(data)
    for i in range(new_data.shape[1]):	
	new_data[:, i] *= (ranges[i]*3.0)
	new_data[:, i] += mins[i]
    return new_data


def load_vc_all_speakers():
    from glob import iglob
    from os import path, popen
    from os.path import exists
    import pickle
    data = np.zeros((630*3500,24*15),dtype=np.float32)
    st=0
    cnt = 0
    if exists('../TIMIT_code/spk_wav/'):
	iter_directory = iglob('../TIMIT_code/spk_wav/M*.pkl')
    else:
	iter_directory = iglob('../gitlab/voice-conversion/src/spk_wav/M*.pkl')
    for fid in iter_directory:
	print'read_TIMIT_append_all: reaing file '+ fid
	f=open(fid, 'r')
	cur_fx=pickle.load(f)
	f.close()
	data[st:st+cur_fx.shape[0],:] = cur_fx
	st += cur_fx.shape[0]
	cnt+=1
	#if cnt > 10:
	#    break
    data = data[:st,:]
    return data
##def load_vc(dataset='c2s.npy', num_sentences=200):
def load_vc(dataset, num_sentences):
    #import sys
    #sys.path.append('../gitlab/voice-conversion/src')
    #import voice_conversion
    
    import pickle
    f=open(dataset,'r')
    #vcdata=pickle.load(f)
    #x=vcdata['aligned_data1'][:,:24]
    #y=vcdata['aligned_data2'][:,:24]
    x=numpy.load(f).astype(numpy.float32)
    y=numpy.load(f).astype(numpy.float32)
    #x=numpy.log(x)##
    #y=numpy.log(y)##
    f.close()
    num = x.shape[0]
    st_train = 0
    en_train = int(num * (num_sentences/200.0)) # 64 train,18 valid, 18 test
    #st_valid = en_train
    #en_valid = en_train+int(num * (18.0/100.0))
    st_test = num-+int(num * (50.0/200.0))
    en_test = num
    st_valid = en_train
    en_valid = en_train+int(num * (50.0/200.0))
    if 0:# not now
	x_mean = x[st_train:en_train,:].mean(axis=0)
	y_mean = y[st_train:en_train,:].mean(axis=0)
	x_std = x[st_train:en_train,:].std(axis=0)
	y_std = y[st_train:en_train,:].std(axis=0)
	x -= x_mean
	y -= y_mean
	x /= x_std
	y /= y_std

    import theano
    train_set_x = theano.shared(numpy.asarray(x[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y = theano.shared(numpy.asarray(y[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_x = theano.shared(numpy.asarray(x[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_y = theano.shared(numpy.asarray(y[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_x = theano.shared(numpy.asarray(x[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_y = theano.shared(numpy.asarray(y[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
def load_vc_siamese(dataset):
    #import sys
    #sys.path.append('../gitlab/voice-conversion/src')
    #import voice_conversion
    
    import pickle
    f=open(dataset,'r')
    #vcdata=pickle.load(f)
    #x=vcdata['aligned_data1'][:,:24]
    #y=vcdata['aligned_data2'][:,:24]
    x=np.load(f).astype(np.float32)[:6766924,:]
    y=np.load(f).astype(np.float32)[:6766924,:]
    spk=np.load(f).astype(np.float32)[:6766924,:]
    phon=np.load(f).astype(np.float32)[:6766924,:]
    #x=numpy.log(x)##
    #y=numpy.log(y)##
    f.close()
    num = x.shape[0]
    st_train = 0
    en_train = int(num * (90.0/100.0)) # 64 train,18 valid, 18 test
    #st_valid = en_train
    #en_valid = en_train+int(num * (18.0/100.0))
    st_test = en_train
    en_test = num
    st_valid = en_train
    en_valid = num
    if 0:# not now
	x_mean = x[st_train:en_train,:].mean(axis=0)
	y_mean = y[st_train:en_train,:].mean(axis=0)
	x_std = x[st_train:en_train,:].std(axis=0)
	y_std = y[st_train:en_train,:].std(axis=0)
	x -= x_mean
	y -= y_mean
	x /= x_std
	y /= y_std

    import theano
    train_set_x = theano.shared(np.asarray(x[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y = theano.shared(np.asarray(y[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_spk = theano.shared(np.asarray(spk[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_phon = theano.shared(np.asarray(phon[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_x = theano.shared(np.asarray(x[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_y = theano.shared(np.asarray(y[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_spk = theano.shared(np.asarray(spk[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_phon = theano.shared(np.asarray(phon[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_x = test_set_x
    valid_set_y = test_set_y
    valid_set_spk = test_set_spk
    valid_set_phon = test_set_phon
    rval = [(train_set_x, train_set_y, train_set_spk, train_set_phon),
            (valid_set_x, valid_set_y, valid_set_spk, valid_set_phon),
            (test_set_x, test_set_y, test_set_spk, test_set_phon)]
    return rval

def load_xy(dataset, num_sentences, mins, ranges):
    from utils import load_vc
    print '... loading the data'
    
    f=open(dataset,'r')
    x=np.load(f).astype(np.float32)#[:,24*7:24*7+24]##$
    y=np.load(f).astype(np.float32)#[:,24*7:24*7+24]##$
    
    f.close()
    
    x=normalize_data(x, mins, ranges)
    y=normalize_data(y, mins, ranges)

    num = x.shape[0]
    st_train = 0
    en_train = int(num * (num_sentences/200.0))
    st_test = num-int(num * (50.0/200.0))
    en_test = num
    st_valid = st_test-int(num * (50.0/200.0))
    en_valid = st_test
    import theano
    train_set_x = theano.shared(np.asarray(x[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    train_set_y = theano.shared(np.asarray(y[st_train:en_train,:],##
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_x = theano.shared(np.asarray(x[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    test_set_y = theano.shared(np.asarray(y[st_test:en_test,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_x = theano.shared(np.asarray(x[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    valid_set_y = theano.shared(np.asarray(y[st_valid:en_valid,:],
                                dtype=theano.config.floatX),
                                 borrow=True)
    return train_set_x, train_set_y, test_set_x, test_set_y, valid_set_x, valid_set_y

def load_mnist_half(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # LOAD DATA #
    import os
    import cPickle
    import gzip
    import theano
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, train_xy, borrow=True):       
        data_x, data_y = data_xy
        data_x = data_x.reshape((data_x.shape[0], 28,28))
        data_y = data_x[:,:,14:].reshape((data_x.shape[0], 28*14))
        data_x = data_x[:,:,:14].reshape((data_x.shape[0], 28*14))
        t_x, t_y = train_xy
        t_x = t_x.reshape((t_x.shape[0], 28,28))
        t_y = t_x[:,:,14:].reshape((t_x.shape[0], 28*14))
        t_x = t_x[:,:,:14].reshape((t_x.shape[0], 28*14))
        #data_x = data_x - t_x.mean(axis=0)
        #data_y = data_y - t_y.mean(axis=0)

        #for j in range(data_x.shape[1]):
            #data_x[:, j] -= numpy.mean(data_x[:, j])
        #for j in range(data_y.shape[1]):
            #data_y[:, j] -= numpy.mean(data_y[:, j])
        #data_x = data_x[:5000,:]
        #data_y = data_y[:5000,:]

        #data_y = data_y[:]

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        
        return shared_x, shared_y

    
    train_set_x, train_set_y = shared_dataset(train_set, train_set)
    test_set_x, test_set_y = shared_dataset(test_set, train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set, train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval




def load_mnist(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    # LOAD DATA #

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def plot_weights(w, M=28, N=28, num=10):
    import numpy as np
    try:
        from matplotlib import pyplot as pp
        import matplotlib.cm as cm
    except ImportError:
        print 'matplotlib is could not be imported'

    a=np.zeros((M*num,N*num))
    for i in range(num*num):
        m=i%num
        n=i/num
        a[m*M:(m+1)*M, n*N:(n+1)*N] = w[i,:].reshape((M,N))
    pp.imshow(a,interpolation='none',aspect='auto',cmap=cm.Greys)
    #pp.show()
    
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array