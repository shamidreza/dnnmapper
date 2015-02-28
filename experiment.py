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
from utils import ReLU, Sigmoid, Tanh, SoftMax

CUR_ACIVATION_FUNCTION = Sigmoid

def model0(): # GMM,MCEP
    pass

def model1(): # 4,(C_backprop),MCEP
    pass

def model2(): # 1.(A, Bb_backprop, C_backprop, C_backprop),MCEP15
    pass

def model3(): # 2.(A, Ba, Bb_backprop, C_backprop),MCEP15
    pass

def model4(): # 3.(A, Ba, Bb_backprop, C_backprop,SLT14),MCEP15
    pass

def model5(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP15
    pass

def model6(): # 1,(A, Bb_joint, Bb_backprop, C_joint, C_backprop),MCEP
    pass

def experiment():
    if 1: # load all TIMIT
        timit_all = load_timit()
    if 1: # load xy20
        x20, y20 = load_xy20()
    if 1: # load xy
        x, y = load_xy20()
        
    if 1: # train all TIMIT AE
        pass
    
    models = [model0, model1, model2, model3, model4, model5, model6]
    
    for model in models:
        model(x20, y20, x, y)
    
    