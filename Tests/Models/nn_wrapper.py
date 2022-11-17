import numpy as np
import torch
import sys
sys.path.append('./')
from GPEC.utils.utils_torch import *

class nn_wrapper():
    def __init__(self, model_path = None, model = None, output_shape = 'multiclass', output_type = 'default', target_class = 0):
        '''
        args:
            model_path: model path. either model_path or model required.
            model: xgboost model object. either model_path or model required.
            output_shape: "singleclass" or "multiclass". singleclass returns a vector, multiclass returns a matrix, where each column is the probability of that class.
            target_class: if output shape is singleclass, then target class must be provided to generate one vs all probability.
        '''
        if model_path is None and model is None:
            raise ValueError("Either model_path or model is required")
        if model_path is not None:
            raise ValueError('Not Implemented Yet.')
        else:
            self.model = model
        self.output_shape = output_shape
        self.target_class = target_class
        self.output_type = output_type
        print('done!')
    def __call__(self, x):
        '''
        args:
            x: n x d np matrix
        return:
            prediction
        '''
        if len(x.shape) == 1:
            # x must be a n x d matrix, not a vector.
            x = x.reshape(1,-1)
        
        input_type = type(x)
        if input_type == np.ndarray:
           x = numpy2cuda(x) 

        self.model.eval()
        output = self.model(x.type(dtype=torch.float32))
        if self.output_type == 'prob':
            output = F.softmax(output, dim = 1)
        if input_type == np.ndarray:
           output = tensor2numpy(output)

        if self.output_shape == 'singleclass':
            return output[:,self.target_class]
        elif self.output_shape == 'multiclass':
            return output
                
    def predict(self, x):
        '''
         Returns prediction as a vector. For use in some explainers which require predict function.
        '''
        input_type = type(x)
        if input_type == np.ndarray:
           x = numpy2cuda(x) 
        else:
            device = x.device
            x = tensor2cuda(x)

        output = self.model(x.type(dtype=torch.float32)).argmax(dim = 1)
        output = F.one_hot(output, num_classes = 10)
        if input_type == np.ndarray:
           output = tensor2numpy(output)
        else:
            output = output.to(device)
        return output