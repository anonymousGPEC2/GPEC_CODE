import numpy as np

class model():
    '''
    toy model with cosine function classifier. Input is 2d.

    '''

    def __init__(self, output_shape = 'singleclass'):
        self.output_shape = output_shape
    def __call__(self, x):
        # # numerical stability
        # output = np.zeros_like(x)
        # # numerical stability
        # idx_geq = np.where(np.abs(x[:,0]) >= 1e-6)[0]
        # idx_l = np.where(np.abs(x[:,0]) < 1e-6)[0]
        # output[idx_geq,0] = (2*np.cos(10/x[idx_geq,0]) - x[idx_geq,1])
        # output[idx_l,0] = np.zeros_like(x[idx_l,0]) - x[idx_l,1]
        x[:,0] = np.sign(x[:,0]) * np.maximum(np.abs(x[:,0]), np.zeros_like(x[:,0]) + 1e-20)

        # numerical stability
        idx = np.where(x[:,0] == 0)[0]
        # zero_vector = x[idx,0]
        # rand_sign = np.random.binomial(1,0.5,zero_vector.shape)*2-1
        # x[idx,0] = rand_sign * 1e-20
        x[idx,0] = 1e-20


        output = (2*np.cos(10/x[:,0]) - x[:,1])
        if self.output_shape == 'singleclass':
            return output
        else:
            return np.vstack((1-output, output)).transpose()
    def predict_proba(self, x):
        x[:,0] = np.sign(x[:,0]) * np.maximum(np.abs(x[:,0]), np.zeros_like(x[:,0]) + 1e-5) # numerical stability

        output = (2*np.cos(10/x[:,0]) - x[:,1])
        return np.vstack((1-output, output)).transpose()
    def grad(self,x):
        '''
        returns gradient. input should be n x 2
        '''

        # numerical stability
        x[:,0] = np.sign(x[:,0]) * np.maximum(np.abs(x[:,0]), np.zeros_like(x[:,0]) + 1e-20)

        # numerical stability
        idx = np.where(x[:,0] == 0)[0]
        # zero_vector = x[idx,0]
        # rand_sign = np.random.binomial(1,0.5,zero_vector.shape)*2-1
        # x[idx,0] = rand_sign * 1e-20
        x[idx,0] = 1e-20

        output = np.zeros_like(x)
        output[:,0] = (20 * np.sin(10/x[:,0])) / x[:,0] ** 2
        output[:,1] = -1
        return output



    def db(self, x):
        '''
        returns samples from the decision boundary

        input:
            x: 1d numpy vector

        return:
            np vector of dim n x 2
        '''
        x = np.sign(x) * np.maximum(np.abs(x), np.zeros_like(x) + 1e-5) # numerical stability
        y = 2*np.cos(10/x)
        return np.vstack((x,y)).transpose()

class data():
    '''
    generate synthetic data for toy models
    '''

    def __init__(self, toy_model, n_train = 5000, n_test = 2000, seed = 0):
        np.random.seed(seed)
        low = -10
        high = 10
        x = np.random.uniform(low = low, high = high, size = (n_train+n_test, 2))
        y = (toy_model(x)>0)*1

        self.x_train = x[:n_train,:]
        self.x_test = x[n_train:,:]
        self.y_train = y[:n_train]
        self.y_test = y[n_train:]

        # # create grid
        # xmin, xmax = low,high
        # ymin, ymax = low,high
        # int_x = (xmax-xmin) / (n_train**0.5)
        # int_y = (ymax-ymin) / (n_train**0.5)
        # xx, yy = np.mgrid[xmin:xmax+int_x:int_x, ymin:ymax+int_y:int_y]
        # grid = np.c_[xx.ravel(), yy.ravel()]
        # self.x_train = grid
        # self.y_train = (toy_model(self.x_train)>0)*1

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
