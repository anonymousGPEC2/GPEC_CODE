import numpy as np
import pandas as pd
import torch

import matplotlib.cm as cm
import seaborn as sns
import time

sns.set_theme()
rc={'font.size': 19, 'axes.labelsize': 20, 'legend.fontsize': 18, 
    'axes.titlesize': 21, 'xtick.labelsize': 17, 'ytick.labelsize': 17}
sns.set(rc=rc)
sns.set_style('white')

from tqdm import tqdm
import os
import sys
from argparse import ArgumentParser 

os.chdir('../../')
sys.path.append('./')
from GPEC.utils import * # utility functions
from GPEC import * # GPEC functions
from GPEC.utils import utils_tests # utility functions



parser = ArgumentParser(description='Kernel Tests')

parser.add_argument('--method', type = str,default='census_Age_Hours',
                    help='germancredit_3_1')

parser.add_argument('--explainer', type = str,default='bayesshap',
                    help='')

parser.add_argument('--n_train_samples', type = int,default=100, help='number of training samples for GP')
parser.add_argument('--lam', type = float,default=0.5,
                    help='lambda parameter for kernel')
parser.add_argument('--rho', type = float,default=0.1,
                    help='rho parameter for kernel')
parser.add_argument('--n_test_samples', type=int, default=10000, help='number of test samples')
parser.add_argument('--n_iterations', type = int, default = 50)
parser.add_argument('--kernel', type = str,default='WEG',
                    help='')
parser.add_argument('--kernel_normalization', type = int,default=1, help='normalize kernel s.t. k(x,x)=1')
parser.add_argument('--max_batch_size', type = int,default=1024, help='Max number of GPs to train simultaneously. Number of batches == #features / max_batch_size')
parser.add_argument('--plot_explanations', type = int,default=0, help='flag to plot explanations')
parser.add_argument('--plot_flag', type = int,default=0, help='flag to save plots. overrides plot_explanations.')
parser.add_argument('--plot_feat', type = int,default=1, help='which feature to plot (0 or 1)')
parser.add_argument('--save_data', type = int,default=1, help='flag to save output')

#########
parser.add_argument('--use_gpec', type = int,default=1, help='flag to use GPEC')
parser.add_argument('--use_labelnoise', type = int,default=1, help='flag to use label noise (if using GPEC). only implemented for bayesshap, bayeslime, cxplain.')
parser.add_argument('--n_labelnoise_samples', type = int,default=10, help='if using labelnoise and explainer does not return uncertainty. Number of explanations to get from explainer for uncertainty estimate.')
parser.add_argument('--n_mc_samples', type = int,default=200, help='number of samples for approximating explanations')
parser.add_argument('--gpec_lr', type = float,default=1.0,help='Learning Rate for GPEC')
parser.add_argument('--learn_noise', type = int,default= 0, help='learn heteroskedastic GP noise for labels')

args = parser.parse_args()
utils_io.print_args(args)

# cxplain, bayeslime, bayesshap can use labelnoise. and they can export uncertainty.
# shapleysampling can use labelnoise. it cannot export uncertainty.
# kernelshap is not implemented. it cannot export uncertainty.
if args.use_labelnoise == 1 and args.explainer == 'kernelshap':
    raise ValueError('LabelNoise not implemented for KernelSHAP.')
if args.use_gpec == 0 and args.explainer == 'kernelshap':
    raise ValueError('KernelSHAP does not have uncertainty estimate by itself.')

if args.use_labelnoise == 1:
    n_labelnoise_samples = args.n_labelnoise_samples
else:
    n_labelnoise_samples = 1

lam = args.lam
rho = args.rho
if args.use_gpec == 0:
    lam = rho = 'NA'
plotfeat = args.plot_feat

if args.kernel_normalization == 1:
    kernel_normalization = True
else:
    kernel_normalization = False

plot_train = True
if args.explainer == 'kernelshap':
    output_shape = 'singleclass'
elif args.explainer == 'lime':
    output_shape = 'multiclass'
else:
    output_shape = 'multiclass'

'''
###############################################
  _____        _           _____      _               
 |  __ \      | |         / ____|    | |              
 | |  | | __ _| |_ __ _  | (___   ___| |_ _   _ _ __  
 | |  | |/ _` | __/ _` |  \___ \ / _ \ __| | | | '_ \ 
 | |__| | (_| | || (_| |  ____) |  __/ |_| |_| | |_) |
 |_____/ \__,_|\__\__,_| |_____/ \___|\__|\__,_| .__/ 
                                               | |    
                                               |_|    
###############################################
'''

if args.method == 'cosinv':

    from Tests.Models import synthetic_cosinv
    f_blackbox = synthetic_cosinv.model(output_shape = output_shape)

    dataset_name = 'cosinv'
    post_str = ''
    geo_matrix = np.load('./Files/Models/%s_geomatrix%s.npy' % (dataset_name, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples%s.npy' % (dataset_name, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')
    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')

    # synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    xmax = ymax = 10
    xmin = ymin = -10
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    feat1 = 'x1'
    feat2 = 'x2'
    decision_threshold = 0
    xmin, xmax, ymin, ymax = x_train[:,0].min(), x_train[:,0].max(), x_train[:,1].min(), x_train[:,1].max()
    axislim = [xmin, xmax, ymin, ymax]

elif args.method[:6] == 'census':

    if args.method == 'census_Age_Hours':
        feat1 = 'Age'
        feat2 = 'Hours per week'
        post_str = ''
        axislim = [20, 70, 20, 75] # axislim = [xmin, xmax, ymin, ymax]
    elif args.method == 'census_Age_Education':
        feat1 = 'Age'
        feat2 = 'Education-Num'
        post_str = ''
        axislim = [20, 70, 8, 16] # axislim = [xmin, xmax, ymin, ymax]
    elif args.method == 'census_Age_Hours_reg':
        feat1 = 'Age'
        feat2 = 'Hours per week'
        post_str = '_reg'
        axislim = [20, 70, 20, 75] # axislim = [xmin, xmax, ymin, ymax]
    dataset_name = 'census'

    # Load Pretrained Model
    from Tests.Models import xgb_models
    model_path = './Files/Models/model_census_%s_%s%s.json' % (feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    # Load Geo Matrix and Manifold Samples
    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    # Load Data
    x_train = pd.read_pickle('./Files/Data/%s_x_train.pkl' % dataset_name)
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv'% dataset_name)
    x_test = pd.read_pickle('./Files/Data/%s_x_test.pkl'% dataset_name)
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv'% dataset_name)
    x_train = x_train[[feat1, feat2]].to_numpy()
    x_test = x_test[[feat1, feat2]].to_numpy()

    # Create synthetic test data
    xmin, xmax, ymin, ymax = x_train[:,0].min()*1.2, x_train[:,0].max()*0.8, x_train[:,1].min(), x_train[:,1].max()
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    decision_threshold = 0.5

elif args.method[:6] == 'german':

    dataset_name = 'germancredit'
    if args.method == '%s_3_1' % dataset_name:
        feat1 = 3
        feat2 = 1
        post_str = ''
        axislim = [0, 150, 0, 50] # axislim = [xmin, xmax, ymin, ymax]

    from Tests.Models import xgb_models
    model_path = './Files/Models/model_%s_%s_%s%s.json' % (dataset_name, feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')
    x_train = x_train[:,[feat1, feat2]]
    x_test = x_test[:,[feat1, feat2]]

    # synthetic test data
    xmin, xmax = 0,170
    ymin, ymax = 0,60
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1


    decision_threshold = 0.5

elif args.method[:6] == 'online':

    dataset_name = 'onlineshoppers'
    if args.method == '%s_4_8' % dataset_name:
        feat1 = 4
        feat2 = 8
        post_str = ''
        axislim = [0, 100, 0, 80] # axislim = [xmin, xmax, ymin, ymax]

    from Tests.Models import xgb_models
    model_path = './Files/Models/model_%s_%s_%s%s.json' % (dataset_name, feat1, feat2, post_str)
    f_blackbox = xgb_models.xgboost_wrapper(model_path, output_shape = output_shape)

    geo_matrix = np.load('./Files/Models/%s_geomatrix_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))
    manifold_samples = np.load('./Files/Models/%s_samples_%s_%s%s.npy' % (dataset_name, feat1, feat2, post_str))

    x_train = np.loadtxt('./Files/Data/%s_x_train.csv' % (dataset_name), delimiter = ',')
    y_train = np.loadtxt('./Files/Data/%s_y_train.csv' % (dataset_name), delimiter = ',')

    x_test = np.loadtxt('./Files/Data/%s_x_test.csv' % (dataset_name), delimiter = ',')
    y_test = np.loadtxt('./Files/Data/%s_y_test.csv' % (dataset_name), delimiter = ',')
    x_train = x_train[:,[feat1, feat2]]
    x_test = x_test[:,[feat1, feat2]]

    # synthetic test data
    xmin, xmax = 0,150
    ymin, ymax = 0, 80
    int_x = (xmax-xmin) / 100
    int_y = (ymax-ymin) / 100
    xx, yy = np.mgrid[xmin:xmax:int_x, ymin:ymax:int_y]
    grid = np.c_[xx.ravel(), yy.ravel()]
    x_test = grid
    if output_shape == 'singleclass':
        y_test = (f_blackbox(x_test) >=0.5)*1
    else:
        y_test = (f_blackbox(x_test)[:,1] >= 0.5)*1

    decision_threshold = 0.5


# limit train/test samples if specified
x_train, y_train = utils_np.subsample_rows(matrix1 = x_train, matrix2 = y_train, max_rows = args.n_train_samples)
x_test,y_test = utils_np.subsample_rows(matrix1 = x_test, matrix2 = y_test, max_rows = args.n_test_samples)

'''
###############################################
  ______            _       _       
 |  ____|          | |     (_)      
 | |__  __  ___ __ | | __ _ _ _ __  
 |  __| \ \/ / '_ \| |/ _` | | '_ \ 
 | |____ >  <| |_) | | (_| | | | | |
 |______/_/\_\ .__/|_|\__,_|_|_| |_|
             | |                    
             |_|                    
###############################################
'''
print('=================================')
print('Generating Explanations...')

# if using GPEC calculate explanations on training set. Otherwise test set.
if args.use_gpec == 1:
    x_tmp = x_train
else:
    x_tmp = x_test

if args.explainer == 'kernelshap':
    explainer = explainers.kernelshap(f_blackbox, x_train)
    attr_list = explainer(x_tmp)
    var_list = None
    ci_list = None
elif args.explainer == 'lime':
    explainer = explainers.tabularlime(f_blackbox, x_train)
    attr_list = explainer(x_tmp)
    var_list = None
    ci_list = None
elif args.explainer == 'shapleysampling':
    sys.path.append('../BivariateShapley/BivariateShapley')
    from shapley_sampling import Shapley_Sampling
    from shapley_datasets import *
    from utils_shapley import *
    from shapley_explainers import XGB_Explainer
    # Initialize Explainer
    #baseline = x_train.mean(axis = 0).reshape(1,-1)
    baseline = 'mean'
    dataset = pd.DataFrame(x_train)
    Explainer = XGB_Explainer(model_path = model_path, baseline = baseline, dataset = dataset, m = args.n_mc_samples)

    # Get uncertainty estimate from a standard explainer
    labelnoise_list = []
    for j in range(n_labelnoise_samples):
        attr_list = []
        for i,x_sample in tqdm(enumerate(x_tmp), total = x_tmp.shape[0]):
            shapley_values, _ = Explainer(x_sample.reshape(1,-1))
            attr_list.append(shapley_values)
        attr_list = np.array(attr_list)
        labelnoise_list.append(attr_list)
    labelnoise_list = np.array(labelnoise_list) # m x n x d
    
    if args.use_labelnoise == 1:
        var_list = labelnoise_list.var(axis = 0)
        
        if n_labelnoise_samples > 50:
            # if there are enough samples, estimate empirically
            ci_list = np.quantile(labelnoise_list, .95, axis = 0) - np.quantile(labelnoise_list, .05, axis = 0)
        else:
            ci_list = (var_list ** 0.5)*2
    else:
        var_list = None
        ci_list = None

elif args.explainer == 'bayesshap' or args.explainer == 'bayeslime':
    if args.explainer == 'bayesshap':
        kernel = 'shap'
    else:
        kernel = 'lime'
    sys.path.append('../Modeling-Uncertainty-Local-Explainability')
    from bayes.explanations import BayesLocalExplanations, explain_many
    from bayes.data_routines import get_dataset_by_name
    exp_init = BayesLocalExplanations(training_data=x_train,
                                                data="tabular",
                                                kernel=kernel,
                                                categorical_features=np.arange(x_train.shape[1]),
                                                verbose=True)
    ci_list = []
    attr_list = []
    var_list = []
    for i,x_sample in tqdm(enumerate(x_tmp), total = x_tmp.shape[0]):
        rout = exp_init.explain(classifier_f=f_blackbox,
                                data=x_sample,
                                label=int(y_test[0]),
                                #cred_width=cred_width,
                                n_samples = int(args.n_mc_samples),
                                focus_sample=False,
                                l2=False)
        ci_list.append(rout['blr'].creds)
        attr_list.append(rout['blr'].coef_)
        var_list.append(rout['blr'].draw_posterior_samples(num_samples = 10000).var(axis = 0))


    ci_list = np.array(ci_list)
    attr_list = np.array(attr_list)
    var_list = np.array(var_list)

elif args.explainer == 'cxplain':
    sys.path.append('../cxplain')
    from tensorflow.python.keras.losses import categorical_crossentropy
    from cxplain import MLPModelBuilder, ZeroMasking, CXPlain

    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
    import tensorflow as tf
    tf.compat.v1.experimental.output_all_intermediates(True)

    model_builder = MLPModelBuilder(num_layers=2, num_units=24, activation="selu", p_dropout=0.2, verbose=0,
                                    batch_size=8, learning_rate=0.01, num_epochs=250, early_stopping_patience=15)
    masking_operation = ZeroMasking()
    loss = categorical_crossentropy

    explainer = CXPlain(f_blackbox, model_builder, masking_operation, loss, num_models=10)
    explainer.fit(x_train, y_train);
    attributions, confidence = explainer.explain(x_tmp, confidence_level=0.95)
    # attributions are n x d.
    #confidence is shape n x d x 2. for each sample/feature, ...0 = lower bound, ...1 = upper bound. Calculate Upper - Lower to get width. 

    ci_list = confidence[...,1] - confidence[...,0]
    attr_list = attributions
    var_list = (ci_list / 2)**2


if args.use_gpec == 1:
    '''
    ###############################################
      _____ _____  ______ _____ 
     / ____|  __ \|  ____/ ____|
    | |  __| |__) | |__ | |     
    | | |_ |  ___/|  __|| |     
    | |__| | |    | |___| |____ 
     \_____|_|    |______\_____|
    ###############################################
    '''
    pred_list = []
    gpec_attr_list = []
    gpec_var_list = []
    gpec_ci_list = []
    data, labels = utils_torch.numpy2cuda(x_train), utils_torch.numpy2cuda(attr_list) 
    # data should be n x d
    # labels should be n x d

    max_batch_size = args.max_batch_size # Max number of GP models to train simultaneously
    batched_labels = torch.split(labels, max_batch_size, dim = 1)

    print('=================================')
    print('Training GP...')
    print('Number of Batches: %s' % str(len(batched_labels)))

    timestamp_start = time.time()
    for i, batch_y in tqdm(enumerate(batched_labels), position = 0, desc = 'Batch Progress'):

        # reshape data and labels by batch
        batch_size = batch_y.shape[1]
        batch_shape = torch.Size([batch_size])
        tmp_x = data.unsqueeze(0).expand(batch_size,-1,-1)
        tmp_y = batch_y.t()
        # data should be b x n x d
        # labels should be b x n. Features should be in batch dimension.

        # Variance List
        min_var = np.zeros_like(tmp_y) + 1e-8 # for numberical stability
        if args.use_labelnoise == 1:
            tmp_var_list = var_list.transpose()
            tmp_var_list = np.maximum(tmp_var_list, min_var)
        else:
            tmp_var_list = min_var
        if args.kernel == 'RBF' or args.learn_noise == 1: tmp_var_list = None

        model, likelihood = GP.train_GPEC(tmp_x, tmp_y, manifold_samples, geo_matrix, var_list = tmp_var_list, kernel = args.kernel, n_iter = args.n_iterations, lam = args.lam, rho = args.rho, kernel_normalization = kernel_normalization, batch_shape = batch_shape, lr = args.gpec_lr)

        time_train = time.time() - timestamp_start
        timestamp_start = time.time()

        # Predictions
        model.eval()
        likelihood.eval()
        pred_list.append(model(utils_torch.numpy2cuda(x_test).float()))
        time_pred = time.time() - timestamp_start
        timestamp_start = time.time()
        
        # mean
        gpec_attr_list.append(pred_list[i].mean.cpu().detach().numpy())
        time_mean = time.time() - timestamp_start
        timestamp_start = time.time()
        
        # variance
        gpec_var_list.append(pred_list[i].variance.cpu().detach().numpy())
        time_var = time.time() - timestamp_start
        timestamp_start = time.time()
        
        # confidence interval
        ci_lower = pred_list[i].confidence_region()[0].cpu().detach().numpy()
        ci_upper = pred_list[i].confidence_region()[1].cpu().detach().numpy()
        gpec_ci_list.append(ci_upper - ci_lower)
        time_ci = time.time() - timestamp_start

    # concatenate predictions
    gpec_attr_list = np.concatenate(gpec_attr_list, axis = 0).transpose()
    gpec_ci_list = np.concatenate(gpec_ci_list, axis = 0).transpose()
    gpec_var_list = np.concatenate(gpec_var_list, axis = 0).transpose()

'''
###############################################
   _____                 
  / ____|                
 | (___   __ ___   _____ 
  \___ \ / _` \ \ / / _ \
  ____) | (_| |\ V /  __/
 |_____/ \__,_| \_/ \___|
###############################################
'''
feat_list = [plotfeat]
if args.plot_flag == 1:
    ###############################################
    # Plot
    ###############################################
    #sns.cubehelix_palette(as_cmap=True)
    # coolwarm

    if args.use_gpec == 1:
        plot_unc_list = gpec_ci_list
    else:
        plot_unc_list = ci_list
    filename = '_'.join([
    args.kernel,
    args.method,
    args.explainer,
    'rho'+str(rho),
    'lam'+str(lam),
    ])
    save_path = './Files/Results/uncertaintyplot/%s/%s/%s/%s/%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat), 'gpec'+str(args.use_gpec), 'labelnoise'+str(args.use_labelnoise), filename)
    utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = plot_unc_list, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, rho = args.rho, lam = args.lam, plot_train = True, axislim = axislim)

    ###############################################
    # Plot Explanations
    ###############################################
    if args.plot_explanations == 1:
        if args.explainer not in ['kernelshap', 'lime']:
            raise ValueError('plot_explanations not yet implemented')
        
        # plot uncertainty
        exp_test = explainer(x_test) # get test explanations
        exp_test = exp_test[:,feat_list] # plot only one feature

        save_path = './Files/Results/uncertaintyplot/%s/%s/%s/explanations_%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat) ,str(args.kernel_normalization), filename)
        utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = exp_test, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, cmap = cm.coolwarm, rho = args.rho, lam = args.lam, plot_train = True, center_cmap = True, center = 0)
        
    # Plot model output
    if output_shape == 'multiclass':
        output_list = f_blackbox(x_test)[:,1].reshape(-1,1)
    else:
        output_list = f_blackbox(x_test).reshape(-1,1)

    save_path = './Files/Results/uncertaintyplot/%s/%s/%s/%s/output_%s.jpg' % (args.method, 'plotfeat'+ str(plotfeat), str(args.use_gpec), str(args.use_labelnoise), str(args.method))
    utils_tests.uncertaintyplot(x_train = x_train, x_test = x_test, hue_list = output_list, save_path = save_path, f_blackbox = f_blackbox, feat_list = feat_list, cmap = cm.coolwarm, rho = args.rho, lam = args.lam, plot_train = True, center_cmap=True, center = decision_threshold, axislim = axislim)


###############################################
# Save Data for Figure
###############################################
if args.save_data == 1:
    '''
    if args.plot_flag ==0:
        exp_test = explainer(x_test)
        exp_test = exp_test[:,feat_list]
    '''
    # Plot model output
    if output_shape == 'multiclass':
        output_list = f_blackbox(x_test)[:,1].reshape(-1,1)
    else:
        output_list = f_blackbox(x_test).reshape(-1,1)

    if args.use_gpec == 1:
        attr_list = gpec_attr_list
        ci_list = gpec_ci_list
        var_list = gpec_var_list

    saved_data = {
        'x_train': x_train,
        'x_test': x_test,
        'attr_list': attr_list, # explanations from explainer
        'ci_list': ci_list, # ci from explainer
        'var_list': var_list, # variance from explainer
        'output_list': output_list, # black-box model output for test points
        'rho': rho,
        'lam': lam,
        'method': args.method,
        'explainer': args.explainer,
        'kernel': args.kernel,
        'xx': xx,
        'yy': yy,
        'feat1': feat1,
        'feat2': feat2,
        #'time_train': time_train,
        #'time_pred': time_pred,
        #'time_attr': time_mean,
        #'time_var': time_var,
        #'time_ci': time_ci,
    }

    filename = '_'.join([
        args.method,
        args.explainer,
        args.kernel,
        'rho'+str(rho),
        'lam'+str(lam),
        'uselabelnoise' + str(args.use_labelnoise),
        'mcsamples' + str(args.n_mc_samples),
        ])
    prepend = ''
    if args.use_labelnoise: prepend = 'labelnoise'
    save_path = './Files/Results/uncertaintyplot/saved_results_%s/%s.pkl' % (prepend, filename)
    foldername = os.path.dirname(save_path)
    utils_io.make_dir(foldername)
    utils_io.save_dict(saved_data, save_path)