# Source code for GPEC experiments.

# GPEC Code Overview
The code for training GPEC is located in ./GPEC

* **decision_boundary.py** contains the algorithms for sampling the model decision boundary
* **explainers.py** contains the explainers used in conjunction with GPEC and also for comparison experiments
* **GP.py** contains the code to train the GP model parametrizing GPEC
* **riemannian_tree.py** contains the code for estimating geodesic distances (adapted from  https://github.com/redst4r/riemannian_latent_space)



# Experiments
The experiment code is located in ./Tests

## Setup
Due to file size constraints, the datasets must be downloaded separately and the models must be trained. For convenience, the data and model files for the census experiments have been uploaded to the Github repository. Other datasets can be downloaded from Pytorch (MNIST and f-MNIST) or from the UCI data repository. To train the black-box models used in the experiments section, the code is located in ./Tests/Models/blackbox_model_training

In addition, the following repositories should be downloaded (optional -- used for explainer comparisons in the experiments). In the folder one level above GPEC_Anonymous_Code, download the following repositories:
* **Modeling-Uncertainty-Local-Explainability** (Used for BayesLIME and BayesSHAP algorithms) https://github.com/dylan-slack/Modeling-Uncertainty-Local-Explainability
* **cxplain** (CXPlain algorithm) https://github.com/d909b/cxplain
* **BivariateShapley** (Used for Shapley Sampling Values algorithm) https://github.com/davinhill/BivariateShapley

## Uncertainty Figures (Figure 4)
These experiments estimate the uncertainty for a grid of test samples, which are visualized as a heatmap in Figure 4. The code is located in ./Tests/uncertaintyfigure/a_kernel_test_rev2.py. Running this file will save the results in ./Files/Results. Note that for performance reasons, the Decision Boundary samples and EG kernel matrix is precomputed in this experiment. The files for Census dataset can be downloaded from https://github.com/anonymousGPEC/GPEC. The files for other datasets can be calculated in ./Tests/Models/blackbox_model_training  



**a_kernel_test_rev2.py options**
> **--method** (specifies dataset)
> * census_Age_Hours: Census dataset for Age / Hours features
> * census_Age_Education: Census dataset for Age / Education features
> * cosinv: Cos model f(x,y) = 2*cos(10/x) - y
> * germancredit_3_1: German Credit Dataset for features 3 and 1
> * onlineshoppers_4_8: Online Shoppers Dataset for features 3 and 1
> 
> **--explainer** (Explainer to use in conjunction with GPEC)
> * kernelshap: KernelSHAP explainer
> * cxplain: CXPlain
> * bayeslime: BayesLIME
> * bayesshap: BayesSHAP
> * shapleysampling: Shapley Sampling Values
> 
> **--kernel**
> * WEG: WEG Kernel
> * RBF: RBF Kernel
> 
> **--use_gpec**
> * 0: Use GPEC
> * 1: Use uncertainty estimate from the explainer only (only implemented for cxplain, bayeslime, bayesshap)
> 
> **--use_labelnoise**
> * 0: Use GPEC with noiseless labels. I.E. only calculates decision boundary-aware uncertainty
> * 1: Incorporate function approximation uncertainty from the explainer
>
> **--lam**
> lambda parameter for GPEC
>
> **--rho**
> rho parameter for GPEC

**Example 1 (GPEC-WEG with KernelSHAP, only using DB-aware uncertainty):** a_kernel_test_rev2.py --method census_Age_Hours --explainer kernelshap --kernel WEG --use_gpec 1 --use_labelnoise 0

**Example 2 (GPEC-WEG with BayesSHAP, using both sources of uncertainty):** a_kernel_test_rev2.py --method census_Age_Hours --explainer bayesshap --kernel WEG --use_gpec 1 --use_labelnoise 1

**Example 3 (GPEC-RBF with BayesSHAP):** a_kernel_test_rev2.py --method census_Age_Hours --explainer bayesshap --kernel RBF --use_gpec 1 --use_labelnoise 0

**Example 4 (BayesSHAP uncertainty without using GPEC):** a_kernel_test_rev2.py --method census_Age_Hours --explainer bayesshap --use_gpec 0




## Regularization Test (Table 1)
These experiments train a black-box model with increasing amounts of regularization and average the uncertainty estimates for that model. The code is located in ./Tests/regularization_test/a_regularization_test.py. Running this file will save the results in ./Files/Results.


> **a_regularization_test.py options**
> 
> **--method** (specifies dataset)
> * census: Census dataset
> * cosinv: Cos model f(x,y) = 2*cos(10/x) - y
> * germancredit: German Credit Dataset 
> * onlineshoppers: Online Shoppers Dataset
> 
> **--explainer** (Explainer to use in conjunction with GPEC)
> * kernelshap: KernelSHAP explainer
> * cxplain: CXPlain
> * bayeslime: BayesLIME
> * bayesshap: BayesSHAP
> 
> **--kernel**
> * WEG: WEG Kernel
> * RBF: RBF Kernel
> 
> **--use_gpec**
> * 0: Use GPEC
> * 1: Use uncertainty estimate from the explainer only (only implemented for cxplain, bayeslime, bayesshap)
> 
> **--lam**
> lambda parameter for GPEC
>
> **--rho**
> rho parameter for GPEC
> 
> **--l2_reg**
> parameter for controlling the l2 regularization for training the neural network models
> 
> **--nn_softplus_beta**
> parameter for controlling the beta value (for softplus activation function) for training the neural network models
> 
> **--gamma**
> parameter for controlling the gamma value for training the XGBoost models

**Example 1 (GPEC-WEG with KernelSHAP):** a_regularization_test.py --method census --explainer kernelshap --kernel WEG --use_gpec 1

**Example 2 (GPEC-RBF with KernelSHAP):** a_regularization_test.py --method census --explainer kernelshap --kernel RBF --use_gpec 1

**Example 3 (BayesSHAP uncertainty without using GPEC):** a_regularization_test.py --method census --explainer bayesshap --use_gpec 0
