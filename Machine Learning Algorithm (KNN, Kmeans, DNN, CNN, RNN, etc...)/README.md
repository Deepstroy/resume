# ML/DL 필수 이론 및 구현코드

<br>

- ## Section 1 : Deep Learning 주요 논문 구현 및 실험 <br>
> **구현의 목적**
>1. 주요 딥러닝 논문을 읽고 코드로 옴길 수 있는 구현능력 함양 : Paper reproduce 능력 <br>
>2. 딥러닝 발전에 기여한 논문(Batch normalization 등)을 기존 모델에 적용후 실험결과 정리<br>

| Networks Models  | Code | Experiment results |
|---|:---:|:---:|
| __Transfer learning using Inception-V3__  | [Code Link](https://github.com/Deepstroy/GoogLeNet_v3_TransferLearning/blob/master/Transfer_Learning.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/inception3_trasfer.png?raw=true) |
| __ResNet reproduced by paper__  | [Code Link](https://github.com/Deepstroy/ResNet_based_on_paper/blob/master/Resnet_version1.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/Resnet_result2.png?raw=true) |
| __GoogLeNet using auxiliary classifier__  | [Code Link](https://github.com/Deepstroy/GoogLeNet-V1-with-auxiliary/blob/master/InceptionV1_without_L2.ipynb)  |![](https://github.com/Deepstroy/Inventory/blob/master/google_acc_loss.png?raw=true)|
| __VGG11 vs Using BatchNorm★__  | [Code Link](https://github.com/Deepstroy/Compare-vanilla-VGGNet-with-BatchNorm/blob/master/vanilla_vs_batchnorm/VGG_11_with_Batch_Normalization%20(1).ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/comparision.png?raw=true) |

<br>

- ## Section 2 : Tensorflow 없이 Numpy로 Deep Learning 구현하기 <br>

> **구현의 목적**
>1. ML/DL의 주요 이론을 수식을 기반으로 직접 코딩이 가능한 수준의 수학적 지식 습득 <br>
>2. Numpy와 python의 coding 능력 향상 <br>
>3. 즉, Neural Network를 numpy와 low level tensorflow api 기반의 python code로 작성 가능한 실력함양<br> 
    (**Tensorflow의 tf.layers. 와 같은 High-level API를 사용하지 않음**)


| Classic ML and DNN | Source Code | Contents |
|---|:---:|:---:|
| __K-NN__ | [Numpy](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/K-NN/KNN_numpy.ipynb) <br> [Tensorflow](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/K-NN/KNN_tensorflow.ipynb) |![](https://github.com/Deepstroy/Inventory/blob/master/KNN_point.png?raw=true)|
| **K-Means** | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/K-Means/K_means_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/k-means.png?raw=true) |
| **K-Median** | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/K-Median/K_Median_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/k-median.png?raw=true) |
| **Principle Component Analysis (PCA)** | [Numpy](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Principle%20Component%20Analysis/Principle_Component_Analysis_numpy.ipynb) <br> [Tensorflow](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Principle%20Component%20Analysis/Principle_Component_Analysis_tensorflow.ipynb) |![](https://github.com/Deepstroy/Inventory/blob/master/pca_.png?raw=true)|
| __Weight Initialization Methods in Neural Networks__ |  |  |  |
| └─ Xavier initialization with logistic function | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Weight%20initialization/Xavier%20initialization/Xavier_initialization_with_logistic_function_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/xavier_2.png?raw=true) |
| └─ He initialization with ReLU function | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Weight%20initialization/He%20initialization/He_initialization_with_relu.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/he_.png?raw=true) |
| __Optimal Parameter Search Methods__ |  |  |  |
| └─ Grid Search | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/Grid%20Search/Grid_Search_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/grid.png?raw=true) |
| └─ Gradient Descent Optimizer | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/Gradient%20Descent%20Optimizer/Gradient_descent_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/grad_des.png?raw=true) |
| └─ Momentum Optimizer | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/Momentum%20Optimizer/Momentum_Optimizer_tensorflow.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/Momentum_result.png?raw=true) |
| └─ Nesterov Momentum Optimizer ★ | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/Nesterov%20Momentum%20Optimizer/Momentum_with_NAG_tensosrflow.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/NAG_result.png?raw=true) |
| └─ RMSProp Optimizer | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/RMSProp%20Optimizer/RMSProp_tensorflow.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/rmsprop_result.png?raw=true) |
| └─ Adam Optimizer ★| [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Optimizer/Adam%20Optimizer/Adam_optimizer_numpy.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/adam_result.png?raw=true) |
| __Regression Methods__ |   |   | |
| └─ Linear Regression |  [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Regression/Linear%20Regression/Linear_Regression_numpy.ipynb) |  ![](https://github.com/Deepstroy/Inventory/blob/master/linear_chusesun.png?raw=true) | |
| └─ Multivariate Linear Regression | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Regression/Multivariate%20Regression/Multivariate_Linear_Regression_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/mulvar_reg.png?raw=true) | |
| __Batch Normalization__ |   |   | |
| └─ Batch Normalizatoin Step by Step |  [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Batch%20Normalization/batchnorm_stepbystep/Batchnorm_stepbystep_tensorflow.ipynb) |   | |
| └─ Batch Normalizatoin Implementation |  [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/Batch%20Normalization/batchnorm_implement/Batch_normalization_Implementation.ipynb)  | ![](https://github.com/Deepstroy/Inventory/blob/master/batnorm.png?raw=true)  | |
| __Deep Neural Networks in numpy ★__ |  |  | |
| └─ for Boston house prices | [Code Link](https://github.com/Deepstroy/resume/blob/master/Machine%20Learning%20Algorithm%20(KNN%2C%20Kmeans%2C%20DNN%2C%20CNN%2C%20RNN%2C%20etc...)/DNN_innumpy/DNN_regression/DNN_regression_numpy.ipynb) | ![](https://github.com/Deepstroy/Inventory/blob/master/DNN_reg.png?raw=true) |
<br>
