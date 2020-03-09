# Codebase for "INVASE: Instance-wise Variable Selection"

Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Paper: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
       "IINVASE: Instance-wise Variable Selection using Neural Networks," 
       International Conference on Learning Representations (ICLR), 2019.
       (https://openreview.net/forum?id=BJg_roAcK7)

This directory contains implementations of INVASE framework for 
the following applications.

-   Instance-wise feature selection
-   Prediction with instance-wise feature selection

To run the pipeline for training and evaluation on time-series 
prediction framwork, simply run python3 -m main_inavse.py.

Note that any model architecture can be used as the actor and critic models 
such as CNN. The condition for models is to have train and predict functions 
as its subfunctions.

## Stages of the time-series prediction:

-   Generate synthetic dataset (6 synthetic datasets)
-   Train INVASE or INVASE- (without baseline)
-   Evaluate INVASE for instance-wise feature selection
-   Evaluate INVASE for prediction

### Command inputs:

-   data_type: synthetic data type (syn1 to syn6)
-   train_no: the number of samples for training set
-   train_no: the number of samples for testing set
-   dim: the number of features

-   model_type: invase or invase_minus
-   model_parameters:
     - actor_h_dim: hidden state dimensions for actor
     - critic_h_dim: hidden state dimensions for critic
     - n_layer: the number of layers
     - batch_size: the number of samples in mini batch
     - iteration: the number of iterations
     - activation: activation function of models
     - learning_rate: learning rate of model training
     - lamda: hyper-parameter of INVASE

### Example command

```shell
$ python3 main_invase.py 
--data_type syn1 --train_no 10000 --test_no 10000 --dim 11
--model_type invase --actor_h_dim 100 --critic_h_dim 200
--n_layer 3 --batch_size 1000 --iteration 10000
--activation relu --learning_rate 0.0001 --lamda 0.1
```

### Outputs

-   Instance-wise feature selection performance:
    - Mean TPR
    - Std TPR
    - Mean FDR
    - Std FDR
-   Prediction performance:
    - AUC
    - APR
    - ACC