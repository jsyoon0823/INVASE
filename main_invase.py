"""Main function for INVASE.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
           "IINVASE: Instance-wise Variable Selection using Neural Networks," 
           International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@gmail.com

---------------------------------------------------

(1) Data generation
(2) Train INVASE or INVASE-
(3) Evaluate INVASE on ground truth feature importance and prediction
"""
       
# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_generation import generate_dataset
from invase import invase
from utils import feature_performance_metric, prediction_performance_metric   


def main (args):
  """Main function for INVASE.
  
  Args:
    - data_type: synthetic data type (syn1 to syn6)
    - train_no: the number of samples for training set
    - train_no: the number of samples for testing set
    - dim: the number of features
    - model_type: invase or invase_minus
    - model_parameters:
      - actor_h_dim: hidden state dimensions for actor
      - critic_h_dim: hidden state dimensions for critic
      - n_layer: the number of layers
      - batch_size: the number of samples in mini batch
      - iteration: the number of iterations
      - activation: activation function of models
      - learning_rate: learning rate of model training
      - lamda: hyper-parameter of INVASE
    
  Returns:
    - performance:
      - mean_tpr: mean value of true positive rate
      - std_tpr: standard deviation of true positive rate
      - mean_fdr: mean value of false discovery rate
      - std_fdr: standard deviation of false discovery rate
      - auc: area under roc curve
      - apr: average precision score
      - acc: accuracy
  """
  
  # Generate dataset
  x_train, y_train, g_train = generate_dataset (n = args.train_no, 
                                                dim = args.dim, 
                                                data_type = args.data_type, 
                                                seed = 0)
  
  x_test, y_test, g_test = generate_dataset (n = args.test_no,
                                             dim = args.dim, 
                                             data_type = args.data_type, 
                                             seed = 0)
  
  model_parameters = {'lamda': args.lamda,
                      'actor_h_dim': args.actor_h_dim, 
                      'critic_h_dim': args.critic_h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'iteration': args.iteration, 
                      'activation': args.activation, 
                      'learning_rate': args.learning_rate}
  
  # Train the model
  model = invase(x_train, y_train, args.model_type, model_parameters)
 
  model.train(x_train, y_train)    
    
  ## Evaluation
  # Compute importance score
  g_hat = model.importance_score(x_test)
  importance_score = 1.*(g_hat > 0.5)
    
  # Evaluate the performance of feature importance
  mean_tpr, std_tpr, mean_fdr, std_fdr = \
  feature_performance_metric(g_test, importance_score)
   
  # Print the performance of feature importance    
  print('TPR mean: ' + str(np.round(mean_tpr,1)) + '\%, ' + \
        'TPR std: ' + str(np.round(std_tpr,1)) + '\%, ')
  print('FDR mean: ' + str(np.round(mean_fdr,1)) + '\%, ' + \
        'FDR std: ' + str(np.round(std_fdr,1)) + '\%, ')
  
  # Predict labels
  y_hat = model.predict(x_test)
    
  # Evaluate the performance of feature importance
  auc, apr, acc = prediction_performance_metric(y_test, y_hat)
   
  # Print the performance of feature importance    
  print('AUC: ' + str(np.round(auc, 3)) + \
        ', APR: ' + str(np.round(apr, 3)) + \
        ', ACC: ' + str(np.round(acc, 3)))
  
  performance = {'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
                 'mean_fdr': mean_fdr, 'std_fdr': std_fdr,
                 'auc': auc, 'apr': apr, 'acc': acc}
  
  return performance
  
      
##
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_type',
      choices=['syn1','syn2','syn3','syn4','syn5','syn6'],
      default='syn1',
      type=str)
  parser.add_argument(
      '--train_no',
      help='the number of training data',
      default=10000,
      type=int)
  parser.add_argument(
      '--test_no',
      help='the number of testing data',
      default=10000,
      type=int)
  parser.add_argument(
      '--dim',
      help='the number of features',
      choices=[11, 100],
      default=11,
      type=int)
  parser.add_argument(
      '--lamda',
      help='inavse hyper-parameter lambda',
      default=0.1,
      type=float)
  parser.add_argument(
      '--actor_h_dim',
      help='hidden state dimensions for actor',
      default=100,
      type=int)
  parser.add_argument(
      '--critic_h_dim',
      help='hidden state dimensions for critic',
      default=200,
      type=int)
  parser.add_argument(
      '--n_layer',
      help='the number of layers',
      default=3,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini batch',
      default=1000,
      type=int)
  parser.add_argument(
      '--iteration',
      help='the number of iteration',
      default=10000,
      type=int)
  parser.add_argument(
      '--activation',
      help='activation function of the networks',
      choices=['selu','relu'],
      default='relu',
      type=str)
  parser.add_argument(
      '--learning_rate',
      help='learning rate of model training',
      default=0.0001,
      type=float)
  parser.add_argument(
      '--model_type',
      help='inavse or invase- (without baseline)',
      choices=['invase','invase_minus'],
      default='invase_minus',
      type=str)
  
  args_in = parser.parse_args() 
  
  # Call main function  
  performance = main(args_in)