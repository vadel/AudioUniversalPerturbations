from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join

import argparse
import os.path

import sys
import struct
import time

import random




if __name__ == '__main__':

    '''
    ##########################
    # GENERAL INPUT PARAMETERS
    ##########################
    # model_path:          Path to the target model (.pb frozen model)
    # train_dataset_path:  Path to the folder containing the training dataset, which will be used to craft the perturbation
    # valid_dataset_path:  Path to the folder containing the test dataset, which will be used to validate the perturbation with new data
    # save_path:           Path in which the results should be stored

    ##########################
    # CRAFTING PARAMETERS
    ##########################
    # delta:        Minimum accuracy threshold (NOTE: in the original algorithm is taken as fooling ratio threshold)
    # norm_bound:   Controls the l_p magnitude of the perturbation
    # p:            Norm to be used
    # num_classes:  Limits the number of classes to test against
    # max_iter_uni: (Optional) Maximum number of iterations for the universal perturbation crafting algorithm
    # max_iter_df:  maximum number of iterations for deepfool
    '''

    #path to the root directory of the project
    base_path = "/tmp/"


    #TARGET MODEL PATH
    ####################
    model_path = base_path + "models/my_frozen_graph_v2.pb"
    print("Path of the target model: \t " + model_path)


    #DATASET PATH
    ####################
    dataset_root = base_path + "datasets/speech_commands_v0.02/"
    print("Path of the training dataset:   \t " + dataset_root)
    valid_dataset_root = base_path + "datasets/speech_commands_test_set_v0.02/"
    print("Path of the validation dataset: \t " + valid_dataset_root)


    #LOAD AUDIO FILENAMES
    ######################
    input_files_root = base_path
    input_files_valid_root = base_path


    #PARAMETERS
    ov_param = 0.1  #Overshoot parameter for deepfool
    maxiter_uni = 5 #Max number of itertions for universal perturbation

    maxiter_df = 100 #Max number of iterations for deepfool
    miniter_df = 0   #Min number of iterations for deepfool

    max_acc_uni = 0.9 #Accuracy threshold

    p_norm = 2 #lp norm
    norm_bound = 0.1 #norm bound for the perturbation (under the specified lp norm)

    #Train files
    input_files_path = base_path + "training_files.npy"
    print("Input files path: " + input_files_path)

    #Validation files
    input_files_valid_path = base_path + "validation_files.npy"
    print("Validation files path: " + input_files_valid_path)

    results_folder = base_path
    print("Results folder: " + results_folder)

    cmd = "python3 Universal_perturbations_UAP_HC.py"
    cmd = cmd + " -i "  + input_files_path        
    cmd = cmd + " -vi " + input_files_valid_path 
    cmd = cmd + " -m "  + model_path                    
    cmd = cmd + " -d "  + dataset_root            
    cmd = cmd + " -vd " + valid_dataset_root     
    cmd = cmd + " -s "  + results_folder          
    cmd = cmd + " -o "  + str(ov_param)           
    cmd = cmd + " -maxiter_df "  + str(maxiter_df)  
    cmd = cmd + " -miniter_df "  + str(miniter_df)  
    cmd = cmd + " -maxiter_uni " + str(maxiter_uni)
    cmd = cmd + " -max_acc "     + str(max_acc_uni)  
    cmd = cmd + " -p "  + str(p_norm)             
    cmd = cmd + " -n "  + str(norm_bound)         

    #Launch command process
    print("Command: " + cmd)

    #Launch process
    os.system(cmd)

    print("Process succesfully launched")

