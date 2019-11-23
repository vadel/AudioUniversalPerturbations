#!/usr/bin/env python3
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

import tensorflow as tf

import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

import scipy.io.wavfile as wav

import random



#Helper function to reset the TF graph and set a random seed
def reset_graph(seed=1996):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    config = tf.ConfigProto(device_count = {'GPU': 0})


#Helper function to load a frozen TF graph
def load_graph(frozen_graph_filename):
    # Load the file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #Import and return the graph 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


# Projection operator
# Project on the l_p ball centered at 0 and of radius 'norm_bound'
def projection_operator(v, norm_bound, p):
    if p == 2:
        v = v * min(1, norm_bound/np.linalg.norm(v.flatten()))
    elif p == np.inf:
        v = np.clip(v, -norm_bound, norm_bound)
    else:
         raise ValueError("Only p = 2 or p = np.inf allowed")
    return v



#Class implementing the UAP-HC attack algorithm proposed for the generation of audio universal adversarial perturbations
class Attack:
    def __init__(self, model_path, num_classes=12):
        """
        Initialization of the main attack algorithm.
        @param model_path:  Path to the target model. We assume that the model is a frozen TF classifier (.pb file). We asssume also an end-to-end differentiable model.
        @param num_classes: Number of classes considered in the problem.
        """
        print("\t+ Loading graph...")
        self.graph = load_graph(model_path)
        self.sess = tf.Session(graph=self.graph)
        print("\t- Graph loaded!")

        self.num_classes = num_classes
        
        print("\t+ Restoring tensors from the model...")
        self.input_layer_name   = "prefix/input_audio:0"    #INPUT TENSOR NAME
        self.logits_layer_name  = "prefix/add_3:0"          #LOGITS TENSOR NAME
        self.softmax_layer_name = "prefix/labels_softmax:0" #SOFTMAX TENSOR NAME

        self.input_tensor   = self.graph.get_tensor_by_name(self.input_layer_name)   #INPUT TENSOR
        self.logits_tensor  = self.graph.get_tensor_by_name(self.logits_layer_name)  #LOGITS TENSOR
        self.softmax_tensor = self.graph.get_tensor_by_name(self.softmax_layer_name) #SOFTMAX TENSOR
        
        self.y_flat = tf.reshape(self.logits_tensor, (-1,))

        with self.graph.as_default():
            self.inds = tf.placeholder(tf.int32, shape=(self.num_classes,))
            self.dydx = self.jacobian(self.y_flat, self.input_tensor, self.inds)
        print("\t- Tensors restored! Initialization sufccesfully completed!")



    def f(self, audio_sample): 
        """
        Classification function:
        @param audio_sample: input audio sample (numpy array). The values must be in the range [0,1].
        @return: array containing the activation logits of the model.
        """
        output = self.sess.run(self.logits_tensor,  feed_dict={self.input_tensor: audio_sample})
        return output
    

    def jacobian(self, y, x, inds):
        """
        Definition of the TF tensor used to compute Jacobians for each class.
        @param y: logits tensor of the model
        @param x: input  tensor of the model
        @param inds: classes for which the dydx derivatives are computed  
        @return: Jacobians for each class specified in inds
        """
        n = self.num_classes
        loop_vars = [
             tf.constant(0, tf.int32),
             tf.TensorArray(tf.float32, size=n),
        ]
        _, jacobian = tf.while_loop(
            lambda j,_: j < n,
            lambda j,result: (j+1, result.write(j, tf.gradients(y[inds[j]], x))), #inds[j]
            loop_vars)
        return jacobian.stack()
    

    def grads(self, audio_sample, indices): 
        """
        Helper function to compute the gradients of the logits w.r.t. an input audio. Based on the definition of the "jacobian" function.
        @param audio_sample: input audio sample
        @param indices: classes for which the gradients are computed
        @return gradients
        """
        return self.sess.run(self.dydx, feed_dict={self.input_tensor: audio_sample, self.inds: indices}).squeeze(axis=1)
    
    
    def deepfool(self, audio, label, num_classes, overshoot, max_iter=50, min_iter=0, verbose=False):
        """
        Deepfool algorithm used to create individual adversarial perturbations. More details in https://arxiv.org/pdf/1511.04599.pdf
        @param audio: audio sample, with a shape [1,16000]
        @param f:     classification function used to compute the logits of the model 
        @param num_classes: number of classes considered in the problem
        @param overshoot:   overshoot parameter of the Deepfool algorithm
        @param max_iter:    maximum number of iterations allowed for the Deepfool algorithm
        @param min_iter:    (optional) minimum number of iterations required for the Deepfool algorithm
        
        @return r  : minimal perturbation able to produce a misclassification
        @return itr: number of iterations needed to fool the model
        @return k_i: adversarial prediction
        @return adversarial examples
        """

        #Boundaries of the signal values
        min_ = -1.0
        max_ =  1.0

        #Initial logits of the model
        f_audio = np.array(self.f(audio)).flatten()

        #Residual labels: possible incorrect output labels 
        I = np.arange(num_classes)
        residual_labels = [l for l in I if l!=label]

        adv = audio #Adversarial example

        #Check the original algorithm for details: https://arxiv.org/pdf/1511.04599.pdf
        f_i = np.array(self.f(adv)).flatten()
        k_i = int(np.argmax(f_i))
        w = np.zeros(audio.shape)
        r = np.zeros(audio.shape)

        itr = 0 #iteration

        #Main loop
        while (k_i == label and itr < max_iter) or itr < min_iter:

            pert = np.inf
            gradients = np.asarray(self.grads(adv, I))

            #Select the closest class according to the greedy criterion of the Deepfool algorithm 
            for k in residual_labels:
                w_k = gradients[k, :, :] - gradients[label, :, :]
                f_k = f_i[k] - f_i[label]
                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # Update the local perturbation r_i and the total perturbation r
            r_i =  pert * w / np.linalg.norm(w)
            r = r + r_i

            #Generate the current adversarial example
            adv = audio + (1+overshoot)*r
            adv = np.clip(adv, min_, max_) #clip the values in order to not exceed the boundaries

            #As the WAV files are stored as 16-bit integer files, unscale and scale again the perturbation in order to
            #ensure that it does not affect the effectiveness of the attack. 
            adv = np.array(np.clip(np.round(np.clip(adv,-1.0,1.0)*(1<<15)), -2**15, 2**15-1),dtype=np.int16)
            adv = adv/(1<<15)

            # Compute the new label
            f_i = np.array(self.f(adv)).flatten()
            k_i = int(np.argmax(f_i))

            itr += 1

        r = (1+overshoot)*r

        return adv, r, itr, k_i



    def universal_perturbation(self, 
                                dataset, 
                                validation_set,
                                alpha=0.2, 
                                max_iter_uni = np.inf, 
                                norm_bound=10, p=np.inf, 
                                num_classes=12, 
                                overshoot=0.02, 
                                max_iter_df=10, 
                                min_iter_df=0,
                                recompute_fooled_samples=False):
        """
        @param dataset: training   set used to create the universal perturbations. Shape: [NumAudios , AudioLength]
        @param dataset: validation set used to compute the effectiveness of the universal perturbations against unseen inputs. Shape: [NumAudios , AudioLength]
        @param alpha:   fooling rate threshold. The algorithm will stop when 1-alpha is surpassed 
        @param max_iter_uni: optional termination criterion (maximum number of iteration, default = np.inf)
        @param norm_bound:   maximum perturbation amount according to the specified l_p norm
        @param p:  l_p norm to be used (only p=2 or p=inf supported)
        @param num_classes:  number of classes considered on the problem
        @param overshoot:    overshoot parameter of Deepfool algorithm
        @param max_iter_df:  maximum number of iterations allowed for Deepfool
        @param min_iter_df:  optional: minimum number of iterations required for Deepfool 
        @param recompute_fooled_samples: If True,  updates the universal perturbation v considering also samples alreayd fooled by v.
                                         If False, updates considering only the samples not fooled by v.
        
        @return: the universal perturbation v. Shape: [NumSteps, AudioLength]
        @return: fooling rates of the perturbation in both training and validation sets, for each step. Shape: [NumSteps, NumAudios]
        @return: distortion produced by the perturbation, computed for both training and validation sets, according to the metric employed by convention: 
                    dB_{max,x}(v)=dB_{max}(v)-dB_{max}(v)  , where dB_{max}(x)=max_i 20*log_{10}(|x_i|).  Shape: [NumSteps, NumAudios]
        """

        
        fooling_rate = 0.0
        prev_iter_fooling_rate = 0.0
        num_samples =  np.shape(dataset)[0] # The samples should be stacked ALONG FIRST DIMENSION
        num_samples_valid =  np.shape(validation_set)[0] # The samples should be stacked ALONG FIRST DIMENSION

        v = np.zeros((16000)) #UNIVERSAL PERTURBATION
        v_mat = np.zeros([max_iter_uni, 16000]) #Matrix to store the perturbations at each iteration
        v_mat_full = np.zeros([max_iter_uni*num_samples, 16000]) #Matrix to store the perturbations at each iteration

        #Vector containing the fooling rates for the training dataset 
        fooling_rates_vec = np.zeros([max_iter_uni]) #one per "epoch"
        fooling_rates_vec_full = np.zeros([max_iter_uni*num_samples]) #one per iteration (per modification)

        #Vectors containing the fooling rates for the validation dataset
        fooling_rates_vec_valid = np.zeros([max_iter_uni]) #one per "epoch"
        fooling_rates_vec_full_valid = np.zeros([max_iter_uni*num_samples]) #one per iteration (per modification)

        #Matrix to store, for each sample, the dB difference at each iteration
        db_diff_max_vec  = np.zeros([num_samples, max_iter_uni])             #Training set
        db_diff_max_vec_valid  = np.zeros([num_samples_valid, max_iter_uni]) #Validation set

        #Matrix to store the order in which the training dataset is processed at each "epoch"
        order_idx_matrix = np.zeros([max_iter_uni, num_samples])

        dataset_ = np.copy(dataset) #Copy of the original dataset

        #Fooling ratio of the perturbation in both training and validation sets
        global_fooling_rate = 0.0
        global_fooling_rate_valid = 0.0

        #Iteration number of the main algorithm
        itr = 0

        #Compute the estimated labels for the original training dataset
        print("Predicting original training dataset")
        est_labels_orig = np.zeros((num_samples), dtype=int)
        for ii in range(0, num_samples):
            ii_original = dataset_[ii, :].reshape(1,16000)
            logits_orig = self.f(ii_original)
            est_labels_orig[ii] = np.argmax(logits_orig, axis=1).flatten()

        #Compute also the labels for the original validation dataset
        print("Predicting original validation dataset")
        est_labels_orig_valid = np.zeros((num_samples_valid), dtype=int)
        for ii in range(0, num_samples_valid):
            ii_original_valid = validation_set[ii, :].reshape(1,16000)
            logits_orig_valid = self.f(ii_original_valid)
            est_labels_orig_valid[ii] = np.argmax(logits_orig_valid, axis=1).flatten()


        while (fooling_rate < 1-alpha) and (itr < max_iter_uni):
            # Shuffle the dataset at the beginning of each "epoch"
            # We will save the order, in case we want to track the positions
            order_idx = np.random.permutation(num_samples)
            dataset = np.copy(dataset_[order_idx,:])
            label_idxs = np.copy(est_labels_orig[order_idx])
            order_idx_matrix[itr,:] = order_idx.flatten()
            #np.random.shuffle(dataset)

            print ('Starting pass number ', itr)

            #Iterate through the training set            
            for k in range(0, num_samples):
                cur_sample = dataset[k, :]
                cur_sample = cur_sample.reshape(1,16000)            
                label = label_idxs[k]
                print("Label: " + str(label))

                if int(np.argmax(np.array(self.f(cur_sample)).flatten())) == int(np.argmax(np.array(self.f(cur_sample+v)).flatten())):
                    print('>> k = ', k, ', pass #', itr)
                    print("Global fooling rate:", str(global_fooling_rate))
                    
                    # Compute adversarial perturbation
                    adv, dr, iter, adv_pred = self.deepfool(cur_sample + v, 
                                                    label, 
                                                    num_classes=num_classes, 
                                                    overshoot=overshoot, 
                                                    max_iter=max_iter_df)
                    
                    
                    #Check the fooling ratio after the evaluation
                    dataset_perturbed = dataset_ + projection_operator(v+dr, norm_bound, p)
                    est_labels_pert = np.zeros((num_samples))
                    FOOLING_RATIO_NOT_IMPROVABLE = False
                    for ii in range(0, num_samples):
                        ii_perturbed = dataset_perturbed[ii, :].reshape(1,16000)
                        logits_pert = self.f(ii_perturbed)
                        est_labels_pert[ii] = np.argmax(logits_pert, axis=1).flatten()

                        #Tip for computational efficiency:
                        #If we already know that we can not improve the global fooling rate, stop computing the fooling ratio
                        pending_samples = num_samples - ii -1
                        max_reachable_fooling_rate =  float( (np.sum(est_labels_pert[0:(ii+1)]!=est_labels_orig[0:(ii+1)])+pending_samples)/float(num_samples) )
                        if max_reachable_fooling_rate < global_fooling_rate:
                            FOOLING_RATIO_NOT_IMPROVABLE = True
                            print("%d) F.R. NOT IMPROVABLE: MAX. %d+%d SAMPLES CORRECT OUT OF %d"%(ii, int(np.sum(est_labels_pert[0:(ii+1)]!=est_labels_orig[0:(ii+1)])) , pending_samples , num_samples))
                            print("GLOBAL F.R: %s (IN SAMPLES: %d)"%( str(global_fooling_rate) , int(global_fooling_rate*num_samples) ))
                            break

                    #If the fooling ratio can not be improved, set a small value for the current fooling ratio, just to reject updating the perturbation
                    if FOOLING_RATIO_NOT_IMPROVABLE:
                        local_fooling_rate = 0.0 #
                    else:
                        local_fooling_rate = float(np.sum(est_labels_pert[0:(ii+1)] != est_labels_orig[0:(ii+1)]) / float(num_samples))
                    
                    # Make sure it converged...
                    if iter < max_iter_df-1 and local_fooling_rate>global_fooling_rate:
                        v = v + dr
                        v = projection_operator(v, norm_bound, p) # Project on l_p ball

                        #UPDATE THE GLOBAL FOOLING RATE
                        global_fooling_rate = local_fooling_rate

                    else:
                        print("Not converged or global fooling rate not improved")

                elif recompute_fooled_samples:
                    print('>> k = ', k, ', pass #', itr)
                    # Compute adversarial perturbation
                    adv, dr,iter,adv_pred = self.deepfool(cur_sample + v, 
                                                    label, 
                                                    num_classes=num_classes, 
                                                    overshoot=overshoot,
                                                    max_iter=max_iter_df,
                                                    min_iter=min_iter_df)
                    # Make sure we fool the model
                    if adv_pred != label:
                        v = v + dr
                        v = projection_operator(v, norm_bound, p)


                #Save the obtained fooling rates. 
                #Note that if the local perturbation hasn't been added to 'v', this values won't change neither
                fooling_rates_vec_full[itr*num_samples + k] = global_fooling_rate

                #Save the perturbation
                v_mat_full[itr*num_samples + k, :] = v.flatten()
                        
            # Perturb the dataset and compute the fooling rate
            dataset_perturbed = dataset_ + v
            est_labels_pert = np.zeros((num_samples))
            for ii in range(0, num_samples):
                ii_perturbed = dataset_perturbed[ii, :].reshape(1,16000)
                logits_pert = self.f(ii_perturbed)
                est_labels_pert[ii] = np.argmax(logits_pert, axis=1).flatten()
            
            # Compute the fooling rate
            fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_samples))
            print('FOOLING RATE = ', fooling_rate)
            print(est_labels_pert)
            fooling_rates_vec[itr] = fooling_rate


            #Decibels difference of the perturbation and the audio
            db_diff_max = 20*np.log10(np.max(np.abs(v))) - 20*np.log10(np.max(np.abs(dataset_), axis=1))
            db_diff_max_vec[:,itr] = db_diff_max.flatten()
            #Also for the validation set
            db_diff_max_valid = 20*np.log10(np.max(np.abs(v))) - 20*np.log10(np.max(np.abs(validation_set), axis=1))
            db_diff_max_vec_valid[:,itr] = db_diff_max_valid.flatten()
            
            #Save the perturbation as wav file
            v_unscaled = np.array(np.clip(np.round(np.clip(v,-1.0,1.0)*(1<<15)), -2**15, 2**15-1),dtype=np.int16)
            v_mat[itr,:] = v

            #If the fooling ratio has not changed in the whole epoch, finish the algorithm
            if np.abs(prev_iter_fooling_rate-fooling_rate)<0.000000001:
                print("Finishing at iteration " + str(itr) + " --> FR has not changed in the whole epoch")
                break
            
            #Store the current fooling rate, to use it in the next epoch
            prev_iter_fooling_rate = fooling_rate

            itr = itr + 1
                    
        return  v_mat, \
                fooling_rates_vec_full, fooling_rates_vec_full_valid, \
                db_diff_max_vec , db_diff_max_vec_valid






if __name__ == '__main__':


    from argparse import ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--input_filenames",      dest="input_files_path",       help="Full path to the numpy file (.npy) containing 'training' audio filenames (array)")
    parser.add_argument("-vi", "--valid_input_filenames",dest="valid_input_files_path", help="Full path to the numpy file (.npy) containing validation audio filenames (array)")

    parser.add_argument("-m",  "--model_path",           dest="model_path",          help="Path to the frozen TF model (.pb)")
    parser.add_argument("-d",  "--dataset_path",         dest="dataset_path",        help="Path to the 'training' dataset folder")
    parser.add_argument("-vd", "--valid_dataset_path",   dest="valid_dataset_path",  help="Path to the validation dataset folder")
    parser.add_argument("-s",  "--save_path",            dest="save_path",           help="Path to the directory in which we want to save the results")

    parser.add_argument("-o",          "--overshoot",    dest="overshoot_df",        help="Overshoot parameter of the Deepfool algorithm", type=float)
    parser.add_argument("-maxiter_df", "--maxiter_df",   dest="max_iter_df",         help="Maximum number of iterations for Deepfool", type=int)
    parser.add_argument("-miniter_df", "--miniter_df",   dest="min_iter_df",         help="Minimum number of iterations for Deepfool", type=int)

    parser.add_argument("-maxiter_uni", "--maxiter_uni", dest="max_iter_uni",        help="Maximum number of iterations for universal", type=int)
    parser.add_argument("-max_acc",     "--max_acc",     dest="max_acc_uni",         help="Desired accuracy in [0,1] range (stopping criterion)", type=float)

    parser.add_argument("-p",       "--p_norm",          dest="p_norm",              help="Norm type  to be used to bound universal perturbation", type=int)
    parser.add_argument("-n",       "--norm_bound",      dest="norm_bound",          help="Norm value to be used to bound universal perturbation", type=float)
    
    parser.add_argument("--centre", action='store_true', default=False,              help="Centres the audios used to craft perturbations")
    parser.add_argument("--recomp", action='store_true', default=False,              help="Recompute already fooled samples")
    args = parser.parse_args()



    #TARGET MODEL PATH
    ####################
    model_path = args.model_path
    print("Path of the target model: \t " + model_path)


    #DATASET PATH
    ####################
    dataset_path = args.dataset_path
    print("Path of the TRAINING   dataset: \t " + dataset_path)
    valid_dataset_path = args.valid_dataset_path
    print("Path of the VALIDATION dataset: \t " + valid_dataset_path)


    #LOAD AUDIO FILENAMES
    ######################
    print("Loading audio filenames")
    input_files_path = args.input_files_path
    print("-- Loading TRAINING   filenames from: \t " + input_files_path)
    train_audios = np.load(input_files_path) #Load audios to execute


    valid_files_path = args.valid_input_files_path
    print("-- Loading VALIDATION filenames from: \t " + valid_files_path)
    valid_audios = np.load(valid_files_path) #Load audios to execute
    print("Audio filenames sufccesfully loaded!")


    #GENERATE SETUP CLASS
    print("Generating setup clas...")
    setup = Attack(model_path=model_path)
    print("Class generated!")


    print("Loading audio files...")
    # LOAD TRAINING FILES
    N_samples = len(train_audios)
    dataset = np.zeros((N_samples, 16000))
    cont = 0
    for audio_path in train_audios:
        #Generate the adversarial example
        fs, audio = wav.read(dataset_path + audio_path)
        scale_factor = 1/(1<<15)
        audio_scaled = audio*scale_factor
        audio_scaled = audio_scaled.reshape(1,16000)
        
        dataset[cont,:] = audio_scaled
        
        cont = cont + 1

    print("-- TRAINING   files succesfully loaded (%d files loaded)"%(cont))


    # LOAD ALSO VALIDATION SET
    N_samples_valid = len(valid_audios)
    validation_set = np.zeros((N_samples_valid, 16000))
    cont_valid = 0
    for audio_path in valid_audios:
        #Generate the adversarial example
        fs, audio = wav.read(valid_dataset_path + audio_path)
        scale_factor = 1/(1<<15)
        audio_scaled = audio*scale_factor
        audio_scaled = audio_scaled.reshape(1,16000)
        
        validation_set[cont_valid,:] = audio_scaled
        
        cont_valid = cont_valid + 1
    
    print("-- VALIDATION files succesfully loaded (%d files loaded)"%(cont_valid))
    
    print("Audio files succesfully loaded!")


    print("Launching process")
    overshoot_df = args.overshoot_df
    max_iter_df = args.max_iter_df
    min_iter_df = args.min_iter_df

    max_acc_uni = args.max_acc_uni
    max_iter_uni = args.max_iter_uni

    p_norm = args.p_norm #norm type
    
    norm_bound  = args.norm_bound #norm bound (value)

    recompute_fooled_samples = args.recomp

    v_mat, \
    fooling_rates_vec_full, fooling_rates_vec_full_valid, \
    db_diff_max_vec, \
    db_diff_max_vec_valid = setup.universal_perturbation(dataset=dataset,
                                                        validation_set=validation_set,
                                                        alpha=1-max_acc_uni, 
                                                        norm_bound=norm_bound, 
                                                        p=p_norm, 
                                                        overshoot=overshoot_df, 
                                                        num_classes=12,
                                                        max_iter_uni = max_iter_uni,
                                                        max_iter_df = max_iter_df, 
                                                        min_iter_df = min_iter_df,
                                                        recompute_fooled_samples=recompute_fooled_samples)

    #SAVE THE RESULTS
    save_path = args.save_path #path to the directory in which the results will be saved

    #Save the perturbations
    filename = "perts.npy"
    np.save(save_path + filename, v_mat)

    #Save the fooling ratios at each step (TRAINING SET)
    filename = "fooling_rates_vec_full_train.npy"
    np.save(save_path + filename, fooling_rates_vec_full)

    #Save the fooling ratios at each step (VALIDATION SET)
    filename = "fooling_rates_vec_full_valid.npy"
    np.save(save_path + filename, fooling_rates_vec_full_valid)

    #Save the computed distortions
    filename = "db_diff_max_vec.npy"
    np.save(save_path + filename, db_diff_max_vec)
    filename = "db_diff_max_vec_valid.npy"
    np.save(save_path + filename, db_diff_max_vec_valid)

    print("Done!")

