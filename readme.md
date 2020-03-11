# Universal Adversarial Examples for Speech Command Classification

This is the code corresponding to the paper 
*"Universal Adversarial Examples for Speech Command Classification"*, 
by Jon Vadillo and Roberto Santana.

## Introduction
In this work we address the generation of universal adversarial perturbations for Speech Command Classification. The attack approach used to create the universal perturbations (hereinafter referred to as UAP-HC algorithm) consists on a Hill-Climbing reformulation of the state-of-the-art method proposed by Moosavi-Dezfooli *et al*. in [3].

We encourage the reader to listen to some of the adversarial examples generated, accessible in [our web](https://vadel.github.io/UniversalAdversarialPerturbations.html).

## Dataset
We used the [Speech Command Dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) [1] (2nd version) as testbed to test our approach.  This dataset consists  on  aset of WAV audio files of 30 different spoken commands. The duration of all the files is fixed to 1 second, and the sample-rate is 16kHz in all the samples, so that each audio waveform is  composed  by 16000 values,  in  the  range [−2^15 , 2^15].   In the paper,  we used a subset of ten classes, those standard labels selected in previous publications: *”Yes”, ”No”, ”Up”, ”Down”, ”Left”, ”Right”, ”On”, ”Off”, ”Stop”,* and *”Go”*. In addition tothis  set,   consider ed two  special  classes:  *”Unknown”*  (a command different to the ones specified before) and *”Silence”* (no speech detected).


## Target models
We selected two different target models, based on the CNN structure proposed in [2] for small-footprint  keyword  spotting, as it is shown in the following figure:

![CNN structure of the target models](/images/CNN_structure.png)


### Target Model A
This model achieves an an 85.2% Top One  Accuracy on the [test set](http://download.tensorflow.org/data/speech_commands_streaming_test_v0.02.tar.gz) of the Speech Command Dataset (2nd version). The training procedure is based on the code provided in [link](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands). However, there are some modifications in order to ensure that the process is end-to-end differentiable, including the MFCC transformation.



### Target Model B
This is a pretrained model accesible from [this link](http://download.tensorflow.org/models/speech_commands_v0.02.zip). It obtained an 82.5% Top One  Accuracy on the [test set](http://download.tensorflow.org/data/speechcommandsstreamingtestv0.02.tar.gz) of the Speech Command Dataset (2nd version).

## User guide
The implementation has been implemented in Python 3.6.
### Generating universal perturbations
The  script "Universal_perturbations_multi.py" contains the implementation of the algorithm used for the generation of universal adversarial perturbations. The program takes as input the following parameters:

- *model_path:*  path to the target model. We assume that the model is a *frozen* TF classifier (.pb file). We asssume also an end-to-end differentiable model.  
- *labels_file_path:* the path to the file containing the label assigned to each class (i.e. 0=silence, 1=unknown, 2=yes...)
- *dataset_path*: the path to the root directory of the training dataset. This directory must contain a folder for each possible class, each of them containing the inputs corresponding to that class.
- *valid_dataset_path*: the path to the root directory of the validation set. This directory must contain a folder for each possible class, each of them containing the inputs corresponding to that class.
- *input_filenames:* text file containing the relative paths (starting from the root of the training dataset) to the files that will be used to construct the perturbation. The file must contain one path per row.
- *input_filenames:* text file containing the relative paths (starting from the root of the validation dataset) to the files that will be used to test the perturbation on unseen samples. The file must contain one path per row.
- *overshoot* and *maxiter_df*:  respectively, the overshoot parameter of Deepfool algorithm and maximum number of iterations allowed for constructing a single individual (local) perturbation.
- *maxiter_uni:* maximum number of iterations (epochs) allowed for UAP-HC algorithm.
- *max_acc:*  accuracy threshold. The algorithm will stop when the accuracy of the model is below the threshold (on the training set).
- *p_norm* and *norm_bound:* respectively, the L_p norm with which the perturbations is measured, and the maximum amount of distortion allowed (according to the specified L_p norm).
- *save_path*: the path in which the results will be saved. 

The program will return the following information:

- *v_full:* numpy matrix containing the universal perturbations achieved at each step. The dimensions of the matrix will be [N,L], being N the number of steps and L the length of the perturbation audio (i.e. 16.000 for the Speech Command Dataset).
- *fooling_rates_vec_full:* numpy array containing the fooling ratio of the perturbation on the training set, at each step.
- *fooling_rates_vec_full:* numpy array containing the fooling ratio of the perturbation on the validation set, at each step.

## References
[1] Peter Warden, “Speech commands: A dataset for limited-vocabulary speechrecognition,”arXiv preprint [arXiv:1804.03209](arXiv:1804.03209), 2018.

[2] T. N. Sainath and C. Parada, “Convolutional neural networks for small-footprint  keyword  spotting,”  in Sixteenth  Annual  Conference  of  theInternational Speech Communication Association, 2015.

[3] S. M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and P. Frossard, “Universal  adversarial  perturbations,”  in Proceedings  of  the  IEEE  Conference on Computer Vision and Pattern Recognition, 2017, pp. 1765–1773.


