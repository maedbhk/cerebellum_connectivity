#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:48:47 2020
Used for preparing data for connectivity modelling and evaluation

@author: ladan
"""

# importing packages
import os # to handle path information
import pandas as pd
import numpy as np
import numpy.matlib # for repmat
#import scipy as sp
import data_integration as di
import essentials as es
import pickle # used for saving data (pip/pip3 install pickle-mixin)

# setting some default paths
baseDir         = '/Users/ladan/Documents/Project-Cerebellum/Cerebellum_Data'
behavDir        = 'data'
imagingDir      = 'imaging_data'
suitDir         = 'suit'
regDir          = 'RegionOfInterest'
connDir         = 'connModels' # setting a different directory than in sc1sc2_connectivity
suitToolDir     = '/Users/ladan/Documents/MATLAB/suit'
encodeDir       = 'encoding'

# sestting defaults for some variables
# returnSubjs = np.array([2,3,4,6,8,9,10,12,14,15,17,18,19,20,21,22,24,25,26,27,28,29,30,31])


# define functions
def get_data(sn, glm = 7, roi = 'grey_nan', which = 'cond', avg = 1):
    """
    get_data prepares per subject data for modelling. It uses cerebellar data for each subject, transformed into suit space!
    INPUTS
    sn    : a numpy array with ids of all the subjects that will be used 
    roi   : string specifying the name of the ROI you want the data for
    glm   : a number repesenting the glm Number you want to use: 7 (conditions), or 8 (tasks)
    which : will be used in creating the indicator matrix. can be 'task' or 'cond'
    avg   : flag indicating whether you want to average across runs within a session or not
    
    OUTPUTS
    DD_all: a dictionary containing data for both studies, all the subjects, both sessions
            The key "hierarchy" for this dictionary is: 
            DD_all{'sc1': {DD1}, 'sc2': {DD2}} The first level key is study id
            DD1{'s02': {DDD1}, 's03': {}, 's04': {}, ...} The second level key is subjects' ids
            DDD1{'sess1': {DDDD1}, 'sess2':{}} The third level key is session id
            DDDD1{'data': np.ndarray} The fourth level key is 'data'
            After four levels of nested dictionaries, you'll get to the numpy array with the data
            
    There are also dictionaries that will be saved during execution of get_data: B_alls
            B_alls has one level of dictionaries lower than DD_all:
            DD1{'s02': {DDD1}, 's03': {}, 's04': {}, ...} The second level key is subjects' ids
            DDD1{'sess1': {DDDD1}, 'sess2':{}} The third level key is session id
            DDDD1{'data': np.ndarray} The fourth level key is 'data'
            After three levels of nested dictionaries, you'll get to the numpy array with the data
    
    
    Some additional notes:
    The data type for the variable that stores all the data should be decided. 
    For now, I'm gonna go with nested dictionaries:
    It can go two ways:
    1. B_dict{experiment_num:{subject:{session:B_sess} with which we will end up with 
    a single dictionary for both experiments which can be saved. This is returned by the function
    2. B_dict{subject:{session:B_sess}} with which we will end up with one dictionary for each experiment, 
    each saved in a directory specific to that experiment. This is the way the matlab code was saving the variable.
    
    Prepared data will be saved using (pickle: pip/pip3 install pickle-mixin) as binary files with .dat format
    To load these saved variables later: 
        pickle.load(open(filename, "rb")) # "rb" stands for Reading Binary file

    """
    # looping over experiments
    DD_all = {} # will store all the data
    for e in np.arange(1, 3):
        print('........ preparing data for study %d roi %s'% (e, roi))

        # setting directories
        glmDir      = os.path.join(baseDir , 'sc%d'% e , 'GLM_firstlevel_%d'% glm)
        encodingDir = os.path.join(baseDir , 'sc%d'% e , encodeDir , 'glm%d'% glm)
        
        B_alls = {} # data for each experiment will be saved in a dictionary
        for s in sn:
            print('.... Doing subject %02d' % s)

            # getting the data for the roi
            print('.. Y_info')
            path2data = os.path.join(encodingDir , 's%02d'%s ,'Y_info_glm%d_%s.mat' %(glm, roi))
            data_dict = di.matImport(path2data, form = 'dict') # import data as a dictionary

            D = data_dict['data']

            # getting SPM_info for the tasks info
            print('.. SPM_info')
            path2spm = os.path.join(glmDir , 's%02d'%s , 'SPM_info.mat')
            T        = di.matImport(path2spm, form = 'dict') # import data as a dictionary
            
            ## for now it assumes that you want to average subtract the average across runs of a session
            # Now generate mean estimates per session
            # Note that in the new SPM_info Instruction is coded as cond =0
            ## This part here is done to have different numbers for different conditions.
            ## I may comment it out in future
            a = np.arange(0, 17).reshape(-1, 1)
            b = np.matlib.repmat(a, 16, 1)
            
            T['cond'][T['inst'] == 0] = T['cond'][T['inst'] == 0] + 16
            T['cond'][T['inst'] == 1] = b.reshape((b.shape[0], ))
            
            #T['cond'] = T['cond'] + 1   # This is only for generating the avergaging matrix
                
            B_sess = {}
            for se in [1, 2]:
                B_sess['sess%d'%se] = {}
                print('.Doing subject %02d, sess %d' % (s, se))
                if avg == 1:
                    X          = es.indicatorMatrix('identity', T[which]*(T['sess'] == se))
                    # np.linalg.pinv is pretty slow!
                    # also, we need to decide on an appropriate data type here
                    ## I'm gonna go with nested dictionaries here!
                    B_sess['sess%d'%se]['data']   = np.linalg.pinv(X) @ D[0:X.shape[0],:] # This calculates average beta across runs, skipping intercepts, for each session
                    
                    
                #elif avg == 0: # UNDER "CONSTRUCTION"
                
            B_alls['s%02d'%s] = B_sess
    
        DD_all['sc%d'%e] = B_alls
        #data_dict['avg'] =
        
        if avg == 1: # saving as mbeta (betas averaged across runs of each session)
            outname = 'mbeta_%s_all.dat'%roi
        elif avg == 0:
            outname = 'beta_%s_all.dat'%roi
        
        outfile = os.path.join(encodingDir, outname)
        pickle.dump(B_alls, open(outfile, "wb")) # "wb": Writing Binary file
    
    return DD_all

def get_wcon(experNum = [1, 2], glm = 7, roi = 'grey_nan', avg = 1):
    """
    get_wcon uses the data saved in get_data and the text file created to prepare the data for modelling
    
    INPUTS
    experNum : the default is set so that it has both experiments, but you can change it
    glm      : glm number
    roi      : the roi you want to use: grey_nan, tesselsWB162, ...
    avg      : use the averaged data across runs or not!
    
    OUTPUTS
    Y  : a single level dictionary with the key being the subject id and the value being a numpy array
         Y{'s02': [np1], 's03': [np2], ...}
         np1: a numpy array containing the concatenated data for both sessions of s02 ...
    Sf : The integrated dataframe that is created using the task_info text file and
    will be used for modelling and evaluation
    
    """
    
    # based on avg flag, determine which file has to be loaded in
    if avg == 1: # data averaged across runs
        inname = 'mbeta_%s_all.dat'%roi
    elif avg == 0: # data for individual runs
        inname = 'beta_%s_all.dat'%roi

    # Load the betas for all the conditions
    YD = {} # this dictionary will store all the data for all the experiments
    for e in experNum:
        print('Doing study %d' % e)
        # setting directories
        encodingDir = os.path.join(baseDir, 'sc%d'%e, encodeDir, 'glm%d'%glm)
        infile      = os.path.join(encodingDir, inname)

        # using pickle.load to load the .dat file saved in get_data
        ## YD will be B_alls from get_data
        YD['sc%d'%e]       = pickle.load(open(infile, "rb"))

    # making an integrated data structure
    ## load in the task info text file made for the connectivity project into a dataframe
    Tf = pd.read_csv(os.path.join(baseDir, 'sc1_sc2_taskConds_conn.txt'), sep = '\t')

    # create an empty dataframe
    Sf = pd.DataFrame(columns = Tf.columns)
    for e in experNum:
        Tfi = Tf.loc[Tf.StudyNum == e]
        T1  = Tfi.copy()
        T2  = Tfi.copy()

        T1['sess'] = (np.ones((len(Tfi.index), 1))).astype(int)
        T2['sess'] = (2*np.ones((len(Tfi.index), 1))).astype(int)

        Tfi = pd.concat([T1, T2], axis = 0, ignore_index = True)
        Sf  = pd.concat([Sf, Tfi], axis = 0, ignore_index = True)

    Y = {} # Y is defined to be a dictionary
    for s in list(YD['sc1'].keys()):
        Y[s] = {}
        tmp2 = {}
        for e in experNum:
            # for each study, it gets the two sessions and concatenates them
            tmp = {} # empty dictionary values of which will be concatenated and put into the new dictionary
            for se in [1, 2]:

                # subtract columnar means
                colMean    = np.nanmean(YD['sc%d'%e][s]['sess%d'%se]['data'], axis = 0)
                subColMean = YD['sc%d'%e][s]['sess%d'%se]['data'] - colMean
                tmp[se]    = subColMean

            # concatenate data
            fin = np.concatenate((tmp[1], tmp[2]), axis = 0)
            tmp2[e] = fin

        Y[s] = np.concatenate((tmp2[1], tmp2[2]), axis = 0)
    
    return (Y, Sf)