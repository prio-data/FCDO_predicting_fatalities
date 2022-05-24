# This script file collects a set of routines used in the ensembling and calibration of models

# Importing modules
# Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
# Views 3
from viewser.operations import fetch
from viewser import Queryset, Column
import views_runs
from views_partitioning import data_partitioner, legacy
from stepshift import views
import views_dataviz
from views_runs import storage, ModelMetadata
from views_runs.storage import store, retrieve, list, fetch_metadata
from views_forecasts.extensions import *




# Calibrate to conform to mean and standard deviation

# Calibration function
def mean_sd_calibrated(y_true_calpart,y_pred_calpart,y_pred_test,shift, threshold=0):
    ''' 
    Calibrates predictions. Expects the input columns from calibration partition to be without infinity values
    '''
    expand = y_true_calpart.loc[y_true_calpart>=threshold].std() / y_pred_calpart.loc[y_pred_calpart>=threshold].std()
    shiftsize = 0
    expanded = y_pred_test.copy()
    expanded.loc[expanded>=threshold] = expanded * expand
    if shift==True:
        shiftsize = y_true_calpart.loc[y_true_calpart>=threshold].mean() - y_pred_calpart.loc[y_pred_calpart>=threshold].mean()
        shifted = expanded
        shifted.loc[shifted>=threshold] = shifted + shiftsize
        calibrated_pred = shifted 
    if shift==False:
        calibrated_pred = expanded       
#    print('Calibration --', 'threshold:',threshold,'Shift:',shiftsize,'Expand:',expand)
    return (calibrated_pred,expand,shiftsize)

# GAM-based calibration function
def gam_calibrated(y_true_calpart,y_pred_calpart,y_pred_test,n_splines):
    ''' 
    Calibrates predictions using GAM.
    Expects the input columns from calibration partition to be without infinity values
    '''
    from pygam import LogisticGAM, LinearGAM, s, te
    gam = LinearGAM(s(0, constraints='monotonic_inc',n_splines = n_splines)).fit(y_pred_calpart, y_true_calpart)

    calibrated_pred = gam.predict(y_pred_test)
#    gam_summary = gam.summary()
    return (calibrated_pred, gam)


# Retrieving the predictions for calibration and test partitions
# The ModelList contains the predictions organized by model

def RetrieveStoredPredictions(ModelList,steps,EndOfHistory,run_id):
    ''' This function retrieves the predictions stored in ViEWS prediction storage for all models in the list passed to it.
    It assumes that each element in the list is a dictionary that contains a model['modelname'] '''
    i=0
    stepcols = ['ln_ged_sb_dep']
    for step in steps:
        stepcols.append('step_pred_' + str(step))
    level = 'cm'
    for model in ModelList:
        print(i, model['modelname'])
        stored_modelname_calib = level + '_' + model['modelname'] + '_calib'
        stored_modelname_test = level + '_' + model['modelname'] + '_test'
        stored_modelname_future = level +  '_' + model['modelname'] + '_f' + str(EndOfHistory)
        model['predictions_calib_df'] = pd.DataFrame.forecasts.read_store(stored_modelname_calib, run=run_id)[stepcols]
        model['predictions_calib_df'].replace([np.inf, -np.inf], 0, inplace=True)
        model['predictions_test_df'] = pd.DataFrame.forecasts.read_store(stored_modelname_test, run=run_id)[stepcols]
        model['predictions_test_df'].replace([np.inf, -np.inf], 0, inplace=True)
        i = i + 1
    print('All done')
    return(ModelList)


    # Calibration
def CalibratePredictions(ModelList, FutureStart, steps):
    '''
    Function that adds dfs with calibrated predictions to ModelList
    '''

    print('Calibrating models')

    stepcols = ['ln_ged_sb_dep']
    for step in steps:
        stepcols.append('step_pred_' + str(step))
        
    for model in ModelList:   
        model['calib_df_cal_expand'] = model['predictions_calib_df'].copy()
        model['test_df_cal_expand'] = model['predictions_test_df'].copy()
    #    if IncludeFuture:
    #        model['future_df_cal_expand'] = model['predictions_future_df'].copy()
        model['calib_df_calibrated'] = model['predictions_calib_df'].copy()
        model['test_df_calibrated'] = model['predictions_test_df'].copy()
    #    if IncludeFuture:
    #        model['future_df_calibrated'] = model['predictions_future_df'].copy()
        print(model['modelname'])
        model['calibration_gams'] = [] # Will hold calibration GAM objects, one for each step
        for col in stepcols[1:]:
            thisstep = int(col[10:])
            thismonth = FutureStart + thisstep
            calibration_gam_dict = {
                'Step': thisstep,
                'GAM': []
            }
            # Remove from model dfs rows where [col] has infinite values (due to the 2011 split of Sudan)
            df_calib = model['predictions_calib_df'][~np.isinf(model['predictions_calib_df'][col])].fillna(0)
            df_test = model['predictions_test_df'][~np.isinf(model['predictions_test_df'][col])].fillna(0)
    #        if IncludeFuture:
    #            df_future = model['predictions_future_df'][~np.isinf(model['predictions_future_df']['step_combined'])].fillna(0)

            (model['calib_df_cal_expand'][col],model['expanded'],model['shiftsize']) = mean_sd_calibrated(
                y_true_calpart = df_calib['ln_ged_sb_dep'], 
                y_pred_calpart = df_calib[col], 
                y_pred_test = df_calib[col], 
                shift=False, 
                threshold = 0
            )
            (model['test_df_cal_expand'][col],model['expanded'],model['shiftsize']) = mean_sd_calibrated(
                y_true_calpart = df_calib['ln_ged_sb_dep'], 
                y_pred_calpart = df_calib[col], 
                y_pred_test = df_test[col], 
                shift=False, 
                threshold = 0
            )
            if model['modelname'] == 'fat_hh20_Markov_glm' or model['modelname'] == 'fat_hh20_Markov_rf':
                model['calib_df_calibrated'][col] = model['calib_df_cal_expand'][col]
                model['test_df_calibrated'][col] = model['test_df_cal_expand'][col]
            else:
                (model['calib_df_calibrated'][col], calibration_gam_dict['calibration_GAM']) = gam_calibrated(
                        y_true_calpart = df_calib['ln_ged_sb_dep'], 
                        y_pred_calpart = df_calib[col], 
                        y_pred_test = df_calib[col], 
                        n_splines = 15
                )
                #print(model['calibration_gam'].summary())
                (model['test_df_calibrated'][col], gam) = gam_calibrated(
                        y_true_calpart = df_calib['ln_ged_sb_dep'], 
                        y_pred_calpart = df_calib[col], 
                        y_pred_test = df_test[col], 
                        n_splines = 15
                )
            model['calibration_gams'].append(calibration_gam_dict)
    return(ModelList)
