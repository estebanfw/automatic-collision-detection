import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import numpy as np
import os


#######CONVERT KELVIN DATASET TO CDM STANDARD VARIABLES
def convertKelvinDatasetToCDMFormat(mydf):
    """
    Input of this function must be Kelvin's Competition as Dataframe
    """
    cdm = pd.DataFrame()
    cdm["event_id"]=mydf["event_id"]
    cdm["time_to_tca"]=mydf['time_to_tca']
    cdm["TCA_timeformat"]=dt.datetime.now()
    cdm["TCA"]=dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

    cdm['CREATION_DATE_timeformat']=cdm["TCA_timeformat"]-pd.to_timedelta(cdm['time_to_tca'], unit='d')
    cdm['CREATION_DATE']=cdm['CREATION_DATE_timeformat'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    cdm['MISS_DISTANCE'] = mydf['miss_distance']
    cdm['RELATIVE_SPEED'] = mydf['relative_speed']
    #OBJECT 1 State vector
    cdm['RELATIVE_POSITION_R'] = mydf['relative_position_r']
    cdm['RELATIVE_POSITION_T'] = mydf['relative_position_t']
    cdm['RELATIVE_POSITION_N'] = mydf['relative_position_n']
    cdm['RELATIVE_VELOCITY_R'] = mydf['relative_velocity_r']
    cdm['RELATIVE_VELOCITY_T'] = mydf['relative_velocity_t']
    cdm['RELATIVE_VELOCITY_N'] = mydf['relative_velocity_n']
    #COLLISSION PROBABILITY
    cdm['COLLISSION_PROBABILITY']=mydf['risk'].apply(lambda x: 10**x)

    ########################################### OBJECT 1 ###########################################
    #OBJECT 1 Covariance matrix
    cdm['OBJECT1_CR_R'] = mydf['t_sigma_r']**2.
    cdm['OBJECT1_CT_R'] = mydf['t_ct_r'] * mydf['t_sigma_r'] * mydf['t_sigma_t']
    cdm['OBJECT1_CT_T'] = mydf['t_sigma_t']**2.
    cdm['OBJECT1_CN_R'] = mydf['t_cn_r'] * mydf['t_sigma_n'] * mydf['t_sigma_r']
    cdm['OBJECT1_CN_T'] = mydf['t_cn_t'] * mydf['t_sigma_n'] * mydf['t_sigma_t']
    cdm['OBJECT1_CN_N'] = mydf['t_sigma_n']**2.
    cdm['OBJECT1_CRDOT_R'] = mydf['t_crdot_r'] * mydf['t_sigma_rdot'] * mydf['t_sigma_r']
    cdm['OBJECT1_CRDOT_T'] = mydf['t_crdot_t'] * mydf['t_sigma_rdot'] * mydf['t_sigma_t']
    cdm['OBJECT1_CRDOT_N'] = mydf['t_crdot_n'] * mydf['t_sigma_rdot'] * mydf['t_sigma_n']
    cdm['OBJECT1_CRDOT_RDOT'] = mydf['t_sigma_rdot']**2.
    cdm['OBJECT1_CTDOT_R'] = mydf['t_ctdot_r'] * mydf['t_sigma_tdot'] * mydf['t_sigma_r']
    cdm['OBJECT1_CTDOT_T'] = mydf['t_ctdot_t'] * mydf['t_sigma_tdot'] * mydf['t_sigma_t']
    cdm['OBJECT1_CTDOT_N'] = mydf['t_ctdot_n'] * mydf['t_sigma_tdot'] * mydf['t_sigma_n']
    cdm['OBJECT1_CTDOT_RDOT'] = mydf['t_ctdot_rdot'] * mydf['t_sigma_tdot'] * mydf['t_sigma_rdot']
    cdm['OBJECT1_CTDOT_TDOT'] = mydf['t_sigma_tdot']**2.
    cdm['OBJECT1_CNDOT_R'] = mydf['t_cndot_r'] * mydf['t_sigma_ndot'] * mydf['t_sigma_r']
    cdm['OBJECT1_CNDOT_T'] = mydf['t_cndot_t'] * mydf['t_sigma_ndot'] * mydf['t_sigma_t']
    cdm['OBJECT1_CNDOT_N'] = mydf['t_cndot_n'] * mydf['t_sigma_ndot'] * mydf['t_sigma_n']
    cdm['OBJECT1_CNDOT_RDOT'] = mydf['t_cndot_rdot'] * mydf['t_sigma_ndot'] * mydf['t_sigma_rdot']
    cdm['OBJECT1_CNDOT_TDOT'] = mydf['t_cndot_tdot'] * mydf['t_sigma_ndot'] * mydf['t_sigma_tdot']
    cdm['OBJECT1_CNDOT_NDOT'] = mydf['t_sigma_ndot']**2.

    cdm['OBJECT1_RECOMMENDED_OD_SPAN'] = mydf['t_recommended_od_span']
    cdm['OBJECT1_ACTUAL_OD_SPAN'] = mydf['t_actual_od_span']
    cdm['OBJECT1_OBS_AVAILABLE'] = mydf['t_obs_available']
    cdm['OBJECT1_OBS_USED'] = mydf['t_obs_used']
    cdm['OBJECT1_RESIDUALS_ACCEPTED'] = mydf['t_residuals_accepted']
    cdm['OBJECT1_WEIGHTED_RMS'] = mydf['t_weighted_rms']
    cdm['OBJECT1_SEDR'] = mydf['t_sedr']
    cdm['OBJECT1_TIME_LASTOB_START_timeformat']=cdm["CREATION_DATE_timeformat"]-pd.to_timedelta(mydf['t_time_lastob_start'], unit='d')
    cdm['OBJECT1_TIME_LASTOB_START']=cdm['OBJECT1_TIME_LASTOB_START_timeformat'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    cdm['OBJECT1_TIME_LASTOB_END_timeformat']=cdm["CREATION_DATE_timeformat"]-pd.to_timedelta(mydf['t_time_lastob_end'], unit='d')
    cdm['OBJECT1_TIME_LASTOB_END']=cdm['OBJECT1_TIME_LASTOB_START_timeformat'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

    #OBJECT1 BALISTIC COEFFICIENTES
    cdm['OBJECT1_CD_AREA_OVER_MASS']=mydf['t_cd_area_over_mass']
    cdm['OBJECT1_CR_AREA_OVER_MASS']=mydf['t_cr_area_over_mass']

    #OBJECT1 ORBITAL PARAMETERS
    cdm['OBJECT1_APOGEE_ALTITUDE']=mydf['t_h_apo']
    cdm['OBJECT1_PERIGEE_ALTITUDE']=mydf['t_h_per']
    cdm['OBJECT1_INCLINATION']=mydf['t_j2k_inc']

    ########################################### OBJECT 2 ###########################################
    cdm['OBJECT2_CR_R'] = mydf['c_sigma_r']**2.
    cdm['OBJECT2_CT_R'] = mydf['c_ct_r'] * mydf['c_sigma_r'] * mydf['c_sigma_t']
    cdm['OBJECT2_CT_T'] = mydf['c_sigma_t']**2.
    cdm['OBJECT2_CN_R'] = mydf['c_cn_r'] * mydf['c_sigma_n'] * mydf['c_sigma_r']
    cdm['OBJECT2_CN_T'] = mydf['c_cn_t'] * mydf['c_sigma_n'] * mydf['c_sigma_t']
    cdm['OBJECT2_CN_N'] = mydf['c_sigma_n']**2.
    cdm['OBJECT2_CRDOT_R'] = mydf['c_crdot_r'] * mydf['c_sigma_rdot'] * mydf['c_sigma_r']
    cdm['OBJECT2_CRDOT_T'] = mydf['c_crdot_t'] * mydf['c_sigma_rdot'] * mydf['c_sigma_t']
    cdm['OBJECT2_CRDOT_N'] = mydf['c_crdot_n'] * mydf['c_sigma_rdot'] * mydf['c_sigma_n']
    cdm['OBJECT2_CRDOT_RDOT'] = mydf['c_sigma_rdot']**2.
    cdm['OBJECT2_CTDOT_R'] = mydf['c_ctdot_r'] * mydf['c_sigma_tdot'] * mydf['c_sigma_r']
    cdm['OBJECT2_CTDOT_T'] = mydf['c_ctdot_t'] * mydf['c_sigma_tdot'] * mydf['c_sigma_t']
    cdm['OBJECT2_CTDOT_N'] = mydf['c_ctdot_n'] * mydf['c_sigma_tdot'] * mydf['c_sigma_n']
    cdm['OBJECT2_CTDOT_RDOT'] = mydf['c_ctdot_rdot'] * mydf['c_sigma_tdot'] * mydf['c_sigma_rdot']
    cdm['OBJECT2_CTDOT_TDOT'] = mydf['c_sigma_tdot']**2.
    cdm['OBJECT2_CNDOT_R'] = mydf['c_cndot_r'] * mydf['c_sigma_ndot'] * mydf['c_sigma_r']
    cdm['OBJECT2_CNDOT_T'] = mydf['c_cndot_t'] * mydf['c_sigma_ndot'] * mydf['c_sigma_t']
    cdm['OBJECT2_CNDOT_N'] = mydf['c_cndot_n'] * mydf['c_sigma_ndot'] * mydf['c_sigma_n']
    cdm['OBJECT2_CNDOT_RDOT'] = mydf['c_cndot_rdot'] * mydf['c_sigma_ndot'] * mydf['c_sigma_rdot']
    cdm['OBJECT2_CNDOT_TDOT'] = mydf['c_cndot_tdot'] * mydf['c_sigma_ndot'] * mydf['c_sigma_tdot']
    cdm['OBJECT2_CNDOT_NDOT'] = mydf['c_sigma_ndot']**2.

    cdm['OBJECT2_OBJECT_TYPE'] = mydf['c_object_type']
    cdm['OBJECT2_RECOMMENDED_OD_SPAN'] = mydf['c_recommended_od_span']
    cdm['OBJECT2_ACTUAL_OD_SPAN'] = mydf['c_actual_od_span']
    cdm['OBJECT2_OBS_AVAILABLE'] = mydf['c_obs_available']
    cdm['OBJECT2_OBS_USED'] = mydf['c_obs_used']
    cdm['OBJECT2_RESIDUALS_ACCEPTED'] = mydf['c_residuals_accepted']
    cdm['OBJECT2_WEIGHTED_RMS'] = mydf['c_weighted_rms']
    cdm['OBJECT2_SEDR'] = mydf['c_sedr']
    cdm['OBJECT2_TIME_LASTOB_START_timeformat']=cdm["CREATION_DATE_timeformat"]-pd.to_timedelta(mydf['c_time_lastob_start'], unit='d')
    cdm['OBJECT2_TIME_LASTOB_START']=cdm['OBJECT2_TIME_LASTOB_START_timeformat'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    cdm['OBJECT2_TIME_LASTOB_END_timeformat']=cdm["CREATION_DATE_timeformat"]-pd.to_timedelta(mydf['c_time_lastob_end'], unit='d')
    cdm['OBJECT2_TIME_LASTOB_END']=cdm['OBJECT2_TIME_LASTOB_START_timeformat'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

    #OBJECT1 BALISTIC COEFFICIENTES
    cdm['OBJECT2_CD_AREA_OVER_MASS']=mydf['c_cd_area_over_mass']
    cdm['OBJECT2_CR_AREA_OVER_MASS']=mydf['c_cr_area_over_mass']

    #OBJECT2 ORBITAL PARAMETERS
    cdm['OBJECT2_APOGEE_ALTITUDE']=mydf['c_h_apo']
    cdm['OBJECT2_PERIGEE_ALTITUDE']=mydf['c_h_per']
    cdm['OBJECT2_INCLINATION']=mydf['c_j2k_inc']

    #### To simulate that the data was obtained from a collection of CDM 
    #### this columns must be deleted.
    cdm.drop([  'TCA_timeformat',
                'time_to_tca',
                'CREATION_DATE_timeformat',
                'OBJECT1_TIME_LASTOB_START_timeformat',
                'OBJECT1_TIME_LASTOB_END_timeformat',
                'OBJECT2_TIME_LASTOB_START_timeformat',
                'OBJECT2_TIME_LASTOB_END_timeformat',
            ], inplace=True, axis=1)
    return cdm

##### CONVERT CDM TIMESTRING COLUMNS TO TIMEDATE COLUMNS
def convertTimestringToTimedate(mydf):
    """
    This function convert string timestamp from CDM to datetime python variables.
    """
    date_format="%Y-%m-%dT%H:%M:%S.%f"
    mydf['TCA']=mydf['TCA'].apply(lambda x: dt.datetime.strptime(x, date_format))
    mydf['CREATION_DATE']=mydf['CREATION_DATE'].apply(lambda x: dt.datetime.strptime(x, date_format))
    mydf['OBJECT1_TIME_LASTOB_START']=mydf['OBJECT1_TIME_LASTOB_START'].apply(lambda x: dt.datetime.strptime(x, date_format))
    mydf['OBJECT1_TIME_LASTOB_END']=mydf['OBJECT1_TIME_LASTOB_END'].apply(lambda x: dt.datetime.strptime(x, date_format))
    mydf['OBJECT2_TIME_LASTOB_START']=mydf['OBJECT2_TIME_LASTOB_START'].apply(lambda x: dt.datetime.strptime(x, date_format))
    mydf['OBJECT2_TIME_LASTOB_END']=mydf['OBJECT2_TIME_LASTOB_END'].apply(lambda x: dt.datetime.strptime(x, date_format))

    return mydf


#####
def CreateSingleRowEventDataFrame(mydf,number_of_events,progress_indicator=500):
    '''
    This function convert all the row events to a single row event dataframe.
    Default parameters take the last 6 events: 5 for training and the collision probability of the last one.
    '''
    #flag variable
    flag=0
    print("Creating dataframe...\n Starting at: {}".format(datetime.now(tz=None)))
    for i in range(number_of_events):
        #create dataframe for each event
        one_event=mydf[(mydf["event_id"]==i)]
        
        #filter only those that have more than 6 CDMs
        if len(one_event)>=6:

            if i%progress_indicator==0:
                print("Computing id_event number: {}".format(i))
            #print(one_event.iloc[0,0])

            #Convert all CDMs to single row event
            #Last CDM must be saved for TARGET PC
            single_row_event=pd.concat([one_event.iloc[-6],one_event.iloc[-5],one_event.iloc[-4],one_event.iloc[-3],one_event.iloc[-2]],axis=0).to_frame().T

            #Rename columns with sufix _1,_2,_3,_4
            cols=pd.Series(single_row_event.columns)
            for dup in single_row_event.columns[single_row_event.columns.duplicated(keep=False)]: 
                cols[single_row_event.columns.get_loc(dup)] = ([dup + '_' + str(d_idx) 
                                                                if d_idx != 0 
                                                                else dup 
                                                                for d_idx in range(single_row_event.columns.get_loc(dup).sum())]
                                                                )
            single_row_event.columns=cols

            #Append single row events in single dataframe: one row for each collision event
            #flag = 0 -> first pass of loop; flag>0 following passes 
            if flag==0:
                data=single_row_event
                flag=1
            else:
                data=data.append(single_row_event)
    print("Dataframe successfully created...\n Finished at: {}".format(datetime.now(tz=None)))
    #Save dataframe to file
    filename="full_dataframe_{}.pkl".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    print("Saving dataframe for future usage filename = {}".format(filename))
    data.to_pickle(filename)
    print("Dataframe was successfully saved at working directory: {}".format(os.getcwd()))
    return data
        
