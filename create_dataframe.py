############################################################################
# SCRIPT TO CREATE DATASET INPUT FOR THE LIGHT GBM MODEL
# IT SAMPLES AN EQUALLY DISTRIBUTED DATAFRAME CONDISERING ONLY RISKY EVENTS 
# RISKY EVENTS ARE THOSE THAT COLLISSION_PROBABILITY IS GREATER THAN 10 E-6
#############################################################################
import pandas as pd
import datetime as dt
import numpy as np
import os

from preparing_data import *

print("------------------------------------------------------")
print("This script creates the dataframe needed as input for the computation of the Machine Learning Model")
print("----------------------------------------------------- \n")
print("Loading full Kelvin's Challenge dataframe...............")
print("............................................")
df=pd.read_csv("./data/train_data.csv")

# CONVERT KELVIN DATASET TO CDM FORMAT TO SIMULATE ACTUAL INPUT
cdm=convertKelvinDatasetToCDMFormat(df)

# DELETE NULLS FROM ONE COLUMN NEEDED TO RUN FOLLOWING TIME CONVERSIONS
cdm.dropna(subset = ["OBJECT2_TIME_LASTOB_START"], inplace=True)

# CONVERT TIME STRING TO TIMEDATE
cdm=convertTimestringToTimedate(cdm)
# CONVERT TIMEDATE TO RANGE IN DAYS
cdm=convertTimedateToDaysRange(cdm)
# CONVERT RISK IN LOGARITHMIC SCALE TO NATURAL SCALE THE SAME THAT COLLISSION PROBABILITY USES IN THE CDMs
cdm=convertPCto10logaritmicscale(cdm)

#DELETE NULS FROM ALL THER OTHER ROWS
cdm.dropna(inplace=True)

# DROP NON NUMERIC COLUMNS
numeric_cols=cdm.select_dtypes(exclude='number')
cdm.drop(numeric_cols, axis=1, inplace=True)

print("Adding correlation matrix elements to the dataframe \n")

# CALCULATE AND ADD CORRELATION COLUMNS TO IMPROVE MACHINE LEARNING MODEL
cdm=addCorrelationColumns(cdm)

#DELETE COVARIANCE MATRIX NON DIAGONAL ELEMENTS
print("Deleting covariance matrix elements from the dataframe \n")

cdm=deleteCovarianceNonDiagonalElements(cdm)
print("Dataframe size without feature engineering {} x {}".format(cdm.shape[0],cdm.shape[1]))
cdm.head()


#DELETING OBSERVATION COLUMNS NO NEEDED IN THE MODEL
cdm.drop([     'OBJECT1_TIME_LASTOB_START',
                'OBJECT1_TIME_LASTOB_END',
                'OBJECT2_TIME_LASTOB_START',
                'OBJECT2_TIME_LASTOB_END'
                ], inplace=True, axis=1)


# REORDERING COLUMNS BRING __time_to_tca TO FRONT
cdm=cdm[ ['__time_to_tca'] + [ col for col in cdm.columns if col != '__time_to_tca' ] ]


#SORT DATAFRAME BY event_id AND THEN BY __time_to_tca DESCENDING
cdm.sort_values(by=['event_id', '__time_to_tca'],ascending=[True, False],inplace=True)


# FEATURE ENGINEERING
print("-----------------------------------------------------------")
print("------------- FEATURE ENGINEERING -------------------------")
print("-----------------------------------------------------------")
print("FROM MAIN VARIABLES MISS_DISTANCE and COLLISSION_PROBABILITY...")
print("Computing...")
print("moving average with window of three elements...")
print("trends...")
print("gradients...")
print("............................................................")
################## COLLISSION_PROBABILITY ##################
# (PC_i + PC_i-1 + PC_i-2) / 3 (MOVING AVERAGE WINDOW 3)
cdm["PC_mavg_1"]=(cdm["COLLISSION_PROBABILITY"]+cdm["COLLISSION_PROBABILITY"].shift(1)+cdm["COLLISSION_PROBABILITY"].shift(2))/3
# PC_i - P_i-1 (TREND)
cdm["PC_trend_1"]=cdm["COLLISSION_PROBABILITY"]-cdm["COLLISSION_PROBABILITY"].shift(1)
# PC_i - P_i-3 (TREND)
cdm["PC_trend_3"]=cdm["COLLISSION_PROBABILITY"]-cdm["COLLISSION_PROBABILITY"].shift(3)
# ( PC_i - PC_i-1 ) / time_delta  (GRADIENT)
cdm["PC_gradient_1"]=(cdm["COLLISSION_PROBABILITY"]-cdm["COLLISSION_PROBABILITY"].shift(1))/(abs(cdm["__time_to_tca"]-cdm["__time_to_tca"].shift(1)))
# ( PC_i - PC_i-3 ) / time_delta  (GRADIENT)
cdm["PC_gradient_3"]=(cdm["COLLISSION_PROBABILITY"]-cdm["COLLISSION_PROBABILITY"].shift(3))/(abs(cdm["__time_to_tca"]-cdm["__time_to_tca"].shift(3)))

################## MISS_DISTANCE ##################
# ( _i + _i-1 + _i-2) / 3 (MOVING AVERAGE WINDOW 3)
cdm["MD_mavg_1"]=(cdm["MISS_DISTANCE"]+cdm["MISS_DISTANCE"].shift(1)+cdm["MISS_DISTANCE"].shift(2))/3
# _i - _i-1 (TREND)
cdm["MD_trend_1"]=cdm["MISS_DISTANCE"]-cdm["MISS_DISTANCE"].shift(1)
# _i -  _i-3 (TREND)
cdm["MD_trend_3"]=cdm["MISS_DISTANCE"]-cdm["MISS_DISTANCE"].shift(3)
#( _i - _i-1 ) / time_delta  (GRADIENT)
cdm["MD_gradient_1"]=(cdm["MISS_DISTANCE"]-cdm["MISS_DISTANCE"].shift(1))/(abs(cdm["__time_to_tca"]-cdm["__time_to_tca"].shift(1)))
#( _i - _i-3 ) / time_delta  (GRADIENT)
cdm["MD_gradient_3"]=(cdm["MISS_DISTANCE"]-cdm["MISS_DISTANCE"].shift(3))/(abs(cdm["__time_to_tca"]-cdm["__time_to_tca"].shift(3)))

#AUXILIAR COLUMN TO DELETE VALUES MIXING CDMs OF DIFFERENT event_id
cdm["VALID_ROW"]=cdm["event_id"]==cdm["event_id"].shift(3)
cdm=cdm[cdm["VALID_ROW"]==True]
cdm.reset_index(inplace=True)
cdm.drop(["index","VALID_ROW","event_id"], inplace=True, axis=1)


# SELECTING DATA TO BUILD MODEL

print("Building dataframe... \n")
aux1=cdm[(cdm["COLLISSION_PROBABILITY"]>-4)& (cdm["__time_to_tca"]<1)]
aux2=cdm[(cdm["COLLISSION_PROBABILITY"]<-4) & (cdm["COLLISSION_PROBABILITY"]>-5)& (cdm["__time_to_tca"]<1)]
aux3=cdm[(cdm["COLLISSION_PROBABILITY"]<-5) & (cdm["COLLISSION_PROBABILITY"]>-6)& (cdm["__time_to_tca"]<1)]


# APPEND SUBPART OF DATAFRAMES EVENTS WITH PROBABILITIES LOWER THAN 10-6 TO CREATE AN EQUALLY DISTRIBUITED PROBABILITY DATAFRAME
data=(aux1.append(aux2)).append(aux3)
print("Dataframe final size {} x {}".format(data.shape[0],data.shape[1]))

# CREATE PICKLE FILE TO LOAD WHEN NEEDED
print("Creating pickle file... \n")
filename="./dataframe/df_{}.pkl".format(dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
print("Saving dataframe for future usage filename = {}".format(filename))
data.to_pickle(filename)
