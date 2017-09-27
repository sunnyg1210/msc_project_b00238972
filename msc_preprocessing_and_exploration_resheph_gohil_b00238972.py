# Resheph Gohil (B00238972)
# MSc Big Data
# University of the West of Scotland

import pandas as pd
import numpy as np
from datetime import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import metrics as mts
from sklearn.cross_validation import train_test_split as tts

sns.set(style="whitegrid", palette="Set3", font="sans-serif", font_scale=1.5)

# read dataset into pandas dataframe with name "initData"
initData = pd.read_csv('data/diabetic_data.csv')

##################
# DATA EXPLORATION
##################
initData.columns # display all columns in dataset
initData.head() # explore first 5 rows of dataset
initData.isnull().any() # checking for any null values in any of the columns
initData.describe() # brief overview of the dataset

initData.race.unique() # get unique values in the race column 
initData.weight[initData.weight == '?'].count() # get count of values in weight column where value = ?

initData.payer_code.unique() # get unique values in the payer_code column
initData.payer_code[initData.payer_code == '?'].count() # get count of values in payer column where value = ?

initData.medical_specialty.unique() # get unique values in the medical_specialty column
initData.medical_specialty[initData.medical_specialty == '?'].count() # get count of values in medical_specialty column where value = ?

initData.diag_2.unique() # get unique values in the diag_2 column
initData.diag_2[initData.diag_2 == '?'].count() # get count of values in diag_2 column where value = ?

initData.diag_3.unique() # get unique values in the diag_3 column
initData.diag_3[initData.diag_3 == '?'].count() # get count of values in diag_3 column where value = ?

initData.num_medications.unique() # get unique values in the num_medications column

initData.race.unique() # get unique values in the race column
initData = initData[~(initData['race'] == '?')] # remove all rows in race column where value = ?

initData.gender.unique() # get unique values in the gender column
initData.gender[(initData['gender'] == 'Unknown/Invalid')].count() # get count of values in gender column where value = Unknown/Invalid
initData = initData[~(initData['gender'] == 'Unknown/Invalid')] # remove all rows whos gender is unknown/invalid

len(initData) # get number of samples in dataset

initData = initData[~(initData['admission_type_id'] == 5) & ~(initData['admission_type_id'] == 6)] # remove all rows whos admission type is either 'not available' or 'null'
len(initData) # get number of samples in dataset

initData.admission_source_id.unique() # get unique values in the admission_source_id column
initData = initData[
    ~(initData['admission_source_id'] == 9) & 
    ~(initData['admission_source_id'] == 15) & 
    ~(initData['admission_source_id'] == 17) &
    ~(initData['admission_source_id'] == 20) &
    ~(initData['admission_source_id'] == 21)
] # remove all rows whos admission source is either unknown, not available, or not mapped
len(initData) # get number of samples in dataset

initData.age.unique() # get unique values in the age column
ageDistribution = sns.countplot(x="age", data=initData).set_title('Age Distribution of Diabetic Patients') # create bar chart displaying age distribution of patients
ageDistribution.figure.savefig('graphs_findings/patient_age_distribution.png') # save generated graph

# Only keep rows where age is 10 or above
initData = initData[~(initData['age'] == '[0-10)')]
len(initData) # get number of samples in dataset

initData.gender[initData.gender == 'Male'].count()
initData.gender[initData.gender == 'Female'].count()

ms_data = initData.groupby('medical_specialty')['medical_specialty'].count() # count patients by medical specialty
ms_data.plot(kind="bar", figsize=(35, 8)) # generate bar chart for patient count by medical_specialty column
ms_data.figure.savefig('graphs_findings/data_spread_by_medical_specialty.png') # save generated graph

len(list(initData.columns)) # count total number of features

initData.drop([
    'patient_nbr','encounter_id','weight','payer_code','diag_1','diag_2','diag_3','metformin','medical_specialty',
    'metformin-pioglitazone','metformin-rosiglitazone', 'glimepiride-pioglitazone','glipizide-metformin','glyburide-metformin',
    'insulin','citoglipton','examide','tolazamide','troglitazone','miglitol','acarbose','rosiglitazone','pioglitazone',
    'tolbutamide','glyburide','glipizide','acetohexamide','glimepiride','chlorpropamide','nateglinide','repaglinide',
    'metformin','number_diagnoses'
], axis=1, inplace=True) # dropping unwanted/unnecessary features
initData.columns # display remaining columns
len(initData) # get number of samples in dataset

genderDistribution = sns.countplot(x="gender", data=initData).set_title('Gender Distribution of Diabetic Patients') # Create Overall Gender Distribution Graph
genderDistribution.figure.savefig('graphs_findings/gender_distribution.png') # save graph

raceDistribution = sns.countplot(y="race", data=initData).set_title('Race Distribution of Diabetic Patients') # Create Overal race distribution graph
raceDistribution.figure.savefig('graphs_findings/race_distribution.png') # Save graph

raceByGender = sns.countplot(y="race", hue="gender", data=initData).set_title('Overall Gender Distribution by Race of Diabetic Patients') # Create Overall race distribution graph by gender
raceByGender.figure.savefig('graphs_findings/gender_distribution_by_race.png') # Save graph

initData.max_glu_serum.unique() # get unique values in the max_glu_serum column
abc = initData.groupby('max_glu_serum')['max_glu_serum'].count() # count patients by max_glu_serum
abcplot = abc.plot(kind="bar", figsize=(35, 8)) # Create graph for max_glu_serum
abcplot.set_ylabel('No. of Patients') # set label for y axis
abcplot.set_xlabel('Glucose Serum Test Result') # set label for x axis
initData.drop(['max_glu_serum'], axis=1, inplace=True) # drop max_glu_serum feature

initData.A1Cresult.unique() # get unique values in the A1Cresult column
a1c = initData.groupby('A1Cresult')['A1Cresult'].count() # count patients by A1Cresult
a1cplot = abc.plot(kind="bar", figsize=(35, 8)) # Create graph for A1Cresult
a1cplot.set_ylabel('No. of Patients') # set label for y axis
a1cplot.set_xlabel('A1Cresult Test Result') # set label for x axis
a1cplot.figure.savefig('graphs_findings/a1cresult_test.png') # save graph

initData.readmitted.unique() # get unique values in the readmitted column
readmitMed = sns.set(style="whitegrid", palette="Set2", font="sans-serif", font_scale=1.5) # set custom graph styling
readmitMed = sns.countplot(x="readmitted", hue="diabetesMed", data=initData, orient="v").set_title('Readmission stats based on medication prescription') # create graph to show readmission stats based on medication prescription

mph = initData.pivot_table(index='race', columns='time_in_hospital', values='num_medications') # create new table where index = race, columns = time in hospital and values = number of medications
mph # display created table
plt.figure(figsize=(20,10)) # set graph size
mph_plot = sns.heatmap(mph, annot=True, annot_kws={"size": 15}, fmt=".0f", cmap="Reds", linewidths=.5, cbar_kws={"orientation": "vertical"}) # create heatmap of data in crated table
mph_plot.set_title('Number of medications (avg/day) consumed by patients based on their race and time in hospital', size=20) # set title of heatmap
mph_plot.set_xlabel('Days spent in hospital', size=20) # set label for x axis of heatmap
mph_plot.set_ylabel('Patient Race', size=20) # set label for y axis of heatmap
plt.show() # display heatmap
mph_plot.figure.savefig('graphs_findings/meds_consumed_avg_per_day.png') # save figure

lph = initData.pivot_table(index='A1Cresult', columns='time_in_hospital', values='num_lab_procedures') # create new table where index = A1Cresult, columns = time in hospital and values = number of lab procedures
lph # display created table
plt.figure(figsize=(20,10)) # set graph size
lph_plot = sns.heatmap(lph, annot=True, annot_kws={"size": 15}, fmt=".0f", cmap="Blues", linewidths=.5, cbar_kws={"orientation": "vertical"}) # create heatmap of data in crated table
lph_plot.set_title('Number of lab procedures conducted on patients based on their HBA1c and time in hospital', size=20) # set title of heatmap
lph_plot.set_xlabel('Days spent in hospital', size=20) # set label for x axis of heatmap
lph_plot.set_ylabel('HBA1c Identifier', size=20) # set label for y axis of heatmap
plt.show() # display heatmap
mph_plot.figure.savefig('graphs_findings/meds_consumed_avg_per_day.png') # save figure

admissionsBySource = initData.admission_source_id.groupby(initData.admission_source_id).count().plot(kind="bar").set_title('Count of admissions by referer') # create graph admissions by referer

sns.set(style="darkgrid", palette="Set2", font="sans-serif", font_scale=1.5) # set custom graph styling
lengthOfStay = initData.time_in_hospital.groupby(initData.time_in_hospital).count().plot(kind="bar").set_title('Count of patients by their Length of Stay') # create graph for count of patients by length of stay

#####################
# DATA TRANSFORMATION
#####################
# Mapping binary values to correspond to existing string values
initData['A1Cresult'] =  initData.A1Cresult.map({'None': 0, 'Norm': 1, '>7': 2, '>8': 3})
initData['change'] =  initData.change.map({'Ch': 1, 'No': 0})
initData['diabetesMed'] =  initData.diabetesMed.map({'Yes': 1, 'No': 0})
initData['readmitted'] =  initData.readmitted.map({'NO': 0, '>30': 1, '<30': 2})
initData['gender'] = initData.gender.map({"Female":0, "Male":1})

initData.race.unique()
initData['race'] = initData.race.map({"Caucasian": 0, "AfricanAmerican": 1, "Other": 2, "Hispanic": 3, "Asian": 4})

initData.age.unique()
initData['age'] = initData.age.map({"[10-20)": 0, "[20-30)": 1, "[30-40)": 2, "[40-50)": 3,"[50-60)": 4, "[60-70)": 5, "[70-80)": 6, "[80-90)": 7, "[90-100)": 8})

# Creating new dataframes based on the originally created dataframe
mlDf_scikit = initData
mlDf_tensorflow = initData

# A function that maps binary values based on condition for time_in_hospital feature
def los_binary_scikit(value):
    if value >= 0 and value <= 7:
        return 0
    elif value >= 8:
        return 1
mlDf_scikit['los'] = mlDf_scikit['time_in_hospital'].apply(los_binary_scikit)
mlDf_scikit.los.unique()

# A function that maps binary values based on condition for time_in_hospital feature
def los_binary_tensorflow(value):
    if value >= 0 and value <= 7:
        return 0
    elif value >= 8 and value <= 10:
        return 1
    elif value >= 11:
        return 2
mlDf_tensorflow['los'] = mlDf_tensorflow['time_in_hospital'].apply(los_binary_tensorflow)
mlDf_tensorflow.los.unique()

# Exporting both dataframes out to the data directory
mlDf_scikit.to_csv('data/mldata_scikit.csv', index=None)
mlDf_tensorflow.to_csv('data/mldata_tensorflow.csv', index=None)