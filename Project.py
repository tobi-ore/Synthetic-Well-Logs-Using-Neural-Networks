# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:34:14 2019

@author: Tobi

description: This project involves using an AI-based model to predict well log measurements. 

data: The data utilized is a fairly large dataset of well logs from the Volve field located in the North Sea. 

"""
# import required packages and libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#%% Data Loading
# load the well data into one DataFrame
data = pd.DataFrame()  #empty DataFrame to serve as container for the data

directory = r".\Logs"  #Directory containing the well data in .csv format

#loops through all the files in the directory, reads the csv and appends it to the data DataFrame
for file in os.listdir(directory):
    x = pd.read_csv(os.path.join(directory, file))
    data = data.append(x, sort=False)
#%%
# A summary of the new DataFrame (data) created
print("The number of rows of the new DataFrame is: {}".format(data.shape[0]))
print("The number of columns of the new DataFrame is: {}".format(data.shape[1]))
print("The number of wells in the new DataFrame is: {}".format(data['wellName'].nunique()))
print("The number of Logs in the new DataFrame is: {}".format(data.shape[1] -4))
#%% Data Wrangling
# Removes the 'datasetName' column
data.drop('datasetName', inplace=True, axis=1)

#Replace all '-9999' values with NaN
data.replace(-9999, np.nan, inplace = True)

#Removes any column that has less than 10,000 measurements
for col in data:
    if data[col].count() < 10000:
        data.drop(col, inplace=True, axis=1)
#%%
# A summary of the DataFrame (data) after cleaning and manipulation
print("The number of rows of the new DataFrame is: {}".format(data.shape[0]))
print("The number of columns of the new DataFrame is: {}".format(data.shape[1]))
print("The number of wells in the new DataFrame is: {}".format(data['wellName'].nunique()))
print("The number of Logs in the new DataFrame is: {}".format(data.shape[1]-3))
#%% Cross match tables
#A table that shows the count of valid well log measurements for each wells
cross_match_1 = data.groupby('wellName').count()
cross_match_1.to_excel(r'cross_match_1.xls')

#A table that shows the percentage of valid well log measurements
cross_match_2 = cross_match_1.div(cross_match_1['MD (M)'], axis=0)*100
cross_match_2 = round(cross_match_2, ndigits=1)  #rounds the values to 1 decimal place
cross_match_2.to_excel(r'cross_match_2.xls')

#A table that puts "X" where there is at least one measurement of the logs for each well in the dataset
cross_match_3 = cross_match_2.replace(0,np.nan)
cross_match_3[cross_match_3>0] = 'X'
cross_match_3.to_excel(r'cross_match_3.xls')
#%% PHASE 2
#check for wells with BS and CALI
well_BS_CALI = []
for well in cross_match_3.index:
    if cross_match_3.loc[well]['CALI (inches)'] == "X" and cross_match_3.loc[well]['BS (IN)'] == "X":
        well_BS_CALI.append(well)
#%%
###creates a folder to save the log images
###checks if a folder called Log_Images already exists to avoid error
if os.path.exists(os.path.join(os.path.split(directory)[0], "Log_Images")):
    pass
else:
    os.mkdir(os.path.join(os.path.split(directory)[0], "Log_Images")) 

#plot for each well
for well in well_BS_CALI:
    MD = data[data['wellName']==well]['MD (M)']
    BS = data[data['wellName']==well]['BS (IN)']
    CALI = data[data['wellName']==well]['CALI (inches)']
    GR = data[data['wellName']==well]['GR (API)']

    #check what other logs are in the well
    other_logs = []
    for log, value in cross_match_3.loc[well].iteritems():
        if value == 'X':
            other_logs.append(log)

    #exclude already plotted logs and depth measurement
    for plotted_log in ('MD (M)','BS (IN)','CALI (inches)','GR (API)','TVD (M)'):
        while plotted_log in other_logs: 
            other_logs.remove(plotted_log)

    x= np.random.choice(other_logs,3)   #randomly select 3 well logs not yet plotted

    fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(20,30), sharey=True)
    fig.suptitle("Well {} Log Display".format(well), fontsize=25)
    fig.subplots_adjust(top=0.93)
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel('MD (M)',fontsize=20)
    ax[0].yaxis.grid(True)
    
    ##Track 1
    ##Bit_size and Caliper
    ax_BS = ax[0].twiny()
    ax_BS.plot(BS,MD, color='brown')
    ax_BS.set_xlabel('BS (in)',color='brown',fontsize=15)
    ax_BS.tick_params('x',colors='brown')  ##change the color of the x-axis tick label
    ax_BS.set_xlim([8,9])

    ax_CALI = ax[0].twiny()
    ax_CALI.plot(CALI,MD, color='red',ls=':')
    ax_CALI.set_xlabel('CALI (in)',color='red',fontsize=15)
    ax_CALI.tick_params('x',colors='red')
    ax_CALI.set_xlim([8,9])
    ax_CALI.spines['top'].set_position(('outward',40)) ##move the x-axis up

    ax_BS.fill_betweenx(MD,BS,CALI, color='yellow')
    ax_BS.grid(True,alpha=0.5)

    ax[0].get_xaxis().set_visible(False) #removing the x-axis label at the bottom of the fig

    ##Track 2
    ##Gamma_ray
    ax_GR = ax[1].twiny()
    ax_GR.plot(GR,MD, color='black')
    ax_GR.set_xlabel('GR (API)',color='black',fontsize=15)
    ax_GR.tick_params('x',colors='black')  ##change the color of the x-axis tick label
    ax[1].get_xaxis().set_visible(False)
    ax[1].yaxis.grid(True)

    ax_GR.fill_betweenx(MD,GR,75, where = GR>75, color='brown')
    ax_GR.fill_betweenx(MD,GR,75, where = GR<75, color='yellow')
    ax_GR.grid(True,alpha=0.5)

    ##Track 3,4 and 5 (PLOTTING THE 3 RANDOMLY SELECTED LOGS)
    color = ['green','blue','purple']
    for i in range(0,3):
        ax_3 = ax[i+2].twiny()
        ax_3.plot(data[data['wellName']==well][x[i]], MD, color=color[i])
        ax_3.set_xlabel(x[i],color=color[i],fontsize=15)
        ax_3.tick_params('x',colors=color[i])  ##change the color of the x-axis tick label
        ax[i+2].get_xaxis().set_visible(False)
        ax[i+2].yaxis.grid(True)
        ax_3.grid(True,alpha=0.5)
    fig.savefig(r'{}\Log_Images\{}.png'.format(os.path.split(directory)[0], well.replace('/','_')), dpi=300)
    plt.close()
#%% PHASE 3
#remove logs available in less than 15 wells
for col in cross_match_3:
    if cross_match_3[col].count() < 16:
        cross_match_3.drop(col, inplace=True, axis=1)

print("The number of rows of the new DataFrame is: {}".format(data.shape[0]))
print("The number of columns of the new DataFrame is: {}".format(cross_match_3.shape[1]))
print("The number of wells in the new DataFrame is: {}".format(cross_match_3.shape[0]))
print("The number of Logs in the new DataFrame is: {}".format(cross_match_3.shape[1]-2))
#%%
#remove wells that have less than 16 logs
model_cross_match = cross_match_3[cross_match_3.count(axis=1)>15]  #new cross match table
model_cross_match.to_excel(r"model_cross_match.xls")

new_data = data.set_index("wellName")

new_data = new_data.loc[model_cross_match.index,cross_match_3.columns]

print("The number of rows of the new DataFrame is: {}".format(new_data.shape[0]))
print("The number of columns of the new DataFrame is: {}".format(new_data.shape[1]))
print("The number of wells in the new DataFrame is: {}".format(model_cross_match.shape[0]))
print("The number of Logs in the new DataFrame is: {}".format(model_cross_match.shape[1]-2))
#%% 
#remove any incomplete rows

new_data.dropna(axis=0, inplace=True)

print("The number of rows of the new DataFrame is: {}".format(new_data.shape[0]))
print("The number of columns of the new DataFrame is: {}".format(new_data.shape[1]))
print("The number of wells in the new DataFrame is: {}".format(model_cross_match.shape[0]))
print("The number of Logs in the new DataFrame is: {}".format(model_cross_match.shape[1]-2))
#%% Feature Selection
correlation = new_data.corr()

ticks = np.arange(new_data.shape[1])
variables = new_data.columns

fig, ax = plt.subplots(figsize=(12,12))
my_plot = ax.matshow(correlation, vmin=-1, vmax=1, cmap='RdGy')
plt.colorbar(my_plot, shrink=0.7)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(variables,rotation = 90)
ax.set_yticklabels(variables)

fig.tight_layout()
fig.savefig(r'{}\{}.png'.format(os.path.split(directory)[0], "correlation"), dpi=900)
#%% drop logs with high correlation

new_data_clean = new_data.drop(["MD (M)", "TVD (M)", "PHIF (V/V)", "SAND_FLAG"], axis = 1)

stat= new_data.describe() 

#%% input and output data allocation
#
y_total = new_data_clean.loc[:,'VSH (V/V)']
X_total = new_data_clean.loc[:, new_data_clean.columns != 'VSH (V/V)'] 

data_blind_well = new_data_clean.loc["15/9-F-1"]
new_data_clean.drop("15/9-F-1", axis = 0, inplace =True)

#dataset used for training and validation
y = new_data_clean.pop('VSH (V/V)').values.reshape(-1,1)      
X = new_data_clean  

#blind well used for verification
y_blind = data_blind_well.pop('VSH (V/V)').values.reshape(-1,1)
X_blind = data_blind_well 


#%% model building
hyper_parameters =  {'hidden_layer_sizes': [(50,),(100,),(150,),(200,),
                                            (50,50),(100,100),(150,150),(200,200)],
                     #'max_iter': [200,300,500],
                     'solver':['lbfgs', 'adam']}


gs = GridSearchCV(MLPRegressor(), param_grid=hyper_parameters, cv=5, scoring='r2', verbose=1, n_jobs=2)

normalized_model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('predictor', gs)
])

normalized_model.fit(X,y)       #training the model

#%%
# gs.best_params_
# gs.best_estimator_
# gs.cv_results_
# gs.best_score_
# =============================================================================

#%%


y_pred_blind = normalized_model.predict(X_blind)

y_total_pred = normalized_model.predict(X_total)

R2_training = gs.best_score_
R2_blind = metrics.r2_score(y_blind,y_pred_blind)
R2_total = metrics.r2_score(y_total,y_total_pred)

#%%ploting the actual and predicted logs for the wells
###creates a folder to save the model images
###checks if a folder called Model_Images already exists to avoid error
if os.path.exists(os.path.join(os.path.split(directory)[0], "Model_Images")):
    pass
else:
    os.mkdir(os.path.join(os.path.split(directory)[0], "Model_Images"))
    
    
y_total = pd.DataFrame(y_total)
y_total['predicted'] = y_total_pred
y_total['MD'] = new_data.loc[:,'MD (M)']

for well in y_total.index.unique():
    
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,12), sharey=True)
    fig.suptitle("Well {} Display of Actual and Predicted VSH".format(well), fontsize=25)
    
    
    ax[0].invert_yaxis()
    ax[0].set_ylabel('MD (M)',fontsize=20)
    ax[0].yaxis.grid(True)
    
    ##Track 1
    ##actual and predicted logs
    ax_actual = ax[0].twiny()
    ax_actual.plot(y_total.loc[well,'VSH (V/V)'],y_total.loc[well,'MD'], color='black')
    ax_actual.set_xlabel('Actual Log',color='black',fontsize=15)
    ax_actual.tick_params('x',colors='black')  ##change the color of the x-axis tick label
    ax_actual.set_xlim([0,1])
    
    ax_predicted = ax[0].twiny()
    ax_predicted.plot(y_total.loc[well,'predicted'],y_total.loc[well,'MD'], color='red',ls=':')
    ax_predicted.set_xlabel('Predicted Log',color='red',fontsize=15)
    ax_predicted.tick_params('x',colors='red')
    ax_predicted.spines['top'].set_position(('outward',40)) ##move the x-axis up
    ax_predicted.set_xlim([0,1])
    ax_actual.grid(True,alpha=0.5)
    
    ax[0].get_xaxis().set_visible(False) #removing the x-axis label at the bottom of the fig
    
    ##Track 2
    ##Error
    ax_error = ax[1].twiny()
    ax_error.plot(y_total.loc[well,'VSH (V/V)']-y_total.loc[well,'predicted'],y_total.loc[well,'MD'], color='black')
    ax_error.set_xlabel('Error (Actual - Predicted)',color='black',fontsize=15)
    ax_error.tick_params('x',colors='black')  ##change the color of the x-axis tick label
    
    ax[1].get_xaxis().set_visible(False)
    ax[1].yaxis.grid(True)
    
    ax_error.grid(True,alpha=0.5)

    plt.tight_layout()
    fig.subplots_adjust(top=0.83)
    
    fig.savefig(r'{}\Model_Images\{}.png'.format(os.path.split(directory)[0], well.replace('/','_')), dpi=300)
    plt.close()
