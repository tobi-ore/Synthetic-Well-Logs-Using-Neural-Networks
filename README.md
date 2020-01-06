# Synthetic-Well-Logs-Using-Neural-Networks
## Objectives
The aim of this project is to build a model that has the capabilities of predicting the well log measurements from other geophysical logs using an Artificial Neural Network. The workflow is typical for any similiar project and it entails:
•	Wrangle and process the data to the desired format;
•	Carry out quick look and statistical analysis of the well log data;
•	Develop a model using a neural network framework; and
•	Deploy the model and investigate its performance.


## Data Source
The geophysical logs used in this study are from the Volve field located in the southern part of the Norwegian North Sea. The data is made available through the open-source subsurface data published by Equinor to aid researchers and academics with real-life data. In this project, twenty-one (21) wells are available for study with a total of fifty-seven (57) geophysical logs with variable availability in the wells. 


## Implementation
The analysis of the well logs will be carried out using R and Python. However, bash and wget will be utilized to download and manipulate the data into the desired format.

__Software__: `Python`

__Packages__: `sklearn`, `numpy`, `pandas` and `matplotlib`

To download the data, run  `bash data_download.sh` in the command line.



The various manipulations and operations on the well logs in done by running the python script called `project.py`. 

## Expected Products
The Output folder contains all the expected products from this project. To test this code, do not download the Output folder from this repo. Again, the data_csv folder is also an output which is used by the R script. download this folder iof you do not have Python on your system.

_Log images_
A folder that contains all the well log images named after the name of the respective wells.

_Vsh images_
A folder that contains the plot of calculated volume of shale for each wells.

_Interpreted logs_
A folder that contains the final csv for each wells with the calculated properties.

## Author

[Tobi Ore](https://github.com/tobi-ore)

## License

[This project is licensed under the MIT License](https://choosealicense.com/licenses/mit/)
