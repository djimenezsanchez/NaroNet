import pandas as pd
import os
import numpy as np

def load_table(path):
    return pd.read_excel(path)

def meanerror_predictions_and_labels(image_names,predictions,labels,risk_group):
    meanerror = []
    for n_image, image_label in enumerate(labels):
        if image_label>risk_group[0] and image_label<risk_group[1]:
            meanerror.append(abs(image_label-max(float(predictions[n_image].split('[')[1].split(']')[0]),0)))
    return np.array(meanerror).mean(), np.array(meanerror).std()



table = load_table('/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_v4/NaroNet/OSmonth_Risk_3_classes/Cross_validation_results/Prediction_values_OSmonth_Fold10.xlsx')

image_names = table['Subject_name']
predictions = table['Prediction']
labels = table['Label']

intervals = [(0,53),(54,119),(120,210)]

for risk_group in intervals:
    meanerror, stderror = meanerror_predictions_and_labels(image_names,predictions,labels,risk_group) 
    print('Risk Group: '+str(risk_group)+'_Mean errror: '+str(meanerror)+'+-'+str(stderror))


