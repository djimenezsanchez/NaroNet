from math import sqrt
from NaroNet.utils.utilz import auc_roc_curve
from NaroNet.utils.utilz import confusion_matrix

def confidence_interval_accuracy(accuracy,n_samples,Experiment_Name):
    z=1.96 # in 95% confidence interval
    interval = z * sqrt( (accuracy * (1 - accuracy)) / n_samples)
    print(Experiment_Name+', Accuracy is: {}%+-{}'.format(accuracy,interval))

# seed as early as possible

import numpy as np
np.random.seed(1234)
rng=np.random.RandomState(1234)



# Calculate confidence_interval_accuracy
confidence_interval_accuracy(accuracy=0.9167,n_samples=12,Experiment_Name='POLE subject-wise')
confidence_interval_accuracy(accuracy=0.9673,n_samples=336,Experiment_Name='POLE image-wise')

# Calculate AUC with confidence interval
# confidence_interval_AUC(y_true, y_pred)


# Recalculate roc curve for pairs of labels.
import pandas as pd
import numpy as np
process_dir = '/gpu-data/djsanchez/Images-ZuriBaselRisk_v4/NaroNet/Risk_3_classes_OSmonth/Cross_validation_results/'
Image_preds = pd.read_excel(process_dir+'Prediction_values_Risk_3_classes_Fold10.xlsx')
subject_preds = []
subject_labels = []
for S_N in np.unique(Image_preds['Subject_name']):
    subject_labels.append(list(Image_preds['Label'][np.array(Image_preds['Subject_name']==S_N)])[0])
    Image_predictions_str = list(Image_preds['Prediction'][np.array(Image_preds['Subject_name']==S_N)])
    Image_predictions=[]
    for i_p_s in Image_predictions_str:
        Image_predictions.append([float(i) for i in i_p_s.split('[')[1].split(']')[0].split(' ') if not ''==i ])
    subject_preds.append(np.stack(Image_predictions).mean(0))

thresholds,fpr,tpr = auc_roc_curve(['R2','R1','R3'],np.stack(subject_labels),
                   np.stack(subject_preds),process_dir,'POLEE','subject_wise')                              
confusion_matrix(['R2','R1','R3'],np.stack(subject_labels),np.stack(subject_preds),
                process_dir,thresholds,fpr,tpr,'POLEE','subject_wise')

