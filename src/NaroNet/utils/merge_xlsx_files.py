import os
import pandas

Patient_label = pandas.read_excel('/gpu-data/djsanchez/Images-ZuriBaselRisk_v4/Raw_Data/Experiment_Information/Image_Labels.xlsx')

Patient_Im_label = pandas.read_excel('/gpu-data/djsanchez/Images-ZuriBaselRisk_v4/Raw_Data/Experiment_Information/ppp.xlsx')

# Add subject information to each image.
for image in Patient_Im_label.iterrows():
    
    # Search for the subject information
    for subject in Patient_label.iterrows():
                
        if subject[1]['Image_Names'] == image[1]['Subject_Name']:
            # Assign subject info to the image info
            Patient_Im_label.Grade[image[0]] = subject[1]['Grade']
            Patient_Im_label.ClinicalSubtype[image[0]] = subject[1]['ClinicalSubtype']
            Patient_Im_label.OSmonth[image[0]] = subject[1]['OSmonth']
            Patient_Im_label.Risk_3_classes[image[0]] = subject[1]['Risk_3_classes']                        
            Patient_Im_label.Risk_2_classes[image[0]] = subject[1]['Risk_2_classes']                        
            Patient_Im_label.Risk_2_classesIvsII[image[0]] = subject[1]['Risk_2_classesIvsII']                                

Patient_Im_label.to_excel('/gpu-data/djsanchez/Images-ZuriBaselRisk_v4/Raw_Data/Experiment_Information/Image_Labels(1pat1im).xlsx')  