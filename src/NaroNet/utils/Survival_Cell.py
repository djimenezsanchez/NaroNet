import os
import numpy as np
import pandas as pd

BaselData = pd.read_excel('/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_Cell_v4/Raw_Data/Experiment_Information/Patient_to_Image.xlsx')
output_folder = '/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_Cell_v4/Patch_Contrastive_Learning/Image_Patch_Representation/'
input_folder = '/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_Cell_v4/Patch_Contrastive_Learning/Image_Patch_Representation(1Im)/'
images = os.listdir('/gpu-data/djsanchez/Images-ZuriBaselRisk_1Im_Cell_v4/Patch_Contrastive_Learning/Image_Patch_Representation(1Im)/')

# Iterate over images
for n_file in range(len(images)): 
    index = list(BaselData['Image_Name']).index(images[n_file][:-3]+'tiff')
    subject_name = BaselData['Subject_Name'][index]
    if str(subject_name)+'.npy' in os.listdir(output_folder):
        previous_patient = np.load(output_folder+str(subject_name)+'.npy')
        image = np.load(input_folder+images[n_file]) + previous_patient[:,[0,1]].max() + 100
        updated_patient = np.concatenate((previous_patient,image),axis=0)
        np.save(output_folder+str(subject_name)+".npy",updated_patient)
    else:
        np.save(output_folder+str(subject_name)+".npy",np.load(input_folder+images[n_file]))  
