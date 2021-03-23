import os 

directoryName = '/gpu-data/djsanchez/Images-SyntheticV2_v4/Patch_Contrastive_Learning/Image_Patch_Representation/'

for d in os.listdir(directoryName):
    if 'Super' in d:
        os.rename(directoryName+d,directoryName+'Image'+d[10:])