import os

basefolder = '/gpu-data/djsanchez/Images-Endometrial_POLE_v4/Patch_Contrastive_Learning/Image_Patch_Representation/'

for f in os.listdir(basefolder):
    if 'Super' in f:
        os.rename(basefolder+f,basefolder+'Endom'+f[10:])