import csv
import os
import fnmatch
import numpy as np
from mahotas.features.texture import haralick_labels
import nibabel as nib

import ExtendGaborFeatures
import Biopsy5by5WindowV3
import FirstOrderFeatures
import MultiModality

def String_f(listOfStrings,Pstrings):
    for Strings in listOfStrings:
        if fnmatch.fnmatch(Strings,Pstrings):
            return Strings


def Read3DImage(ImagefileName,Dataform):
    img3nii = nib.load(ImagefileName)
    img3Array_temp = img3nii.get_fdata()
    if fnmatch.fnmatch(Dataform, "BrainNormalized"):
        img3Array = MultiModality.GrayScaleNormalizationK(img3Array_temp,255.0,img3Array_temp.min(),img3Array_temp.max())
    elif fnmatch.fnmatch(Dataform, "NoNormalized"):
        img3Array = img3Array_temp
    return img3Array

def Read2DImage(img3Array, slicenum):
    imgArray_temp = img3Array[:, :, slicenum - 1]
    #imgArray_1=ImageReadModality_GBM.rotate(imgArray_temp.T, -180)
    imgArray_1=imgArray_temp.T
    imgArray=np.matrix(imgArray_1)
    return imgArray

def Read2DImageROI(img3Array, slicenum):
    imgArray_temp = img3Array[:, :, slicenum - 1]
    #imgArray_1=ImageReadModality_GBM.rotate(imgArray_temp.T, -180)
    imgArray_1=imgArray_temp
    imgArray=np.matrix(imgArray_1)
    return imgArray

def genTextures(Patientfoldername,Datafile,  slicenum):
    ###Texture Name
    GLCMAngleList = ['Avg']
    featureTitle = ['Image Contrast', 'Image Filename', 'X', 'Y', 'Boundary (1) or not (inside: 0), (outside:2)',
                    'Biopsy(1) or not (0)']

    Distances = (1, 3)
    for Distance in Distances:
        for GLCMAngle in GLCMAngleList:
            for featureName in haralick_labels[:-1]:
                featureTitle.append(featureName + '_' + GLCMAngle + '_' + str(Distance))

    Gaborsigma_range = (0.4, 0.7)
    Gaborfreq_range = (0.1, 0.3, 0.5)
    GaborFeatureList = ['Gabor_Mean', 'Gabor_Std']

    for GaborSigma in Gaborsigma_range:
        for GaborFreq in Gaborfreq_range:
            for featureName in GaborFeatureList:
                featureTitle.append(featureName + '_' + str(GaborSigma) + '_' + str(GaborFreq))

    Gaborkernel_bank = ExtendGaborFeatures.genKernelBank(Gaborsigma_range, Gaborfreq_range)

    FirstOrderFeatureNames = FirstOrderFeatures.FirstOrderFeatureNames()
    featureTitle = featureTitle + FirstOrderFeatureNames

    # start to do T1
    featuresOutFn = 'Texture.csv'
    T2featuresOutFn = Patientfoldername + '_' + '_slice' + str(slicenum) + '_' + Modality + '_' + featuresOutFn
    featuresCSVFn = os.path.join(outputDir, T2featuresOutFn)

    with open(featuresCSVFn, 'w', newline='') as featureCSVFile:
        featureWriter = csv.writer(featureCSVFile, dialect='excel')
        featureWriter.writerow(featureTitle)

        Tdim = np.shape(ROIImage)
        RowImagedim = Tdim[0]
        ColImagedim = Tdim[1]

        for Rowi in range(RowImagedim):
            for Coli in range(ColImagedim):
                ROIvalue = ROIImage[Rowi, Coli]

                if ROIvalue != 0:
                    Rowcoord = Rowi+1
                    Colcoord = Coli+1

                    if ROIImage[Rowi + 1, Coli] == 0 or ROIImage[Rowi - 1, Coli] == 0 or ROIImage[
                        Rowi, Coli + 1] == 0 or ROIImage[Rowi, Coli - 1] == 0:
                        boundaryornot = 1
                    else:
                        boundaryornot = 0

                    biopsyornot = 0

                    aFeature = [Modality,Datafile, Colcoord, Rowcoord, boundaryornot, biopsyornot]

                    Alltextures = Biopsy5by5WindowV3.TextureExtraction(Rowcoord - 1, Colcoord - 1, dicomImageBrain,
                                                                       dicomImageRaw,
                                                                       Distances, Gaborkernel_bank)

                    aFeature = aFeature + Alltextures
                    featureWriter.writerow(aFeature)


###########################
Root= 'D:\\Rec GBM\\'
Patientfolderdir=Root+'Recurrent_v_TxEffect_PlateSeq\\Data\\Normalized\\'
outputDir= Root+'General code output\\Recurrent_v_TxEffect_PlateSeq\\Texture 5by5\\'

for Patientfoldername in os.listdir(Patientfolderdir):
    if Patientfoldername.startswith('.'):
        continue
    if Patientfoldername.startswith('..'):
        continue
    if fnmatch.fnmatch(Patientfoldername, "desktop.ini"):
        continue
    if fnmatch.fnmatch(Patientfoldername, "__MACOSX"):
        continue

    print(Patientfoldername)

    Patient_folder_path = Patientfolderdir + Patientfoldername

    if fnmatch.fnmatch(Patientfoldername, "*CU1349*") or fnmatch.fnmatch(Patientfoldername, "*CU1361*") or fnmatch.fnmatch(Patientfoldername, "*CU1370*"):
        ModalityName = [ 'FLAIR', 'SWI', 'T1_', 'T1Gd', 'T2']
    else:
        ModalityName = ['ADC', 'FLAIR', 'SWI', 'T1_', 'T1Gd', 'T2']
    for Modality in ModalityName:
        if Modality=='T1':
            Modality=Modality+'_'
        Datafile = String_f(os.listdir(Patient_folder_path), '*' + Modality + '*' + 'Normalized' + '*')
        ROIfile = String_f(os.listdir(Patient_folder_path), 'BIG_ROI.nii.gz')
        #if not ROIfile:
         #   continue
        ImagefileName =Patient_folder_path + '\\' + Datafile
        ROI_name = Patient_folder_path + '\\' + ROIfile

        img3ArrayBrain = Read3DImage(ImagefileName, Dataform="BrainNormalized")
        img3ArrayRaw = Read3DImage(ImagefileName, Dataform="NoNormalized")

        ROI3Array = Read3DImage(ROI_name, "NoNormalized")
        print(np.shape(ROI3Array))

        for slicenum_temp in range(np.shape(ROI3Array)[2]):
            slicenum = slicenum_temp + 1

            dicomImageRaw = Read2DImage(img3ArrayRaw, slicenum)
            dicomImageBrain = Read2DImage(img3ArrayBrain, slicenum)

            #ROIImage = Read2DImageROI(ROI3Array, slicenum)
            ROIImage = Read2DImage(ROI3Array, slicenum)
            genTextures(Patientfoldername, Datafile, slicenum)



