import numpy
from mahotas.features.texture import haralick_labels

import GLCMFeatures
import ExtendGaborFeatures
import FirstOrderFeatures


def TextureExtraction(Rowcoord,Colcoord,dicomImageBrain,dicomImageRaw,Distances,Gaborkernel_bank):
    windowsize=5
    subImageBrain = dicomImageBrain[Rowcoord - 2:Rowcoord + 3, Colcoord - 2:Colcoord + 3]
    # Raw_mean = numpy.mean(subImageBrain)
    # Raw_std = numpy.std(subImageBrain)
    subImageRaw = dicomImageRaw[Rowcoord - 2:Rowcoord + 3, Colcoord - 2:Colcoord + 3]
    ###First order feature
    FirstOrderFeature=FirstOrderFeatures.calcFeatures(subImageRaw,windowsize)

    ###GLCM
    GLCMAngleList = ['Avg']
    GLCM = list()
    for Distance in Distances:
        glcmFeatures = GLCMFeatures.calcFeatures(subImageBrain,Distance)

        for GLCMAngle in GLCMAngleList:
            for featureName in haralick_labels[:-1]:
                GLCM.append(glcmFeatures[GLCMAngle][featureName])

    # use extended ROI
    Gabor = list()
    GaborFeatures = ExtendGaborFeatures.calcFeatures(dicomImageRaw, Rowcoord - 2, Colcoord - 2,
                                                     5, 5,
                                                     Gaborkernel_bank)

    for gaborfeature in GaborFeatures:
        for eachg in gaborfeature:
            Gabor.append(eachg)

    Texture_chain= GLCM + Gabor + FirstOrderFeature
    return Texture_chain
