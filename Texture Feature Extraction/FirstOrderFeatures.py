from radiomics import firstorder
import six
import SimpleITK as sitk
import numpy
from numpy import newaxis
import fnmatch

def calcFeatures(img,windowsize):
    #####Mean and stand
    FirstOrderFeature = list()
    Raw_mean = numpy.mean(img)
    Raw_std = numpy.std(img)
    FirstOrderFeature.append(Raw_mean)
    FirstOrderFeature.append(Raw_std)

    #####Added
    img3= img[:, :, newaxis]
    image = sitk.GetImageFromArray(img3)
    maskarray=numpy.ones((windowsize, windowsize, 1))
    mask = sitk.GetImageFromArray(maskarray)

    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask)
    firstOrderFeatures.enableAllFeatures()
    FirstOrderresults = firstOrderFeatures.execute()

    # for (key, val) in six.iteritems(FirstOrderresults):
    #     if fnmatch.fnmatch(key, 'Variance') or fnmatch.fnmatch(key, 'Mean'):
    #         continue
    #     print(key, val)
    #     FirstOrderFeature.append(val)

    FList = dict()
    for (key, val) in six.iteritems(FirstOrderresults):
        if fnmatch.fnmatch(key, 'Variance') or fnmatch.fnmatch(key, 'Mean'):
            continue
        FList[key] = val

    NamesT = FirstOrderFeatureNames()
    Names = NamesT[2:len(NamesT)]
    for Name in Names:
        val = FList[Name]
        FirstOrderFeature.append(val)

    return FirstOrderFeature

# Python 2
def FirstOrderFeatureNames():
    Name=['Raw_Mean','Raw_Std',
          'InterquartileRange',
         'Skewness',
         'Uniformity',
         'Median',
         'Energy',
         'RobustMeanAbsoluteDeviation',
         'MeanAbsoluteDeviation',
         'TotalEnergy',
         'Maximum',
         'RootMeanSquared',
         '90Percentile',
         'Minimum',
         'Entropy',
         'Range',
         #'Variance',
         '10Percentile',
         'Kurtosis']#,'Mean']
    return(Name)

# Name=list()
# for (key, val) in six.iteritems(FirstOrderresults):
#     Name.append(key)