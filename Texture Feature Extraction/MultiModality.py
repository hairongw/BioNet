import numpy


def rotate(matrix, degree):
    if abs(degree) not in [0, 90, 180, 270, 360]:
        print("raise error or just return nothing or original")
    if degree == 0:
        return matrix
    elif degree > 0:
        return rotate(zip(*matrix[::-1]), degree-90)
    else:
        return rotate(zip(*matrix)[::-1], degree+90)

def GrayScaleNormalization(imgArray):
    imgMax=imgArray.max()
    imgMin=imgArray.min()

    imgRange = imgMax - imgMin

    imgArray = (imgArray - imgMin) * (255.0 / imgRange)
    # transfer to closest int
    imgArray = numpy.rint(imgArray).astype(numpy.int16)

    return imgArray

def GrayScaleNormalizationK(imgArray,K,imgMin,imgMax):
    #imgMax=imgArray.max()
    #imgMin=imgArray.min()

    imgRange = imgMax - imgMin

    imgArray = (imgArray - imgMin) * (K / imgRange)
    # transfer to closest int
    imgArray = numpy.rint(imgArray).astype(numpy.int16)
    return imgArray
