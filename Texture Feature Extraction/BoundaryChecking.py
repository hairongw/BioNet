import csv
import xml.etree.ElementTree as ET
import fnmatch
import math

# process boundary points coordinate from XML file
def D3ParseXMLROI(xmlfile, slicenum):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    xcoordlist = list()
    ycoordlist = list()
    xycoordlist = list()
    for a1 in root.findall("./dict/array"):
        for a2 in a1.findall("dict"):  # slice
            count = len(list(a2.findall('integer')))
            if count==1:
                b3 = 'integer'
            else:
                b3 = 'integer[' + str(count - 1) + ']'
            for a4 in a2.findall(b3):  # revise in the future
                if a4.text == str(slicenum):
                    countArray = len(list(a2.findall('array')))
                    #b5 = 'array[' + str(countArray - 1) + ']/string'
                    b5='array'
                    for child in a2.findall(b5)[countArray - 1]:
                        #print(child.text)

                        xcoords = str(child.text).split(',')[0]
                        ycoords = str(child.text).split(',')[1]

                        if fnmatch.fnmatch(xcoords,'}'):
                            xc = float(xcoords.split('{')[1])
                            yc = float(ycoords.split('}')[0].replace(' ', ''))
                        else:
                            xc = float(xcoords.split('{')[1])
                            yc = float(ycoords.split('}')[0].replace(' ', ''))

                        #xc=256-xc

                        xcoordlist.append(xc)
                        ycoordlist.append(yc)

                        xycoordlist.append(list())
                        xycoordlist[len(xycoordlist) - 1].append(xc)
                        xycoordlist[len(xycoordlist) - 1].append(yc)

    if(len(xcoordlist))!=0:
        # Add the initial point
        xcoordlist.append(xcoordlist[0])
        ycoordlist.append(ycoordlist[0])

        # get x/y min/max in coords
        xmin = min(xcoordlist)
        ymin = min(ycoordlist)
        xmax = max(xcoordlist)
        ymax = max(ycoordlist)

        # ceil: get higher int
        # floor: get lower int
        xmin = int(math.floor(xmin))
        xmax = int(math.ceil(xmax))
        ymin = int(math.floor(ymin))
        ymax = int(math.ceil(ymax))
    else:
        xmin=[]
        xmax=[]
        ymin=[]
        ymax=[]

    return xmin, xmax, ymin, ymax, xycoordlist, xcoordlist, ycoordlist



# process boundary points coordinate from XML file
def NewParseXMLROI(xmlfile, slicenum):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    xcoordlist = list()
    ycoordlist = list()
    xycoordlist = list()
    for a1 in root.findall("./dict/array"):
        for a2 in a1.findall("dict"):  # slice
            for a3 in a2.findall("array/dict"):
                count = len(list(a3.findall('integer')))
                b3 = 'integer[' + str(count - 1) + ']'
                for a4 in a3.findall(b3):  # revise in the future
                    if a4.text == str(slicenum):
                        countArray = len(list(a3.findall('array')))
                        b5 = 'array[' + str(countArray - 1) + ']/string'
                        for child in a3.findall(b5):
                            #print(child.text)

                            xcoords = str(child.text).split(',')[0]
                            ycoords = str(child.text).split(',')[1]

                            if fnmatch.fnmatch(xcoords,'}'):
                                xc = float(xcoords.split('{')[1])
                                yc = float(ycoords.split('}')[0].replace(' ', ''))
                            else:
                                xc = float(xcoords.split('(')[1])
                                yc = float(ycoords.split(')')[0].replace(' ', ''))

                            xcoordlist.append(xc)
                            ycoordlist.append(yc)

                            xycoordlist.append(list())
                            xycoordlist[len(xycoordlist) - 1].append(xc)
                            xycoordlist[len(xycoordlist) - 1].append(yc)

    # Add the initial point
    xcoordlist.append(xcoordlist[0])
    ycoordlist.append(ycoordlist[0])

    # get x/y min/max in coords
    xmin = min(xcoordlist)
    ymin = min(ycoordlist)
    xmax = max(xcoordlist)
    ymax = max(ycoordlist)

    # ceil: get higher int
    # floor: get lower int
    xmin = int(math.floor(xmin))
    xmax = int(math.ceil(xmax))
    ymin = int(math.floor(ymin))
    ymax = int(math.ceil(ymax))

    return xmin, xmax, ymin, ymax, xycoordlist, xcoordlist, ycoordlist

# get sliding window coordinates from prediction file
def generatexyp(predfile):
    with open(predfile, 'r') as predictFile:
        predictFile.readline()

        rowFile = csv.reader(predictFile, delimiter=',')

        xybiopsylist = list()
        for row in rowFile:
            bio=int(row[3])
            if bio==1:
                xy=[int(row[0]),int(row[1])]
                xybiopsylist.append(xy)

    return xybiopsylist

# check if point is inside ROI boundary or outside boundary
def point_inside_polygon(x,y,poly):

    n = len(poly)
    inside =False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

# check if box covers part of boundary
def checkboxinout(testx, testy, xycoord):
    # check if box covers part of boundary
    b1 = point_inside_polygon(testx - 4, testy - 4, xycoord)
    b1h = point_inside_polygon(testx, testy - 4, xycoord)
    b2 = point_inside_polygon(testx - 4, testy + 4, xycoord)
    b2h = point_inside_polygon(testx - 4, testy, xycoord)
    b3 = point_inside_polygon(testx + 4, testy - 4, xycoord)
    b3h = point_inside_polygon(testx, testy + 4, xycoord)
    b4 = point_inside_polygon(testx + 4, testy + 4, xycoord)
    b4h = point_inside_polygon(testx + 4, testy, xycoord)

    if b1 != True or b1h != True or b2 != True or b2h != True or b3 != True or b3h != True or b4 != True or b4h != True:
        # False means one point of them is outside boundary, that means window is in boundary
        return False
    else:
        return True


# get points inside ROI
# biopsy: 3 conditions: 1. in boundary 2. off boundary but in rectangle box 3. off boundary and out of box
def T2chooseinoutcoord(contourx1,contourx2,contoury1,contoury2,T1xycoord,T2xycoord,xybiopsylist):
    windowptlist = list()
    # for each point inside rectangle plot, check if each point inside boundary or outside boundary, inside: True, outside: False
    for testx in range(contourx1,contourx2+1):
        for testy in range(contoury1,contoury2+1):
            # check if point is inside T2 boundary or not
            T2inorout = point_inside_polygon(testx, testy, T2xycoord)
            if T2inorout == True:

                # check if point is inside T1 boundary or not
                T1inorout = point_inside_polygon(testx, testy, T1xycoord)

                windowptlist.append(list())
                windowptlist[len(windowptlist)-1].append(testx)
                windowptlist[len(windowptlist)-1].append(testy)

                # False means in boundary, window is boundary window, 1: center pt of boundary window, 0: not boundary, inside boundary pt
                if checkboxinout(testx,testy,T2xycoord) == False:
                    windowptlist[len(windowptlist) - 1].append(1)
                else:
                    windowptlist[len(windowptlist) - 1].append(0)

                # 1: biopsy pt (inside), 0: not biopsy pt (inside)
                if [testx,testy] in xybiopsylist:
                    # print(1)
                    windowptlist[len(windowptlist) - 1].append(1)
                else:
                    windowptlist[len(windowptlist) - 1].append(0)

                # 1: this window is also T1 window, 0: not belong to T1
                if T1inorout == True:
                    windowptlist[len(windowptlist)-1].append(1)
                else:
                    windowptlist[len(windowptlist)-1].append(0)

    # if point is off boundary and in rectangle box, but in biopsy, then add this pt, and add 2: off boundary pt, and 1: biopsy pt, and 1: T1 window center pt
    for [testx, testy] in xybiopsylist:
        biopsyinorout = point_inside_polygon(testx, testy, T2xycoord)
        if biopsyinorout == False:
        # if testx<contourx1 or testx>contourx2 or testy<contoury1 or testy>contoury2:
        #    print([testx, testy])

            # print 'biopsy in rectangle box:',[testx,testy]

            # if out of boundary but in rectangle box

            T1inorout = point_inside_polygon(testx, testy, T1xycoord)

            windowptlist.append(list())
            windowptlist[len(windowptlist) - 1].append(testx)
            windowptlist[len(windowptlist) - 1].append(testy)

            windowptlist[len(windowptlist) - 1].append(2) # out of boundary
            windowptlist[len(windowptlist) - 1].append(1) # 1: biopsy pt (inside), 0: not biopsy pt (inside)

            # 1: this window is also T1 window, 0: not belong to T1
            if T1inorout == True:
                windowptlist[len(windowptlist) - 1].append(1)
            else:
                windowptlist[len(windowptlist) - 1].append(0)

    return windowptlist


# process boundary points coordinate from XML file
def ParseXMLROI(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    xcoordlist = list()
    ycoordlist = list()
    xycoordlist = list()
    for child in root.iter('string'):
        if not child.text:
            continue

        if not fnmatch.fnmatchcase(child.text,'*{*}*'):
            continue

        xcoords = str(child.text).split(',')[0]
        ycoords = str(child.text).split(',')[1]

        xc = float(xcoords.split('{')[1])
        yc = float(ycoords.split('}')[0].replace(' ',''))

        xcoordlist.append(xc)
        ycoordlist.append(yc)

        xycoordlist.append(list())
        xycoordlist[len(xycoordlist) - 1].append(xc)
        xycoordlist[len(xycoordlist) - 1].append(yc)

    # Add the initial point
    xcoordlist.append(xcoordlist[0])
    ycoordlist.append(ycoordlist[0])

    # get x/y min/max in coords
    xmin = min(xcoordlist)
    ymin = min(ycoordlist)
    xmax = max(xcoordlist)
    ymax = max(ycoordlist)

    # ceil: get higher int
    # floor: get lower int
    xmin = int(math.floor(xmin))
    xmax = int(math.ceil(xmax))
    ymin = int(math.floor(ymin))
    ymax = int(math.ceil(ymax))

    return xmin,xmax,ymin,ymax,xycoordlist,xcoordlist,ycoordlist
