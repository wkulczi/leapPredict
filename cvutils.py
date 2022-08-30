import numpy as np
from operator import itemgetter
import cv2


def getImageFixedHeight(oldImg, newHeight, newWidth, fillVal=0, isGrayscale=True):
    if isGrayscale:
        oldHeight, oldWidth = oldImg.shape
        result = np.full((newHeight, newWidth), fillVal, dtype=np.uint8)
    else:
        oldHeight, oldWidth, oldDepth = oldImg.shape
        result = np.full((newHeight, newWidth, oldDepth), fillVal, dtype=np.uint8)

    xCenter = (newWidth - oldWidth) // 2
    yCenter = (newHeight - oldHeight) // 2

    result[abs(yCenter): yCenter + oldHeight, abs(xCenter):xCenter + oldWidth] = oldImg

    return result


def getImageBorders(coordsList):
    topY = min(coordsList, key=itemgetter(1))[1]
    bottomY = max(coordsList, key=itemgetter(1))[1]
    startX = min(coordsList, key=itemgetter(0))[0]
    endX = max(coordsList, key=itemgetter(0))[0]
    return topY, bottomY, startX, endX


# todo rename to drawPointOnCanvas
def drawJointPosOnCanvas(jointCanvas, colorCoords):
    for entry in colorCoords:
        jointCanvas = cv2.circle(jointCanvas, (entry["coords"]),
                                 radius=2, color=entry["color"], thickness=-2)
    return jointCanvas


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized