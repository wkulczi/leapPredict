import enum

import numpy as np
import cv2
import ctypes
import Leap


class LeapCamType(enum.IntEnum):
    RAW_IMG = 0
    CROPPED_HAND = 1
    JOINT_CANVAS = 2
    MP_JOINTS = 3

class ImportDataType(enum.IntEnum):
    CROPPED_HAND = 0
    JOINT_CANVAS = 1
    MP_JOINT_CANVAS = 2
    LEAP_JOINTS = 3
    MP_JOINTS = 4


def convert_distortion_maps(image):
    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length // 2, dtype=np.float32)
    ymap = np.zeros(distortion_length // 2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length // 2 - i // 2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length // 2 - i // 2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width // 2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width // 2))

    # resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    # Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation=False)

    return coordinate_map, interpolation_coefficients


def getDistortedImg(image):
    # wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    return img


def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype=np.ubyte)

    img = getDistortedImg(image)

    # remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation=cv2.INTER_LINEAR)

    # resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination


# https://developer-archive.leapmotion.com/documentation/java/devguide/Leap_Images.html?proglang=java#draw-tracking-data-over-image
def getPixelLocation(positionVector: Leap.Vector, currentFrameImage: Leap.Image, targetWidth=400,
                     targetHeight=400):
    cameraOffset = 20

    # hSlope = -(fingerTip.x + cameraOffset * (2 * image iter - 1))/fingerTip.y
    hSlope = (positionVector.x + cameraOffset * (2 * 0 - 1)) / positionVector.y
    vSlope = positionVector.z / positionVector.y

    ray = Leap.Vector(hSlope * currentFrameImage.ray_scale_x + currentFrameImage.ray_offset_x,
                      vSlope * currentFrameImage.ray_scale_y + currentFrameImage.ray_offset_y,
                      0)

    # Vector(ray.getX() * targetWidth, ray.getY() * targetHeight, 0);
    return int(ray.x * targetWidth), int(ray.y * targetHeight)


def getFingerBones(finger: Leap.Finger, withMetacarpal=False):
    fingerBones = [
        finger.bone(Leap.Bone.TYPE_PROXIMAL),
        finger.bone(Leap.Bone.TYPE_INTERMEDIATE),
        finger.bone(Leap.Bone.TYPE_DISTAL)
    ]
    if withMetacarpal:
        fingerBones.append(finger.bone(Leap.Bone.TYPE_METACARPAL))
    return fingerBones


def getFingerJoints(finger: Leap.Finger, mainImage, withMetacarpal=False, withColors=False):
    boneEnds = []
    fingerBones = getFingerBones(finger, withMetacarpal=withMetacarpal)
    for bone in fingerBones:
        if withColors:
            boneEnds.append(
                {"color": getFingerJointColor(finger, bone), "coords": getPixelLocation(bone.next_joint, mainImage)})
        else:
            boneEnds.append(getPixelLocation(bone.next_joint, mainImage))
    return boneEnds


def getRawJointLocation(finger: Leap.Finger, withMetacarpal=False):
    boneEnds = []
    fingerBones = getFingerBones(finger, withMetacarpal=withMetacarpal)
    for bone in fingerBones:
        boneEnds.append({"pointID": str(finger.type) + str(bone.type), "coords": unpackLeapVector(bone.next_joint)})
    return boneEnds


def unpackLeapVector(locationVector: Leap.Vector):
    return locationVector.x, locationVector.y, locationVector.z


def getFingerJointColor(finger: Leap.Finger, bone: Leap.Bone):
    colorMat = [
        # [metacarpal, proximal, intermediate, distal],
        [(255,255,255), (36, 115, 0), (81, 255, 0), (216, 255, 198)],  # thumb
        [(140, 22, 6), (255, 57, 0), (181, 114, 105), (255, 234, 107)],  # index
        [(65, 1, 90), (90, 48, 79), (255, 99, 214), (235, 181, 255)],  # middle
        [(152, 147, 151), (67, 73, 144), (0, 9, 132), (0, 18, 255)],  # ring
        [(0, 65, 107), (65, 113, 144), (0, 155, 255), (149, 206, 243)],  # pinky
    ]

    return colorMat[finger.type][bone.type]
