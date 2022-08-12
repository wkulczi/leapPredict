import cv2
import Leap
from cvutils import getImageBorders, drawJointPosOnCanvas, getImageFixedHeight
from leaputils import convert_distortion_maps, undistort, getPixelLocation, unpackLeapVector, getFingerJoints, \
    getRawJointLocation
import numpy as np
import mediapipe as mp

from tensorflow import keras

from mputils import normalizeLandmarksToPx, drawFromMpLandmarks, getMyLandmarkStyles
from collections import Counter

controller = Leap.Controller()
controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

maps_initialized = False
right_coords, right_coeff = None, None

mp_hands = mp.solutions.hands


models = {
            '171cropped': keras.models.load_model('171cropped.h5'),
            '64cropped': keras.models.load_model('64cropped.h5'),
            '171leapjoints': keras.models.load_model('171leapjoints.h5'),
            '64leapjoints': keras.models.load_model('64leapjoints.h5'),
            '171mpjoints': keras.models.load_model('171mpjoints.h5'),
            '64mpjoints': keras.models.load_model('64mpjoints.h5'),

        }
classes = ['A', 'B', 'E', 'G', 'H', 'I', 'L', 'P', 'R', 'V', 'W']

def predictFromImages(leaphand, leapjoints, mpjoints):
    preds= []
    resized_image = cv2.resize(leaphand, (64, 64))
    resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 1))
    predIndex = np.argmax(models['64cropped'].predict(resized_image)[0])
    preds.append(classes[predIndex])

    open_cv_image = getImageFixedHeight(leaphand, 171, 171, fillVal=0, isGrayscale=True)
    open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 1))
    predIndex = np.argmax(models['171cropped'].predict(open_cv_image)[0])
    preds.append(classes[predIndex])

    resized_image = cv2.resize(leapjoints, (64, 64))
    resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 3))
    predIndex = np.argmax(models['64leapjoints'].predict(resized_image)[0])
    preds.append(classes[predIndex])

    open_cv_image = getImageFixedHeight(leapjoints, 171, 171, fillVal=(0, 0, 0), isGrayscale=False)
    open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 3))
    predIndex = np.argmax(models['171leapjoints'].predict(open_cv_image)[0])
    preds.append(classes[predIndex])

    if mpjoints:
        resized_image = cv2.resize(mpjoints, (64, 64))
        resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 3))
        predIndex = np.argmax(models['64mpjoints'].predict(resized_image)[0])
        preds.append(classes[predIndex])

        open_cv_image = getImageFixedHeight(mpjoints, 171, 171, fillVal=(0, 0, 0), isGrayscale=False)
        open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 3))
        predIndex = np.argmax(models['171mpjoints'].predict(open_cv_image)[0])
        preds.append(classes[predIndex])

    votes = ",".join(["{}:{}".format(key, val) for key,val in dict(Counter(preds)).items()])
    return votes

while True:
    frame = controller.frame()
    rawCoords = []
    if frame:
        if not frame.images.is_empty:
            leftImage, mainImage = frame.images[0], frame.images[1]

            if not maps_initialized:
                right_coords, right_coeff = convert_distortion_maps(mainImage)
                maps_initialized = True

            undistorted_main = undistort(leftImage, right_coords, right_coeff, 400, 400)

            if not frame.hands.is_empty:
                colorCoords = []
                # get wrists pos
                for hand in frame.hands:
                    colorCoords.append(
                        {"color": (0, 202, 255),
                         "coords": getPixelLocation(hand.wrist_position, mainImage, 400, 400)})
                    colorCoords.append(
                        {"color": (185, 190, 255),
                         "coords": getPixelLocation(hand.palm_position, mainImage, 400, 400)})
                    rawCoords.append(
                        {"pointID": "wrist", "position": unpackLeapVector(hand.wrist_position)})
                    rawCoords.append({"pointID": "palm", "position": unpackLeapVector(hand.palm_position)})
                    for finger in hand.fingers:
                        colorCoords = colorCoords + getFingerJoints(finger, mainImage, withMetacarpal=True,
                                                                    withColors=True, targetWidth=400, targetHeight=400)
                        rawCoords = rawCoords + getRawJointLocation(finger, withMetacarpal=True)

                coords = [i["coords"] for i in colorCoords]
                topY, bottomY, startX, endX = getImageBorders(coords)

                #cropped unpadded hand photo
                cropHandCanvas = undistorted_main[topY - 20: bottomY + 20, startX - 20: endX + 20]

                #cropped unpadded leap joints
                jointCanvas = np.zeros((400, 400, 3), dtype=np.uint8)
                jointCanvas = drawJointPosOnCanvas(jointCanvas, colorCoords)
                jointCanvas = jointCanvas[topY - 20: bottomY + 20, startX - 20: endX + 20]

                #mediapipe cropped joint canvas
                mpJointCanvas = np.ones((400, 400, 3), dtype=np.uint8)
                with mp_hands.Hands(
                        model_complexity=0,
                        max_num_hands=1,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.5
                ) as hands:
                    undistorted_main = cv2.cvtColor(undistorted_main, cv2.COLOR_GRAY2RGB)
                    results = hands.process(undistorted_main)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            pxNormalizedCoords = normalizeLandmarksToPx(hand_landmarks, 400, 400)
                            mpJointCanvas = drawFromMpLandmarks(mpJointCanvas, pxNormalizedCoords,
                                                                getMyLandmarkStyles())

                            jointLocations = [i for i in pxNormalizedCoords.values()]
                            topY, bottomY, startX, endX = getImageBorders(jointLocations)
                            mpJointCanvas = mpJointCanvas[topY - 20: bottomY + 20, startX - 20: endX + 20]

                votes = predictFromImages(cropHandCanvas, jointCanvas, None)

                cv2.rectangle(undistorted_main, (startX-10, topY-20), (endX+20, bottomY+20), (36, 255, 12), 1)
                cv2.putText(undistorted_main, votes, (startX, topY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow('frame', undistorted_main)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
