import threading
import tkinter as tk

import PIL.Image, PIL.ImageTk
import cv2
import numpy as np
from tensorflow import keras

import Leap
from cvutils import getImageBorders, drawJointPosOnCanvas, getImageFixedHeight, image_resize
from leaputils import LeapCamType, convert_distortion_maps, undistort, getPixelLocation, unpackLeapVector, \
    getFingerJoints, getRawJointLocation
import mediapipe as mp

from mputils import normalizeLandmarksToPx, drawFromMpLandmarks, getMyLandmarkStyles

mp_hands = mp.solutions.hands


class LeapCapture:
    def __init__(self, width=400, height=400, leapCamType: LeapCamType = LeapCamType.RAW_IMG):
        self.width = width
        self.height = height

        self.ret: bool = False
        self.frame = None
        self.joint_data = None

        self.controller = Leap.Controller()
        self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

        ## leapMotion inits

        self.right_coeff = None
        self.right_coordinates = None
        self.maps_initialized = False

        self.cam_type = leapCamType

        # thread management
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def get_cam_type(self):
        return self.cam_type

    def process(self):
        while self.running:
            self.controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)

            leapFrame: Leap.Frame = self.controller.frame()
            rawMediaPipeCoords = []
            rawCoords = []

            if not leapFrame.images.is_empty:
                leftImage, mainImage = leapFrame.images[0], leapFrame.images[1]
                cropHandCanvas = np.zeros((400, 400), dtype=np.uint8)
                mpJointCanvas = np.ones((400, 400, 3), dtype=np.uint8)
                jointCanvas = np.zeros((400, 400, 3), dtype=np.uint8)
                if leftImage.is_valid:
                    if not self.maps_initialized:
                        self.right_coordinates, self.right_coeff = convert_distortion_maps(mainImage)
                        self.maps_initialized = True

                    undistorted_main = undistort(leftImage, self.right_coordinates, self.right_coeff, self.width,
                                                 self.height)

                    if self.cam_type == LeapCamType.RAW_IMG:
                        self.ret = True
                        self.frame = undistorted_main
                        self.joint_data = None
                    else:
                        if not leapFrame.hands.is_empty:
                            colorCoords = []
                            # get wrists pos
                            for hand in leapFrame.hands:
                                colorCoords.append(
                                    {"color": (0, 202, 255),
                                     "coords": getPixelLocation(hand.wrist_position, mainImage)})
                                colorCoords.append(
                                    {"color": (185, 190, 255),
                                     "coords": getPixelLocation(hand.palm_position, mainImage)})
                                rawCoords.append(
                                    {"pointID": "wrist", "position": unpackLeapVector(hand.wrist_position)})
                                rawCoords.append({"pointID": "palm", "position": unpackLeapVector(hand.palm_position)})
                                for finger in hand.fingers:
                                    colorCoords = colorCoords + getFingerJoints(finger, mainImage, withMetacarpal=True,
                                                                                withColors=True)
                                    rawCoords = rawCoords + getRawJointLocation(finger, withMetacarpal=True)

                            coords = [i["coords"] for i in colorCoords]
                            topY, bottomY, startX, endX = getImageBorders(coords)
                            if self.cam_type == LeapCamType.CROPPED_HAND:
                                # crop image based on coords
                                cropHandCanvas = undistorted_main[topY - 20: bottomY + 20, startX - 20: endX + 20]
                                self.ret = True
                                self.frame = cropHandCanvas
                                self.joint_data = None

                            elif self.cam_type == LeapCamType.JOINT_CANVAS:
                                # draw on image and crop
                                jointCanvas = drawJointPosOnCanvas(jointCanvas, colorCoords)
                                jointCanvas = jointCanvas[topY - 20: bottomY + 20, startX - 20: endX + 20]
                                self.ret = True
                                self.frame = jointCanvas
                                self.joint_data = rawCoords

                        else:
                            self.ret = True
                            self.joint_data = None
                            if self.cam_type == LeapCamType.CROPPED_HAND:
                                self.frame = cropHandCanvas
                            elif self.cam_type == LeapCamType.JOINT_CANVAS:
                                self.frame = jointCanvas

                        if self.cam_type == LeapCamType.MP_JOINTS:
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
                                        pxNormalizedCoords = normalizeLandmarksToPx(hand_landmarks, self.width,
                                                                                    self.height)
                                        mpJointCanvas = drawFromMpLandmarks(mpJointCanvas, pxNormalizedCoords,
                                                                            getMyLandmarkStyles())

                                        jointLocations = [i for i in pxNormalizedCoords.values()]
                                        topY, bottomY, startX, endX = getImageBorders(jointLocations)
                                        mpJointCanvas = mpJointCanvas[topY - 20: bottomY + 20, startX - 20: endX + 20]
                                        rawMediaPipeCoords.append(results.multi_hand_landmarks[0])
                            self.ret = True
                            self.frame = mpJointCanvas
                            self.joint_data = rawMediaPipeCoords

    def get_frame(self):
        return self.ret, self.frame, self.joint_data  # todo

    def __del__(self):
        self.stop_and_kill()

    def stop_and_kill(self):
        if self.running:
            self.running = False
            self.thread.join()

    def getType(self):
        return self.cam_type


class tkCamera(tk.Frame):
    def __init__(self, window, width=400, height=400, fps=60, row=0, column=0, vid=None):
        super().__init__(window)

        self.window = window

        if not vid:
            self.vid = LeapCapture(leapCamType=LeapCamType.RAW_IMG)
        else:
            self.vid = vid
        self.camType = self.vid.get_cam_type()
        self.frame = None
        self.image = None
        self.joint_data = None
        self.running = True
        self.photo = None
        self.update_frame()

        self.width = width
        self.height = height
        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.grid(row=row, column=column, sticky=tk.NSEW)

        self.fps = fps
        self.delay = int(1000 / self.fps)

        self.models = {
            '171cropped': keras.models.load_model('171cropped.h5'),
            '64cropped': keras.models.load_model('64cropped.h5'),
            '171leapjoints': keras.models.load_model('171leapjoints.h5'),
            '64leapjoints': keras.models.load_model('64leapjoints.h5'),
            '171mpjoints': keras.models.load_model('171mpjoints.h5'),
            '64mpjoints': keras.models.load_model('64mpjoints.h5'),

        }
        self.model171 = keras.models.load_model('171cropped.h5')
        self.model64 = keras.models.load_model('64cropped.h5')
        self.classes = ['A', 'B', 'E', 'G', 'H', 'I', 'L', 'P', 'R', 'V', 'W']

    def get_type(self):
        return self.camType

    def data_snapshot(self):
        if len(self.frame):
            return self.frame, self.joint_data

    #todo please refactor this function
    def predictFeed(self):
        if self.image:
            if self.camType == LeapCamType.CROPPED_HAND:
                open_cv_image = np.array(self.image)
                resized_image = cv2.resize(open_cv_image, (64, 64))
                resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 1))
                predIndex = np.argmax(self.models['64cropped'].predict(resized_image)[0])
                print(f"64_leap_cropped_cnn prediction:  {self.classes[predIndex]}")
                # resize to 64x64
                # predict with model1

                # pad to 171x171
                open_cv_image = getImageFixedHeight(open_cv_image, 171, 171, fillVal=0, isGrayscale=True)
                open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 1))
                predIndex = np.argmax(self.models['171cropped'].predict(open_cv_image)[0])
                print(f"171_leap_cropped_cnn prediction:  {self.classes[predIndex]}")
                # predict with model2
            elif self.camType == LeapCamType.JOINT_CANVAS:
                # resize to 64x64
                # predict with model3
                open_cv_image = np.array(self.image)
                resized_image = cv2.resize(open_cv_image, (64, 64))
                resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 3))
                predIndex = np.argmax(self.models['64leapjoints'].predict(resized_image)[0])
                print(f"64_leap_joints_cnn prediction:  {self.classes[predIndex]}")

                # pad to 171x171
                # predict with model4
                open_cv_image = getImageFixedHeight(open_cv_image, 171, 171, fillVal=(0,0,0), isGrayscale=False)
                open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 3))
                predIndex = np.argmax(self.models['171leapjoints'].predict(open_cv_image)[0])
                print(f"171_leap_joints_cnn prediction:  {self.classes[predIndex]}")
            elif self.camType == LeapCamType.MP_JOINTS:
                # resize to 64x64
                # predict with model5

                open_cv_image = np.array(self.image)
                resized_image = cv2.resize(open_cv_image, (64, 64))
                resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], 3))
                predIndex = np.argmax(self.models['64mpjoints'].predict(resized_image)[0])
                print(f"64_mp_joints_cnn prediction:  {self.classes[predIndex]}")

                # pad to 171x171
                # predict with model6
                open_cv_image = getImageFixedHeight(open_cv_image, 171, 171, fillVal=(0,0,0), isGrayscale=False)
                open_cv_image = open_cv_image.reshape((1, open_cv_image.shape[0], open_cv_image.shape[1], 3))
                predIndex = np.argmax(self.models['171mpjoints'].predict(open_cv_image)[0])
                print(f"171_mp_joints_cnn prediction:  {self.classes[predIndex]}")
        if self.joint_data:
            if self.camType == LeapCamType.JOINT_CANVAS:
                # normalize
                # calc angle matrix
                # calc distance matrix
                # predict with angle matrix
                # pred witn dist matrix
                pass
            elif self.camType == LeapCamType.MP_JOINTS:
                # add middle point
                # normalize
                # calc angle matrix
                # calc distance matrix
                # predict with angle matrix
                # pred witn dist matrix
                pass

    def update_frame(self):
        # try with one frame first
        ret, frames, joint_data = self.vid.get_frame()

        if ret and frames is not None:
            self.joint_data = joint_data
            self.frame = frames
            if self.camType == LeapCamType.CROPPED_HAND:
                paddedFrame = getImageFixedHeight(frames, 400, 400, fillVal=0, isGrayscale=True)
            else:
                paddedFrame = getImageFixedHeight(frames, 400, 400, fillVal=(0, 0, 0), isGrayscale=False)
            self.image = PIL.Image.fromarray(frames)
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)

            self.canvas.create_image(200, 200, image=self.photo, anchor='center')

        if self.running:
            self.window.after(int(1000 / 60), self.update_frame)

    def stop(self):
        self.running = False
        self.vid.stop_and_kill()

    def start(self):
        if not self.running:
            self.running = True
            self.update_frame()


class App:
    def __init__(self, root):
        self.root = root
        # load models
        self.initGui(root)
        self.leapJointCapture = LeapCapture(leapCamType=LeapCamType.JOINT_CANVAS)
        self.leapJointElement = tkCamera(self.root, row=0, column=0, vid=self.leapJointCapture)
        self.croppedHandCapture = LeapCapture(leapCamType=LeapCamType.CROPPED_HAND)
        self.croppedHandElement = tkCamera(self.root, row=0, column=1, vid=self.croppedHandCapture)
        self.mpJointsCapture = LeapCapture(leapCamType=LeapCamType.MP_JOINTS)
        self.mpJointsCaptureElement = tkCamera(self.root, row=0, column=2, vid=self.mpJointsCapture)
        self.camFeeds = [self.leapJointElement, self.croppedHandElement,
                         self.mpJointsCaptureElement]

        self.root.bind("<KeyPress>", self.onKeyPress)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def initGui(self, root: tk.Tk):
        for i in range(3):
            root.columnconfigure(i, weight=1)

    def snap_and_predict(self):
        print("\n====== It's predictin' time ====== ")
        for camFeed in self.camFeeds:
            camFeed.predictFeed()  # just send the message to threads, don't wait for anything

    def on_closing(self):
        for cam in self.camFeeds:
            cam.stop()
        self.root.destroy()

    def onKeyPress(self, event: tk.Event):
        if event.keysym in ['Escape', 'space', 'Return']:
            if event.keysym == 'Escape':
                for cam in self.camFeeds:
                    cam.stop()
                self.root.destroy()
                self.root.quit()

            if event.keysym in ['space', 'Return']:
                self.snap_and_predict()


if __name__ == "__main__":
    main = tk.Tk()
    main.configure(background='black')
    app = App(main)
