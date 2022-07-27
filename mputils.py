import math

import mediapipe as mp
from typing import Mapping, Tuple, Dict
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def getMyLandmarkStyles() -> Mapping[int, mp_drawing_styles.DrawingSpec]:
    thickness = -1
    radius = 2

    return {
        mp_hands.HandLandmark.WRIST: mp_drawing_styles.DrawingSpec((0, 202, 255), thickness, radius),
        mp_hands.HandLandmark.THUMB_CMC: mp_drawing_styles.DrawingSpec((255, 255, 255), thickness, radius),
        mp_hands.HandLandmark.THUMB_MCP: mp_drawing_styles.DrawingSpec((36, 115, 0), thickness, radius),
        mp_hands.HandLandmark.THUMB_IP: mp_drawing_styles.DrawingSpec((81, 255, 0), thickness, radius),
        mp_hands.HandLandmark.THUMB_TIP: mp_drawing_styles.DrawingSpec((216, 255, 198), thickness, radius),
        mp_hands.HandLandmark.INDEX_FINGER_MCP: mp_drawing_styles.DrawingSpec((140, 22, 6), thickness, radius),
        mp_hands.HandLandmark.INDEX_FINGER_PIP: mp_drawing_styles.DrawingSpec((255, 57, 0), thickness, radius),
        mp_hands.HandLandmark.INDEX_FINGER_DIP: mp_drawing_styles.DrawingSpec((181, 114, 105), thickness, radius),
        mp_hands.HandLandmark.INDEX_FINGER_TIP: mp_drawing_styles.DrawingSpec((255, 234, 107), thickness, radius),
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP: mp_drawing_styles.DrawingSpec((65, 1, 90), thickness, radius),
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP: mp_drawing_styles.DrawingSpec((90, 48, 79), thickness, radius),
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP: mp_drawing_styles.DrawingSpec((255, 99, 214), thickness, radius),
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP: mp_drawing_styles.DrawingSpec((235, 181, 255), thickness, radius),
        mp_hands.HandLandmark.RING_FINGER_MCP: mp_drawing_styles.DrawingSpec((152, 147, 151), thickness, radius),
        mp_hands.HandLandmark.RING_FINGER_PIP: mp_drawing_styles.DrawingSpec((67, 73, 144), thickness, radius),
        mp_hands.HandLandmark.RING_FINGER_DIP: mp_drawing_styles.DrawingSpec((0, 9, 132), thickness, radius),
        mp_hands.HandLandmark.RING_FINGER_TIP: mp_drawing_styles.DrawingSpec((0, 18, 255), thickness, radius),
        mp_hands.HandLandmark.PINKY_MCP: mp_drawing_styles.DrawingSpec((0, 65, 107), thickness, radius),
        mp_hands.HandLandmark.PINKY_PIP: mp_drawing_styles.DrawingSpec((65, 113, 144), thickness, radius),
        mp_hands.HandLandmark.PINKY_DIP: mp_drawing_styles.DrawingSpec((0, 155, 255), thickness, radius),
        mp_hands.HandLandmark.PINKY_TIP: mp_drawing_styles.DrawingSpec((149, 206, 243), thickness, radius),
    }


def normalizeToPx(normalizedX: float, normalizedY: float, imageWidth: int, imageHeight: int) -> Tuple[int, int]:
    x_px = min(math.floor(normalizedX * imageWidth), imageWidth - 1)
    y_px = min(math.floor(normalizedY * imageHeight), imageHeight - 1)
    return x_px, y_px


def normalizeLandmarksToPx(hand_landmarks, imageWidth, imageHeight) -> Dict[int, Tuple[int, int]]:
    idx_to_coords = {}
    for idx, landmark in enumerate(hand_landmarks.landmark):
        idx_to_coords[idx] = normalizeToPx(landmark.x, landmark.y, imageWidth, imageHeight)

    return idx_to_coords


def drawFromMpLandmarks(mpJointCanvas: np.ndarray, pxNormalizedCoords: Dict[int, Tuple[int, int]],
                        landmarkStyles: Mapping[int, mp_drawing_styles.DrawingSpec]) -> np.ndarray:
    for idx, landmarkPx in pxNormalizedCoords.items():
        mpJointCanvas = cv2.circle(mpJointCanvas, landmarkPx, landmarkStyles[idx].circle_radius,
                                   landmarkStyles[idx].color, landmarkStyles[idx].thickness)
    return mpJointCanvas
