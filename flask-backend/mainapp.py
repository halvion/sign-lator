from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import cv2 as cv
import mediapipe as mp
import numpy as np

import copy
import csv
from collections import Counter, deque
import itertools
from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc
import base64
import argparse

from app import *

# app instance
app = Flask(__name__)
CORS(app)




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

# function to generate frames
def generate_frames():

    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2, # detects the number of hands in the program
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label2.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        number = -1
        mode = 0

        success, frame = cap.read()
        if not success:
            break
        else:
            # # Convert the frame to RGB
            # frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # results = hands.process(frame_rgb)

            # # Draw hand landmarks
            # if results.multi_hand_landmarks:
            #     for hand_landmarks in results.multi_hand_landmarks:
            #         mp.solutions.drawing_utils.draw_landmarks(
            #         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            frame = cv.flip(frame, 1)

            debug_frame = copy.deepcopy(frame)

            # Detection implementation
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_frame, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_frame, point_history)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    # Drawing part
                    debug_frame = draw_bounding_rect(use_brect, debug_frame, brect)
                    debug_frame = draw_landmarks(debug_frame, landmark_list)
                    debug_frame = draw_info_text(
                        debug_frame,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
            else:
                point_history.append([0, 0])

            debug_frame = draw_point_history(debug_frame, point_history)
            debug_frame = draw_info(debug_frame, fps, mode, number)

            ret, buffer = cv.imencode('.jpg', debug_frame)

            frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# route to return message
@app.route('/api/home', methods = ['GET'])
def return_home():
    return jsonify({'message': 'kanye west.'})

# route to return video feed
@app.route('/api/video_feed', methods = ['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(port=8000, debug=True)


# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#     landmark_array = np.empty((0, 2), int)
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         landmark_point = [np.array((landmark_x, landmark_y))]
#         landmark_array = np.append(landmark_array, landmark_point, axis=0)
#     x, y, w, h = cv.boundingRect(landmark_array)
#     return [x, y, x + w, y + h]

# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]
#     landmark_point = []
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         landmark_point.append([landmark_x, landmark_y])
#     return landmark_point

# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]
#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
#     temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
#     max_value = max(list(map(abs, temp_landmark_list)))
#     temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
#     return temp_landmark_list

# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]
#     temp_point_history = copy.deepcopy(point_history)
#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]
#         temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
#     temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
#     return temp_point_history
