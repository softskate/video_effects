import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

input_movie = cv2.VideoCapture("input.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
ret, frame = input_movie.read()
h,w,z = frame.shape
# h,w = h//2, w//2
output_movie = cv2.VideoWriter('output.mp4', fourcc, 29.97, (h,w))

frame_number = 0
max_retry = 5
while True:
    try:
        ret, image = input_movie.read()
        frame_number += 1
        # image = cv2.resize(image, (h,w), interpolation = cv2.INTER_AREA)

        pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

        if not ret:
            if max_retry>0:
                max_retry-=1
                print('retry', max_retry)
                continue
            else:
                print('finned', max_retry)
                break

        print("Writing frame {} / {}".format(frame_number, length))
        img = np.zeros((h,w,z), np.uint8)
        results = pose.process(image[:, :, ::-1])
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # cv2.imshow('MediaPipe Pose', img)
        output_movie.write(img)
        if cv2.waitKey(5) & 0xFF == 27:
            print('stopped', max_retry)
            break

    except Exception as e:
        print(e)
        
input_movie.release()
cv2.destroyAllWindows()