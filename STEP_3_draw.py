'''
STEP 3 part2
Draw all the kalman filtered targets onto the video.
'''

import pandas as pd
import numpy as np
import math
import cv2

video_path = 'video/pred_video1.mp4'
video_save_path = 'kalman_output/kalman_video1.mp4'

class_id_car = 3
class_name_car = 'car'
df_car = pd.read_csv('csv/video1_car_kalman.csv')
color_car = (255, 255, 0) # cyan

# Draw multiple kalman filtered targets onto video by pushing the targets into list
df_list = [df_car]
class_name_list = [class_name_car]
color_list = [color_car]

def convert_video_to_frame(path: str):
    print('Converting video file to frames...')
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    img = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img.append(frame)
        else:
            break

    cap.release()
    return img, fps

img, fps = convert_video_to_frame(video_path)

def draw_prediction(df_list, class_name_list, img, out, color_list):
    # Initialize tracking dots mask
    circle_mask = np.zeros_like(img[0])

    # For Each Frame
    for i in range(len(img)):
        tmp_img = img[i]

        # For Each Target Object In The Frame
        for idx, df in enumerate(df_list):
            # Get Box Position
            row = df.iloc[i]

            # Get Box Color
            # Box is set as white if the target is lost
            obj_color = color_list[idx]
            if row.confidence != -1:
                name = class_name_list[idx]
                color = obj_color
            else:
                name = 'lost'
                color = (255, 255, 255) # white

            # Get Tracking Dots Position
            x_cen, y_cen = row.x_cen, row.y_cen

            # Draw Box, Text, Tracking Dots
            cv2.rectangle(tmp_img, (int(row.xmin), int(row.ymin)),
                          (int(row.xmax), int(row.ymax)), color, 2)
            cv2.putText(tmp_img, name + " " + str(round(row.confidence, 2)),
                        (int(row.xmin) - 10, int(row.ymin) - 10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            circle_mask = cv2.circle(circle_mask, (math.floor(x_cen), math.floor(y_cen)),
                                     radius=1,
                                     color=color,
                                     thickness=3)

            tmp_img = cv2.add(tmp_img, circle_mask)

        out.write(tmp_img)

    return img



out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (img[0].shape[1], img[0].shape[0]))

draw_prediction(df_list, class_name_list, img, out, color_list)

out.release()