'''
STEP 2
The model best.pt is trained on yolov8n.pt and finetuned with VisDrone Dataset
from the folder EXTRA_CREDITS.

This py script selects a target object, and stores the predicted box data into dataframe.
If the target is lost, its bbox and confidence are set as -1.
'''

import pandas as pd
import cv2
from ultralytics import YOLO

video_path = 'video/video1.mp4'
csv_path = 'csv/video1_car.csv'
class_id = 3 # 0: pedestrian, 1: people, 2: bycicle, 3: car

model = YOLO('best.pt')

# ----------------------------------------------------------------

# GET VIDEO FRAMES
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

# GET TRACED OBJECT FROM VIDEO FRAMES
results = model.predict(img)

# SELECT A TARGET AND TRACE THE BOX FOR EACH FRAME
def detections_to_dataframe(results, class_id):
    print("Detecting video frames...")
    box_data = [] # xmin, ymin, xmax, ymax, confidence, class
    frame_ids = []  # List to store frame IDs
    counter = 1
    xmin_last = 0

    # Get All Predicted Objects For Each Frame
    for frame_result in results:
        print(f"Detecting frame {counter}")
        found_class = False
        all_temp = []

        # Here we may have detected many boxes of different class.
        # But we only need one target each time.
        for box in frame_result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            cls_id = box.cls[0].item()
            conf = round(box.conf[0].item(), 2)
            temp = [xmin, ymin, xmax, ymax, conf, cls_id]

            if cls_id == class_id:
                all_temp.append(temp)
                found_class = True

        # If potential box found, choose the box nearist to the last box as target.
        # This can be done with IOU comparison as well.
        if found_class:
            min_diff = float('inf')
            closest_temp = None

            for temp in all_temp:
                diff = temp[0] - xmin_last
                if diff < min_diff:
                    min_diff = diff
                    if min_diff < 300: # change threshold if needed
                        closest_temp = temp

            if closest_temp is not None:
                xmin_last = closest_temp[0]
                box_data.append(closest_temp)
            else:
                box_data.append([-1, -1, -1, -1, -1, class_id])

            frame_ids.append(counter)

        # If target not found, set box data to -1
        else:
            box_data.append([-1,-1,-1,-1,-1,class_id])
            frame_ids.append(counter)

        counter += 1

    # Save Box To DataFrame
    df = pd.DataFrame(box_data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
    df.insert(0, 'frame_id', frame_ids)

    return df

df = detections_to_dataframe(results, class_id)
print(df)
df.to_csv(csv_path, index=False)
print(f"DataFrame saved to {csv_path}.")

