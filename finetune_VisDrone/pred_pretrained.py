from ultralytics import YOLO

# Pretrained model test
model = YOLO('yolov8x.pt')

# 'person', 'bicycle', 'car'
# model.predict('videos/Cyclist and vehicle Tracking - 1.mp4', save = True, classes = [0, 1, 2], project = 'pred', name = 'pretrained')
# model.predict('videos/Cyclist and vehicle tracking - 2.mp4', save = True, classes = [0, 1, 2], project = 'pred', name = 'pretrained')
# model.predict('videos/Drone Tracking Video.mp4', save = True, classes = [0, 1, 2], project = 'pred', name = 'pretrained')

model.predict('test_video_2.mp4', save = True, classes = [0, 1, 2], project = 'pred', name = 'test_vid_2_pretrained')
