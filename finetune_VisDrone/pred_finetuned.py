from ultralytics import YOLO

# Finetuned model test
model = YOLO('best.pt')

# model.predict('videos/Cyclist and vehicle Tracking - 1.mp4', save = True, classes = [1, 2, 3], project = 'pred', name = 'finetuned') # 'people', 'bicycle', 'car'
# model.predict('videos/Cyclist and vehicle tracking - 2.mp4', save = True, classes = [1, 2, 3], project = 'pred', name = 'finetuned')
# model.predict('videos/Drone Tracking Video.mp4', save = True, classes = [1, 2, 3], project = 'pred', name = 'finetuned')

model.predict('test_video_2.mp4', save = True, classes = [1, 2, 3], project = 'pred', name = 'test_vid_2_finetuned')

