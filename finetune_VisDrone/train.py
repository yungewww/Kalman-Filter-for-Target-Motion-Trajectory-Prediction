from ultralytics import YOLO

import torch

torch.cuda.set_device(0)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('yolov8n.pt')
model.to(device='cuda')

if __name__ == '__main__':
    model.train(data = 'VisDrone.yaml', epochs = 100)
    model.val()