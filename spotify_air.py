import torch
import torch.nn as nn
import numpy
import cv2 as cv

# setting device as cpu to make it less computationally intensive
device = torch.device("cpu")

# model class
class GestureModel(nn.Module):
    """Neural net for recognizing hand gestures
    Input dims: m x 1 x 28 x 28
    Output dims: m x 4
    """
    def __init__(self):
        super().__init__()
        # input: m x 1 x 28 x 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # m x 16 x 28 x 28
        self.conv2 = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        ) # m x 32 x 14 x 14
        self.conv3 = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        ) # m x 64 x 6 x 6
        self.conv4 = nn.Sequential(
            # nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        ) # m x 128 x 2 x 2
        self.classifier = nn.Sequential(
            nn.Flatten(), # m x 128*2*2
            nn.Dropout(p=0.2),
            nn.Linear(128*2*2, 4),
            nn.Softmax(),
        ) # m x 4

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.classifier(x)
        return output

model = GestureModel()
model.load_state_dict(torch.load("hand_gesture_model.pth"))
model.to(device)

# def frameResize(frame, scale = 0.75):
#     height = int(frame.shape[0] * scale)
#     width = int(frame.shape[1] * scale)
#     dimensions = (width, height)

#     resizedFrame = cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)
#     return resizedFrame

video = cv.VideoCapture(0)
n = 10
while n > 0:
    isTrue, frame = video.read()
    print(frame.shape)
    n -= 1
    
