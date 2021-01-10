import torch
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
import torch.nn as nn
import numpy
import cv2 as cv
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

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
            nn.Softmax(dim=1),
        ) # m x 4

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.classifier(x)
        return output

# creating webdriver
browser = webdriver.Chrome()
browser.get("https://open.spotify.com/")

# creating a sleep timer to allow initializing the browser
time.sleep(4)

# finds login button
login_button = browser.find_elements_by_class_name("_3f37264be67c8f40fa9f76449afdb4bd-scss _1f2f8feb807c94d2a0a7737b433e19a8-scss")
#clicking the login button
login_button[0].click()

time.sleep(2)

username = browser.find_elements_by_id("login-username") 
username[0].send_keys('USER-NAME') # enter username here

password = browser.find_elements_by_id("login-password")
password[0].send_keys('PASSWORD') # enter hardcoded password here

# finds login button
login = browser.find_elements_by_id("login-button")
#clicking the login button
login[0].click()

time.sleep(3)


model = GestureModel() # initialising model object
model.load_state_dict(torch.load("hand_gesture_model.pth")) # loading the pretrained weights
model.to(device) # moving the model to CPU for less computation
model.eval() # setting the model to evaluation mode


# defining the set of transformations
transform = Compose([
    ToTensor(),
    Grayscale(),
    Resize([28, 28])
])

# runnig cv to capture video frames and perform inference on it
video = cv.VideoCapture(0)
while True:
    isTrue, frame = video.read()
    frame = torch.unsqueeze(transform(frame), 0)
    print(model(frame))
    break
    
