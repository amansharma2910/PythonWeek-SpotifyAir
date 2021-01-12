# ThePythonWeek-SpotifyAir
Hackathon submission for ThePythonWeek.

## Problem Statement
---

The problem this project addresses is technically a personal one. I listen to music a lot while programming. Often times, in order to change the music, I have to switch between screens, which often times breaks the flow of programming and concentration, and can be really frustrating. 

SpotifyAir, the solution uses Computer Vision to address this problem. Just leave the application running in the background, and you can then use your hand gestures (via your laptop's webcam) for music control actions like pause, play, next and previous. 

## Technologies used:
---

* PyTorch: For training the image classification model and inference.
* OpenCV: For reading the video feed from the device's webcam.
* Selenium: For automating browser
* VS Code (w/ Python Extension): For coding!

## Steps to run locally:
1. Clone the repository.

`$ git clone https://github.com/amansharma2910/PythonWeek-SpotifyAir.git`

2. Install project dependencies

`$ conda create --name spotify-air --file environment.txt`

3. Running the app:

`$ conda activate spotify-air`

`$ python spotify_air.py`

