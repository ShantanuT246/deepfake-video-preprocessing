import cv2 as cv
import os
from matplotlib import pyplot as plt

def extract_frames(video_path, output_path, interval = 10):
    video = cv.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)
    count = 0
    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break
        cv.imshow(f"{video_path[7:]}",frame)
        if count % interval == 0:
            filename = os.path.join(output_path, f"{video_path[7:-5]} frame => {count}.jpg")
            cv.imwrite(filename, frame)
        count += 1
    video.release()

# extract_frames("videos/01_03__hugging_happy__ISF9SP4G.mp4", "frames/01_03__hugging_happy__ISF9SP4G", 30)
# extract_frames("videos/02_09__kitchen_pan__HIH8YA82.mp4", "frames/02_09__kitchen_pan__HIH8YA82", 40)
# extract_frames("videos/02_13__outside_talking_still_laughing__CP5HFV3K.mp4", "frames/02_13__outside_talking_still_laughing__CP5HFV3K", 20)
# extract_frames("videos/02_18__outside_talking_pan_laughing__OXMEEFUQ.mp4", "frames/02_18__outside_talking_pan_laughing__OXMEEFUQ", 50)
# extract_frames("videos/02_21__talking_angry_couch__Z0XHPQAR.mp4", "frames/02_21__talking_angry_couch__Z0XHPQAR", 60)

def detect_faces_and_crop(video_path, output_path):
    video = cv.VideoCapture(video_path)
    os.makedirs(output_path, exist_ok=True)
    haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    count = 1
    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break
        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_rectangle = haar_cascade.detectMultiScale(
            grey_frame,
            scaleFactor=1.05,
            minNeighbors=8,
            minSize=(95, 95)
        )
        
        for x, y, w, h in face_rectangle:
            cropped_face_frame = frame[y:y+h, x:x+w]
            filename = os.path.join(output_path, f"{video_path[7:-5]} frame => {count}.jpg")
            cv.imwrite(filename, cropped_face_frame)
            count += 1
    video.release()
    
# detect_faces_and_crop("videos/01_03__hugging_happy__ISF9SP4G.mp4", "faces/01_03__hugging_happy__ISF9SP4G")
# detect_faces_and_crop("videos/02_09__kitchen_pan__HIH8YA82.mp4", "faces/02_09__kitchen_pan__HIH8YA82")
# detect_faces_and_crop("videos/02_13__outside_talking_still_laughing__CP5HFV3K.mp4", "faces/02_13__outside_talking_still_laughing__CP5HFV3K")
# detect_faces_and_crop("videos/02_18__outside_talking_pan_laughing__OXMEEFUQ.mp4", "faces/02_18__outside_talking_pan_laughing__OXMEEFUQ")
# detect_faces_and_crop("videos/02_21__talking_angry_couch__Z0XHPQAR.mp4", "faces/02_21__talking_angry_couch__Z0XHPQAR") 

#Basic Comparison Example This shows how much the face changed between two frames.
def extract_temporal_faces(path1, path2):
    face1 = cv.imread(path1)
    face2 = cv.imread(path2)
    face1 = cv.resize(face1, (256, 256))
    face2 = cv.resize(face2, (256, 256))
    diff = cv.absdiff(face1, face2)
    cv.imshow("Diff", diff)
    cv.waitKey(0)

# extract_temporal_faces("faces/01_03__hugging_happy__ISF9SP4G/01_03__hugging_happy__ISF9SP4 frame => 525.jpg", "faces/01_03__hugging_happy__ISF9SP4G/01_03__hugging_happy__ISF9SP4 frame => 535.jpg")

#Blurring (detect over-smoothing from deepfakes)
def bluring(img):
    blur = cv.imread(img)
    blur = cv.GaussianBlur(blur, (7,7), 5)
    cv.imshow("Blured", blur)
    cv.waitKey(0)

# bluring("faces/01_03__hugging_happy__ISF9SP4G/01_03__hugging_happy__ISF9SP4 frame => 505.jpg")

#Histogram analysis
def histogram(img):
    frame = cv.imread(img)
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv.calcHist([frame], [i], None, [256], [0,256])
        plt.plot(hist, color = col)
    plt.title("Color Histogram")
    plt.show()

# histogram("faces/02_13__outside_talking_still_laughing__CP5HFV3K/02_13__outside_talking_still_laughing__CP5HFV3 frame => 107.jpg")

# Canny edge detection which helps detect unnatural edges or artifacts.
def canny_edge_detection(img):
    frame = cv.imread(img)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200)
    cv.imshow("Canny Edges", edges)
    cv.waitKey(0)

# canny_edge_detection("faces/02_13__outside_talking_still_laughing__CP5HFV3K/02_13__outside_talking_still_laughing__CP5HFV3 frame => 10.jpg")

# Color Channel Splitting which can reveal irregularities in one color channel not visible in the full image.
def split_channels(img):
    frame = cv.imread(img)
    b, g, r = cv.split(frame)
    cv.imshow("Blue", b)
    cv.imshow("Green", g)
    cv.imshow("Red", r)
    cv.waitKey(0)

# split_channels("faces/02_13__outside_talking_still_laughing__CP5HFV3K/02_13__outside_talking_still_laughing__CP5HFV3 frame => 50.jpg")
