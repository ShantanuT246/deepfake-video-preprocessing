import cv2 as cv
import os

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