from flask import Response
from flask import Flask
from flask import render_template
from imutils.video import VideoStream
import threading
import argparse
import datetime
import imutils
import time
import cv2.cv2 as cv2
import tellopy
import numpy as np
import av
from facedetector import FaceDetector
from numpy import linalg

# initialize a flask framework
app = Flask(__name__)

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize the video stream and allow the camera sensor to start
prev_flight_data = ""
drone = tellopy.Tello()
drone.set_loglevel(drone.LOG_ERROR)
drone.connect()
drone.wait_for_connection(60.0)
container = av.open(drone.get_video_stream())
fd = FaceDetector("trained_data/haarcascade_frontalface_default.xml")


@app.route("/")
def hello():
    return "Hello, World!"


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def process_frame(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (faceRects, face_areas) = fd.track_faces(image_bw)
    (h, w) = image.shape[:2]
    # These are our center dimensions
    cWidth = int(w/2)
    cHeight = int(h/2)
    cv2.circle(image, (cWidth, cHeight), 10, (0, 255, 0), 1)

    ind = 1
    for face_area in face_areas:
        # color all rectangles in green
        cv2.rectangle(
            image, (face_area[0], face_area[1]), (face_area[2], face_area[3]), (0, 255, 0), 2)
        ind = ind+1

    if(ind > 1):  # if found at least one rectangle
        # color the target rectangle in red
        print("I found: " + str(len(faceRects)) + " faces")

    return image


def cv_video():
    # grab global references to the video stream, output frame, and lock variables
    global outputFrame, lock
    # skip first frames
    frame_skip = 300
    while True:
        for frame in container.decode(video=0):
            if 0 < frame_skip:
                frame_skip = frame_skip - 1
                continue
            start_time = time.time()
            image = cv2.cvtColor(
                np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            image = imutils.resize(image, width=500)
            image = print_onscreen_instructions(image)
            image = process_frame(image)
            outputFrame = image.copy()
            if frame.time_base < 1.0/60:
                time_base = 1.0/60
            else:
                time_base = frame.time_base
            frame_skip = int((time.time() - start_time)/time_base)


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


def flightDataHandler(event, sender, data):
    global prev_flight_data
    text = str(data)
    if prev_flight_data != text:
        prev_flight_data = text


def print_onscreen_instructions(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, prev_flight_data, (int(
        img.shape[1]/5), img.shape[0]-15), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    return img


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    drone.subscribe(drone.EVENT_FLIGHT_DATA, flightDataHandler)
    # start a thread that will perform motion detection
    t = threading.Thread(target=cv_video)
    t.daemon = True
    t.start()
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
