import cv2.cv2 as cv2


class FaceDetector:
    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def track_faces(self, image):
        rects = []
        face_rects = self.faceCascade.detectMultiScale(image,
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(30, 30),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        for (fX, fY, fH, fW) in face_rects:
            rects.append((fX, fY, fX + fW, fY + fH))
        return (face_rects, rects)
