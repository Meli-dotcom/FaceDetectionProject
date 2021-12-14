import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon


        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)


    def findFaces(self, vid, draw=True):
        vidRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(vidRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih,iw, ic = vid.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    vid = self.fancyDraw(vid, bbox)


                cv2.putText(vid, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 10)

        return vid, bboxs
    def fancyDraw(self, vid, bbox, l=30, t=10, rt = 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(vid, bbox, (255, 0, 255), rt)
        #Top Left x,y
        cv2.line(vid, (x,y), (x+l,y), (255, 0, 255), t)
        cv2.line(vid, (x,y), (x, y+l), (255, 0, 255), t)
        # Top Right x1,y
        cv2.line(vid, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(vid, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x,y1
        cv2.line(vid, (x1, y), (x + l, y1), (255, 0, 255), t)
        cv2.line(vid, (x1, y), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1,y1
        cv2.line(vid, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(vid, (x1, y1), (x, y1 - l), (255, 0, 255), t)


        return vid

def main():
    cap = cv2.VideoCapture("Video/Vd_2.mp4")
    pTime = 0
    detector = FaceDetector()

    while True:
        cv2.namedWindow("Video Processing", cv2.WINDOW_NORMAL)  # Create window with freedom of dimension
        success, vid = cap.read()
        vid, bboxs = detector.findFaces(vid)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(vid, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 15, (0, 255, 0), 10)

        # results = pose.process(vidRGB)

        cv2.imshow("Video Processing", vid)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()