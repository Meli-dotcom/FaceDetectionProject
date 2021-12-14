import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Video/Vd_2.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    cv2.namedWindow("Video Processing", cv2.WINDOW_NORMAL)#Create window with freedom of dimension
    success, vid = cap.read()

    vidRGB = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(vidRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(vid, detection)
            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw, ic = vid.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(vid, bbox, (255, 0, 255), 2)
            cv2.putText(vid, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 255), 10)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(vid,f'FPS: {int(fps)}',(20,150), cv2.FONT_HERSHEY_PLAIN, 15, (0, 255, 0), 10)

    # results = pose.process(vidRGB)

    cv2.imshow("Video Processing", vid)
    cv2.waitKey(1)