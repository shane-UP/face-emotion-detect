import cv2
import sys
from face_train import Model


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" %(sys.argv[0]))
        sys.exit(0)

    model = Model()
    model.load_model()

    color = (0, 255, 0)

    cap = cv2.VideoCapture(0)
    cascade_path = "D:\\open_cv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue

        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                image = frame[y-10:y+h+10, x-10:x+w+10]
                faceID = model.face_predict(image)

                if faceID == 0:
                    cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)

                    cv2.putText(frame, 'Shang', (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

                else:
                    pass

        cv2.imshow("人脸识别", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()