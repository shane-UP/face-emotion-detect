import cv2
import sys

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)               #视频来源: 0前置摄像头，1后置摄像头，2usb摄像头

    classfier = cv2.CascadeClassifier("D:\\open_cv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")

    color = (0, 255, 0)

    num = 2002
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, z, h = faceRect

                img_name = '%s/%d.jpg'%(path_name, num)
                image = frame[y-30: y+h+10, x-5: x+z+10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > catch_pic_num:
                    break

                cv2.rectangle(frame, (x-10, y-10), (x+z+10, y+h+10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d'%num, (x+30, y+30), font, 1, (255, 0, 255), 4)

        if num > catch_pic_num:
            break

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
        if len(sys.argv) != 2:
            print('please input only your name')
        else:
            print('./data/{:s}'.format(sys.argv[1]))
            CatchPICFromVideo("face_recognition", 0, 3001, './data/{:s}'.format(sys.argv[1]))
