from ultralytics import YOLO
import cv2
import cvzone as cz
import math 

cap  = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

model = YOLO('yolov8n.pt')

while True:
    success , img = cap.read()
    result = model(img,stream=True)
    
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

            # x1,y1,w,h = box.xywh[0]
            w,h = x2-x1,y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            cz.cornerRect(img,(x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cz.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            # print(x1,y1,x2,y2)
    cv2.imshow("image",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

