import numpy as np
import cv2
import Persons
import time
cnt_up   = 0
cnt_down = 0
count_up = 0
count_down = 0
state =0
back=None
cap = cv2.VideoCapture("test2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 5, (640,480))
for i in range(19):
    print(i, cap.get(i))
w = cap.get(3)
h = cap.get(4)
frameArea = h*w
areaTH = frameArea/300
print(('Area Threshold'), areaTH)
line_up = int(1*(h/6))
line_down   = int(4*(h/6))

up_limit =   int(.5*(h/6))
down_limit = int(4.5*(h/6))
line_down_color = (255,0,0)
line_up_color = (0,0,255)
pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
rect_co = []
max_p_age = 1
pid = 1
val = []

while(cap.isOpened()):    
    ret, frame = cap.read()
    if frame is not None and frame.shape[1] >= 20:
        frame=frame[:,20:]
    else:
        break
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
        if back is None:
            back=mask
            continue
        mask=cv2.absdiff(back,mask)
    except:
        print('EOF')
        print(('UP:'),cnt_up+count_up)
        print (('DOWN:'),cnt_down+count_down)
        break

    contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        rect = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)   #update coordinates in the object and resets age
                        if i.going_UP(line_down,line_up) == True:
                            if w > 100:
                                count_up = int(w/60)
                            else:    
                                cnt_up += 1;
                        elif i.going_DOWN(line_down,line_up) == True:
                            if w > 100:
                                count_down = int(w/60)
                            else:
                                cnt_down += 1;
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        index = persons.index(i)
                        persons.pop(index)
                        del i     #free the memory of i
                if new == True:
                    p = Persons.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1    
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)            
    str_up = 'UP: '+ str(cnt_up+count_up)
    str_down = 'DOWN: '+ str(cnt_down+count_down)
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv2.putText(frame, str_up ,(20,70),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(20,100),font,0.5,(255,255,255),2,cv2.LINE_AA)
    out.write(frame)
    cv2.imshow('Counting',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
