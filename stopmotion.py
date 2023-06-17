import cv2
import mediapipe as mp
import math
import datetime
import numpy as np
def stopmotion():
    def similarity(before, now):
        sum=0
        k = np.subtract(now, before)
        for i in range(15):
            sum+=abs(k[0][i])
        return sum/15

    def taking(continuous):
        l=len(continuous)
        for i in range(10):
            if continuous[l-4-i]!=1:
                return False
        return True
    #initial value
    start=0
    check=0
    continuous=[]
    max_num_hands = 1
    gesture = {
        0:'bud', 1:'LEGO', 2:'Paper'
    }
    data=[[0]*15]
    data=np.array(data)
    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Gesture recognition model
    file = np.genfromtxt('data\canidentify.csv', delimiter=',')
    print(file)
    angle = file[:,:-1].astype(np.float32)
    print(angle)
    label = file[:, -1].astype(np.float32)
    print(label)
    knn = cv2.ml.KNearest_create()
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('SaveVideo.mp4',fourcc,20.0,(width, height))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        cv2.putText(img, str("Taken: ")+str(check-1), (0,100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,255,0))
        #video.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                # Inference gesture
                before=data
                data = np.array([angle], dtype=np.float32)
                #print(data[0][0])
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])
                if idx==0 or idx==1 : 
                    start+=1
                    continuous.append(1)
                else: 
                    start=0
                    continuous.append(0)
                print(idx)
                tmp=similarity(before, data)
                #print(tmp)

                cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        else:
            if start>0 and taking(continuous): 
                print("Success!"+str(check))
                check+=1
            start=0
        cv2.imshow('Game', img)
        out.write(img)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break
    return check