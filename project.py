# ****************************** Final Changes*******************************************************


# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import glob
import PySimpleGUI as sg

# lang_list=['Devanagari','English']
lang_list=['Bengali','Devanagari','English','Gujarati','Gurumukhi','Kannada','Malayalam','Manipuri','Oriya','Tamil','Telugu','urdu']

digit_list = [0,1,2,3,4,5,6,7,8,9]
layout = [
    [sg.Text('Language'),sg.Combo(lang_list,default_value='English',key='-lang_name-')],
    [sg.Text('Digit'),sg.Combo(digit_list,default_value=0, key='-digit_value-')],
	[sg.Image(key = '-IMAGE-'),sg.Image(key = '-IMAGE2-')],
	[sg.Text('Draw the number', key = '-TEXT-', expand_x = True, justification = 'R')]
]
window = sg.Window('Air written character dataset creation', layout)
def process(image):
    cv2.imwrite('temp.jpg',image)
    image = cv2.imread('temp.jpg')
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('mask2',mask)

    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.filter2D(mask, -1, kernel)


    kernel1 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bilateralFilter(mask, 7, 75, 75)


    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]    


    cv2.imwrite('c.jpg',image)

    if len(contours) > 0:
        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(image, (x,y), (x+w+10,y+h), (255, 0, 0), 2)
    else:
        return 0
    
    x1 = int(x)
    y1 = int(y)
    x2 = int(x+w)
    y2 = int(y+h)


    image = mask[y1:y2,x1:x2]
    image2 = cv2.resize(image,(256,256))
    imgbytes = cv2.imencode('.png',image2)[1].tobytes()
    window['-IMAGE2-'].update(data = imgbytes)
    window['-TEXT-'].update('Image Saved')
    cv2.imwrite('out.jpg',image)
    image = cv2.resize(image, (28,28))
    return image
   
   
def clear():
    global bpoints
    global gpoints
    global rpoints
    global ypoints
    
    global blue_index
    global green_index
    global red_index
    global yellow_index
    
    bpoints = [deque(maxlen=512)]
    gpoints = [deque(maxlen=512)]
    rpoints = [deque(maxlen=512)]
    ypoints = [deque(maxlen=512)]

    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    paintWindow[:,:,:] = 255
    
    
    
    
# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

flag = 0
flag2 =0
diff_point= 60



folder_name =''
digit=0
def set_fname_digit():
    global folder_name
    global digit
    global count
    folder_name = values['-lang_name-']
    folder_name = 'dataset\\'+folder_name
    if not os.path.exists(folder_name):
       os.makedirs(folder_name)
       
    digit = str(values['-digit_value-'])
    list_of_files = glob.glob(folder_name+'/'+digit+'*.jpg') # * means all if need specific format then *.csv
    if (len(list_of_files)>0):
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = latest_file.split('\\')[-1].split('.')[0]
        count = int(latest_file[1:]) +1
    else:
        count = 1


# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

#The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 2

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255


#cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam
cap = cv2.VideoCapture(0)
print("Camera ON")

previous_lang=''
previous_digit=''

ret = True
while ret:
    event, values = window.read(timeout = 0)
    if event == sg.WIN_CLOSED:
        break
    # Read each frame from the webcam
    ret, frame = cap.read()
    current_lang=values['-lang_name-']
    current_digit=values['-digit_value-']
    if ((current_lang != previous_lang) or (current_digit!=previous_digit)):
        set_fname_digit()
        previous_lang=current_lang
        previous_digit=current_digit
        print("capturing data for digit {0} in {1}".format(digit,folder_name))
 
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (15,15,15), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 45, 0), 2, cv2.LINE_AA)

    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 640)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        thumb1 = (landmarks[1][0],landmarks[1][1])
        fore_finger = (landmarks[8][0],landmarks[8][1])
        fore_finger5 =(landmarks[5][0],landmarks[5][1])
        middle_finger =(landmarks[12][0],landmarks[12][1])
        middle_finger9 =(landmarks[9][0],landmarks[9][1])
        ring_finger = [landmarks[16][0],landmarks[16][1]]
        ring_finger13 = (landmarks[13][0],landmarks[13][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        #if (thumb[1]-center[1]<30):
        x_diff = fore_finger5[0]-thumb[0]
        y_diff = fore_finger5[1]-thumb[1]
        # y_diff <0 
        if ((x_diff > diff_point)):
            
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            flag2=0
            # if ((thumb[1]> middle_finger9[1]) and thumb[0]> middle_finger9[0]):
            if(((fore_finger[1]>fore_finger5[1]))and flag>5):
                image_name = folder_name + '/' + digit + str(count) + '.jpg'
                
                paintWindow_image= process(paintWindow)
                if ( type(paintWindow_image) != type(0) ):
                    cv2.imwrite(image_name, paintWindow_image)
                    count = count + 1
                flag = 0
            #     clear()
            # elif (((middle_finger[1]>middle_finger9[1])) and flag > 5):
            #     clear()
                
                
        elif (center[1] <= 65) and (40 <= center[0] <= 140): #to clear
            clear()
        
        else :
            if(((fore_finger[1]>fore_finger5[1]))and flag>5):
                # print('clearing')
                clear()
                flag2=0
            elif ((middle_finger[1]>middle_finger9[1])and ((thumb1[1]-thumb[1])<(fore_finger5[1]-fore_finger[1])+40)):
                if flag2>5:
                    # print('drawing')
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)
                    flag=flag+1
                flag2=flag2+1
                window['-TEXT-'].update('')
            else :
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1



    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 9)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 9)

    # cv2.imshow("Output", frame) 
    # cv2.imshow("Paint", paintWindow)
    
    if cv2.waitKey(1) == ord('a'):
        image_name = folder_name + '/' + digit + str(count) + '.jpg'
        count = count + 1
        paintWindow_image= process(paintWindow)
        cv2.imwrite(image_name, paintWindow_image)
        flag = 0
        clear()
    if cv2.waitKey(1) == ord('c'):
        digit = input('Enter digit: ')
        while (not (digit.isdigit())):
            print("Enter number is not digit. Enter agian")
            digit = input("Enter_digit: ")
        list_of_files = glob.glob(folder_name+'/'+digit+'*.jpg') # * means all if need specific format then *.csv
        if (len(list_of_files)>0):
            latest_file = max(list_of_files, key=os.path.getctime)
            latest_file = latest_file.split('\\')[-1].split('.')[0]
            count = int(latest_file[1:]) +1
        else:
            count = 1
        print("capturing data for digit {0} in {1}".format(digit,folder_name))
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    imgbytes = cv2.imencode('.png',frame)[1].tobytes()
    window['-IMAGE-'].update(data = imgbytes)

    # update the text
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

