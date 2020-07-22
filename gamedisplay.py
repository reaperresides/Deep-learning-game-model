import cv2
import numpy as np
import time
import imutils
from win32com.client import Dispatch
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
import os
import matplotlib.pyplot as plt

say=Dispatch("SAPI.SpVoice")






def Model(x1,x2,p1,p2):
    #specifying image pixel size
    img_size=60

    #reading the cropped rectangles from the image
    x1_array=cv2.imread(p1,cv2.IMREAD_GRAYSCALE)
    x2_array=cv2.imread(p2,cv2.IMREAD_GRAYSCALE)

    #resizing the image array to be suited for prediction model
    x1_new_array=cv2.resize(x1_array,(img_size,img_size),1)
    x2_new_array=cv2.resize(x2_array,(img_size,img_size),1)




    #converting the array into 4th dimensional array
    x1_new_array=x1_new_array.reshape(-1,img_size,img_size,1)/255.0
    x2_new_array=x2_new_array.reshape(-1,img_size,img_size,1)/255.0

    #predicting the output
    game= tf.keras.models.load_model("game.h5")
    x1_predict=(np.argmax(game.predict(x1_new_array)))
    x2_predict=(np.argmax(game.predict(x2_new_array)))

    print(x1_predict,x2_predict)
    rock=0;paper=1;scissors=2;
    if x1_predict==rock:
        if x1_predict==rock and x2_predict==paper:
            say.Speak("player two wins")
        elif x1_predict==rock and x2_predict==scissors:
            say.Speak("player one wins")
        else:
            say.Speak("its a tie")

    elif x1_predict==paper:
        if x1_predict==paper and x2_predict==rock:
            say.Speak("player one wins")
        elif x1_predict==paper and x2_predict==scissors:
            say.Speak("player two wins")
        else:
            say.Speak("its a tie")

    elif x1_predict==scissors:
        if x1_predict==scissors and x2_predict==rock:
            say.Speak("player two wins")
        elif x2_predict==scissors and x2_predict==paper:
            say.Speak("player one wins")
        else:
            say.Speak("its a tie")
    
    else:
        pass





def Results():       
        cv2.imwrite("player.jpg",img)
        cv2.imread("player.jpg",cv2.IMREAD_GRAYSCALE)
        image=cv2.imread("player.jpg",cv2.IMREAD_GRAYSCALE)
        cropped_image1=image[32:300,22:495]
        cropped_image2=image[352:648,402:898]
        player1=cv2.imwrite("player1.jpg",cropped_image1)
        player2=cv2.imwrite("player2.jpg",cropped_image2)
        
        path1=os.path.join(os.getcwd(),"player1.jpg")
        path2=os.path.join(os.getcwd(),"player2.jpg")      
        return Model(player1,player2,path1,path2)

cap=cv2.VideoCapture(0)
say.speak("Welcome to rock paper sscissor,please align your fingers in the respective boxes")
while True:
    ret,img=cap.read()
    img = imutils.resize(img, width=1000)
    roi1=cv2.rectangle(img,(20,30),(500,330),(255,0,0),2)
    roi2=cv2.rectangle(img,(400,350),(900,650),(255,0,0),2)
    cv2.imshow("img",img)

    k=cv2.waitKey(33)
    if k==ord("s"):
        Results()
   
    elif k==27:
        break

cap.release()
cv2.destroyAllWindows()
        

  





