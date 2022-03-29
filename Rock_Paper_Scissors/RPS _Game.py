import cv2
from keras.models import load_model
import numpy as np
import time
import random

model = load_model('keras_model_version_2.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
t_0 = time.time()

while True: 

    def RPS_function(started = False, next_round = True, countdown = False, counter = 0, elapsed = 0 , round_time = 0, message = '', yourScore = 0, compScore = 0,round = 0):
   
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        print(time.time() - t_0)
    
        resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
        image_np = np.array(resized_frame)
        normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
        data[0] = normalized_image
        predictionArray = model.predict(data)  
        # Press q to close the window
    
    
            #Set prediction values to Rock/Paper/Scissors
        if predictionArray[0][0] > 0.75: 
                prediction = 'Rock'
        elif predictionArray[0][1] > 0.75: 
                prediction = 'Paper'
        elif predictionArray[0][2] > 0.75: 
                prediction = 'Scissors'
        else: 
                prediction = 'Nothing'
    
    
        if not started: 
                message = 'Press a to start'
        if cv2.waitKey(33) == ord('a'):
            if not started:
                counter = time.time()
                started = True
                countdown = True
                    
        if started:
            elapsed = 5 - (time.time()- counter)
            if elapsed <= -4:
                message = 'Press n to play next round'
                if cv2.waitKey(33) == ord('n'):
                    started = False 
                    elapsed = 0
                    round += 1

            elif elapsed <= 0 and round <= 3:
                countdown = False 
                # prediction = 'Rock'
                computer = random.choice(['Rock','Paper','Scissors'])
                message = 'You showed {prediction} and the computer showed {computer}.'
            
                if prediction == 'Rock':
                    if computer == 'Rock':
                        message += "It's a draw"
                        
                    elif computer == 'Paper': 
                        message += 'Ah, tough luck'
                        compScore += 1
                        
                    else: #  computer == 'Scissors': 
                        message += 'Fantastic, you won'
                        yourScore += 1

                elif prediction == 'Paper':
                    if computer == 'Rock':
                        message += 'Fantastic, you won'
                        yourScore += 1
                    
                    elif computer == 'Paper': 
                        message += "It's a draw"

                    else: #  computer == 'Scissors': 
                        message += 'Ah, tough luck'
                        compScore += 1

                elif prediction == 'Scissors':
                    if computer == 'Rock':
                        message += 'Ah, tough luck'
                        compScore += 1

                    elif computer == 'Paper': 
                        message += 'Fantastic, you won'
                        yourScore += 1

                    else: #  computer == 'Scissors': 
                        message += "It's a draw"
                    
        if round == 3:
            message = 'The Score is {yourScore}: {compScore}'
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()