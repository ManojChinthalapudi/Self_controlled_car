import numpy as np
import cv2
import pickle
from keras.models import load_model
#from tensorflow.keras.optimizers import Adam

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.90  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
#pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
#model = pickle.load(pickle_in)
model = load_model('my_model.p')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getCalssName(classNo):
    if (classNo == 0).any():
        return 'Speed Limit 20 km/h'
    elif (classNo == 1).any():
        return 'Speed Limit 30 km/h'
    elif (classNo == 2).any():
        return 'Speed Limit 50 km/h'
    elif (classNo == 3).any():
        return 'Speed Limit 60 km/h'
    elif (classNo == 4).any():
        return 'Speed Limit 70 km/h'
    elif (classNo == 5).any():
        return 'Speed Limit 80 km/h'
    elif (classNo == 6).any():
        return 'End of Speed Limit 80 km/h'
    elif (classNo == 7).any():
        return 'Speed Limit 100 km/h'
    elif (classNo == 8).any():
        return 'Speed Limit 120 km/h'
    elif (classNo == 9).any():
        return 'No passing'
    elif (classNo == 10).any():
        return 'No passing for vechiles over 3.5 metric tons'
    elif (classNo == 11).any():
        return 'Right-of-way at the next intersection'
    elif (classNo == 12).any():
        return 'Priority road'
    elif (classNo == 13).any():
        return 'Yield'
    elif (classNo == 14).any():
        return 'Stop'
    elif (classNo == 15).any():
        return 'No vechiles'
    elif (classNo == 16).any():
        return 'Vechiles over 3.5 metric tons prohibited'
    elif (classNo == 17).any():
        return 'No entry'
    elif (classNo == 18).any():
        return 'General caution'
    elif (classNo == 19).any():
        return 'Dangerous curve to the left'
    elif (classNo == 20).any():
        return 'Dangerous curve to the right'
    elif (classNo == 21).any():
        return 'Double curve'
    elif (classNo == 22).any():
        return 'Bumpy road'
    elif (classNo == 23).any():
        return 'Slippery road'
    elif (classNo == 24).any():
        return 'Road narrows on the right'
    elif (classNo == 25).any():
        return 'Road work'
    elif (classNo == 26).any():
        return 'Traffic signals'
    elif (classNo == 27).any():
        return 'Pedestrians'
    elif (classNo == 28).any():
        return 'Children crossing'
    elif (classNo == 29).any():
        return 'Bicycles crossing'
    elif (classNo == 30).any():
        return 'Beware of ice/snow'
    elif (classNo == 31).any():
        return 'Wild animals crossing'
    elif (classNo == 32).any():
        return 'End of all speed and passing limits'
    elif (classNo == 33).any():
        return 'Turn right ahead'
    elif (classNo == 34).any():
        return 'Turn left ahead'
    elif (classNo == 35).any():
        return 'Ahead only'
    elif (classNo == 36).any():
        return 'Go straight or right'
    elif (classNo == 37).any():
        return 'Go straight or left'
    elif (classNo == 38).any():
        return 'Keep right'
    elif (classNo == 39).any():
        return 'Keep left'
    elif (classNo == 40).any():
        return 'Roundabout mandatory'
    elif (classNo == 41).any():
        return 'End of no passing'
    elif (classNo == 42).any():
        return 'End of no passing by vechiles over 3.5 metric tons'


while True:

    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
   #classIndex = model.predict(img)
    classIndex = np.argmax(model.predict(img))
    probabilityValue = np.amax(predictions)
   # probabilityValue = np.argmax(predictions)
    if probabilityValue > threshold:
       # print(getCalssName(classIndex))
       cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)
       cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
