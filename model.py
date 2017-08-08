import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Conv2D, Dense, Dropout, MaxPooling2D, Cropping2D

lines = []
with open('./data/driving_log.csv') as csvfile:
          reader = csv.reader(csvfile)
          for line in reader:
              lines.append(line)
print('examples:', len(lines))

images = []
steering = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = './data/' + filename
    image = cv2.imread(current_path)
    angle = float(line[3])
    
    if abs(angle) > 0.15:                          #Pick images with steering angle>0.15
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        images.append(cv2.flip(image, 1))          #Pick flipped versions of those images too
    
        steering.append(angle)                     #Pick corresponding desired "y-output"
        steering.append(angle*(-1.0))              #and its flipped version
        
    elif np.random.random() > 0.8:                 #Of images with steering<0.15 retain ~20%
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        images.append(cv2.flip(image, 1))
    
        steering.append(angle)
        steering.append(angle*(-1.0))

X_train = np.array(images)
y_train = np.array(steering)

print(filename)
print(current_path)

model = Sequential()
model.add(Cropping2D(cropping=((46, 30), (50, 50)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))

model.add(Conv2D(32, 5, 5, activation='relu'))      #Convolution Layer 1
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, 5, 5, activation='relu'))      #Convolution Layer 2
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, 5, 5, activation='relu'))      #Convolution Layer 3
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(16))
#model.add(Dropout(0.3))
model.add(Dense(1))                                 #Fully Connected Layer--out

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=5, verbose=2)

model.save('3convdataflip.h5')
#print(model.summary())
