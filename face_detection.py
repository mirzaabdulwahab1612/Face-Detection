
# coding: utf-8

# In[70]:


#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pmig
from PIL import Image
import pickle


# In[71]:


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('D:/NUST/HackDay/haarcascade_frontalface_default.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    return faces

#     #under the assumption that there will be only one face,
#     #extract the face area
#     (x, y, w, h) = faces[0]

#     #return only the face part of the image
#     return gray[y:y+w, x:x+h], faces[0]


# In[72]:


#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# In[73]:


template_path = "D:/NUST/HackDay/templates/"
path = "D:/NUST/HackDay/test"
count = 0
images = os.listdir(path)
for image in images:
    print(image)
    face_img = pmig.imread(path + image)   
    faces = detect_face(face_img)
    
    print("Number of faces in image: " , image , " are: " , len(faces))
   
     # Crop faces and plot
    for face_rect in faces:
        (x, y, w, h) = face_rect
        face = face_img[y:y+h, x:x+w]
        #Displaying Template
        f, axarr = plt.subplots(1)
        axarr.imshow(face , cmap='gray')
        plt.show()
        print("Assign Label to this face: ")
        label = input()
        
        #Making a directory for each person
        directory = template_path + label
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        face = Image.fromarray(face)
        face.save(label + str(count) + '.png')
        count = count + 1
        
    
#     for face in faces:
#         temp = face_img.copy()
#         (x,y,w,h) = face
# #         face_boundry = gray[y:y+w, x:x+h]
#         print(face)
#         draw_rectangle(temp , face)
#         f, axarr = plt.subplots(1)
#         axarr.imshow(temp , cmap='gray')
#         plt.show()
#         print("Assign Label to this face:")
#         label = input()

    

        
    
    #Displaying Template
#     f, axarr = plt.subplots(1)
#     axarr.imshow(temp , cmap='gray')
#     plt.show()


# In[92]:


templates = os.listdir(template_path)
for template in templates:
    images = os.listdir(template_path + template)
    for image in images:
        print(template_path + template +'/'+ image)
        face_img = pmig.imread(template_path + template +'/'+ image)
        

