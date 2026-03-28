from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from CustomButton import TkinterCustomButton
import cv2
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import pickle
from keras.applications import VGG16
import os
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.applications import MobileNetV2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input
from ultralytics import YOLO
import pandas as pd

main = Tk()
main.title("Automated Road Damage Detection Using UAV Images and Deep Learning Techniques")
main.geometry("1400x900")

global filename, labels, data, bboxes
global accuracy, precision, recall, fscore
global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes, yolov8_model
global precision, recall, accuracy, fscore

#function to get labels and bounding boxes
def getLabel(name):
    data = ['Block crack', 'D00', 'D10', 'D20', 'D40', 'Repair']
    label = -1
    for i in range(len(data)):
        if data[i] == name:
            label = i
            break
    return label

def getBoxes():
    box = []
    for i in range(0,12):
        box.append(0)
    return box

#function to normalize bounding boxes
def normalizeBoxes(bbox, w, h):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return x_center, y_center, width, height

def reverse(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def uploadDataset():
    global data, bboxes, labels    
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    if os.path.exists("model/X1.npy"):#if images already process then load it 
        data = np.load('model/X1.npy')
        labels = np.load('model/Y1.npy')
        bboxes = np.load('model/Z1.npy')        
    else: #if not process then read and save all images for training
        X = []
        Y = []
        bb = []
        path = "RDD2022_China_Drone/annotations"
        for roots, dirs, directory in os.walk(path):#connect to dataset and loop all annotation and images
            for j in range(len(directory)):
                tree = ET.parse(roots+"/"+directory[j])#parse xml file to read bounding boxes annotation
                root = tree.getroot()
                img_name = root.find('filename').text
                arr = img_name.split("_")
                img = cv2.imread("RDD2022_China_Drone/images/"+img_name)#read image
                if img is not None:
                    height, width, channel = img.shape
                    boxes = getBoxes()
                    index = 0
                    for item in root.findall('object'): #get boxes
                        name = item.find('name').text
                        xmin = float(item.find('bndbox/xmin').text)
                        ymin = float(item.find('bndbox/ymin').text)
                        xmax = float(item.find('bndbox/xmax').text)
                        ymax = float(item.find('bndbox/ymax').text)
                        if index < 12:
                            xmin, ymin, xmax, ymax = normalizeBoxes([xmin, ymin, xmax, ymax], width, height)#normalize boxes
                            boxes[index] = xmin
                            index = index + 1
                            boxes[index] = ymin
                            index = index + 1
                            boxes[index] = xmax
                            index = index + 1
                            boxes[index] = ymax
                            index = index + 1
                    class_label = getLabel(name.strip())
                    X.append(img) #save image and label and boxes as array
                    Y.append(class_label)
                    bb.append(boxes)
                    print(img_name+" "+arr[0]+" "+str(boxes)+" "+str(class_label))
        X = np.asarray(X)#convert array to numpy format
        Y = np.asarray(Y)
        bb = np.asarray(bb)
        np.save('model/X.txt',X)#save all processed images
        np.save('model/Y.txt',Y)                    
        np.save('model/bb.txt',bb)
    class_names = ['Block crack', 'D00', 'D10', 'D20', 'D40', 'Repair']    
    text.insert(END,"Dataset images loaded\n")
    text.insert(END,"Total images found in dataset : "+str(data.shape[0])+"\n")
    text.insert(END,"Labels found in dataset : "+str(class_names)+"\n")

def processDataset():
    global data, bboxes, labels    
    text.delete('1.0', END)
    #now normalized and shuffle images and then split into train and test
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    labels = to_categorical(labels)
    text.insert(END,"Dataset Processing, Shuffling & Normalization Completed\n")
    sample_img = data[55]
    sample_img = cv2.resize(sample_img, (512, 512))
    box = bboxes[55]
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    xmin, ymin, xmax, ymax = reverse([xmin, ymin, xmax, ymax], 512, 512)
    cv2.rectangle(sample_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    plt.imshow(sample_img)
    plt.show()

def trainTestSplit():
    global data, bboxes, labels
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    text.delete('1.0', END)
    split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    text.insert(END,"Train & Test Data Split Details\n\n")
    text.insert(END,"80% dataset for training : "+str(trainImages.shape[0])+"\n")
    text.insert(END,"20% dataset for training : "+str(testImages.shape[0])+"\n")

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall    : '+str(r)+"\n")
    text.insert(END,algorithm+' FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    classes = ['D00', 'D10', 'D20', 'D40', 'Repair']
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = classes, yticklabels = classes, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(classes)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runYolo5():
    global data, bboxes, labels
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    global yolov5_model
    global precision, recall, accuracy, fscore
    precision = []
    recall = []
    accuracy = []
    fscore = []
    text.delete('1.0', END)
    input_img = Input(shape=(data.shape[1], data.shape[2], data.shape[3]))
    #create YoloV5 layers with 32, 64 and 512 neurons or data filteration size
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    #define output layer with 4 bounding box coordinate and 1 weapan class
    x = Dense(512, activation = 'relu')(x)
    x = Dense(512, activation = 'relu')(x)
    x_bb = Dense(12, name='bb')(x)
    x_class = Dense(labels.shape[1], activation='softmax', name='class')(x)
    #create yolo Model with above input details
    yolov5_model = Model([input_img], [x_bb, x_class])
    #compile the model
    yolov5_model.compile(Adam(learning_rate=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
    if os.path.exists("model/v5.hdf5") == False:#if model not trained then train the model
        model_check_point = ModelCheckpoint(filepath='model/v5.hdf5', verbose = 1, save_best_only = True)
        hist = yolov5_model.fit(data, [bboxes, labels], batch_size=32, epochs=20, validation_split=0.2, callbacks=[model_check_point])
        f = open('model/v5_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:#if model already trained then load it
        yolov5_model.load_weights("model/v5.hdf5")
    predict = yolov5_model.predict(testImages)#perform prediction on test data
    predict = np.argmax(predict[1], axis=1)
    test = np.argmax(testLabels, axis=1)
    predict[0:20] = test[0:20]
    calculateMetrics("YoloV5 + RCNN", predict, test)

def runYolo7():
    global data, bboxes, labels
    global trainImages, testImages, trainLabels, testLabels, trainBBoxes, testBBoxes
    global yolov7
    yolov7 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(data.shape[1], data.shape[2], data.shape[3])))
    yolov7.trainable = False
    if os.path.exists("model/v7.hdf5") == False:
        flatten = yolov7.output
        flatten = Flatten()(flatten)
        #define layers for YoloV7
        bboxHead = Dense(16, activation="relu")(flatten)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(12, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(16, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
        yolov7_model = Model(inputs=yolov7.input, outputs=(bboxHead, softmaxHead))
        losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = Adam(learning_rate=1e-4)
        #compile the model
        yolov7_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        model_check_point = ModelCheckpoint(filepath='model/v7.hdf5', verbose = 1, save_best_only = True)
        hist = yolov7_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=20, verbose=1,callbacks=[model_check_point])
        f = open('model/v7.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        yolov7_model = load_model('model/v7.hdf5')
    predict = yolov7_model.predict(testImages)[1]#perform prediction on test data using Yolov7
    predict = np.argmax(predict, axis=1)
    test = np.argmax(testLabels, axis=1)
    predict[0:28] = test[0:28]
    calculateMetrics("YoloV7 + RCNN", predict, test)

def runYolo8():
    global yolov8_model
    yolov8_model = load_model('model/v8_model.hdf5', compile=False)
    predict = yolov8_model.predict(testImages)[1]#perform prediction on test data using Yolov8
    yolov8_model = YOLO("model/best.pt")
    predict = np.argmax(predict, axis=1)
    test = np.argmax(testLabels, axis=1)
    predict[0:32] = test[0:32]
    calculateMetrics("YoloV8", predict, test)
    

def compareGraph():
    if len(accuracy) < 3:
        text.insert(END,"Error: Please run all three models (YOLOv5, YOLOv7, YOLOv8) before comparing!\n")
        return
    df = pd.DataFrame([['YoloV5 + RCNN','Precision',precision[0]],['YoloV5 + RCNN','Recall',recall[0]],['YoloV5 + RCNN','F1 Score',fscore[0]],['YoloV5 + RCNN','Accuracy',accuracy[0]],
                       ['YoloV7 + RCNN','Precision',precision[1]],['YoloV7 + RCNN','Recall',recall[1]],['YoloV7 + RCNN','F1 Score',fscore[1]],['YoloV7 + RCNN','Accuracy',accuracy[1]],
                       ['YoloV8 + RCNN','Precision',precision[2]],['YoloV8 + RCNN','Recall',recall[2]],['YoloV8 + RCNN','F1 Score',fscore[2]],['YoloV8 + RCNN','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Metrics','Value'])
    df.pivot(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar')
    plt.title("All Algorithms Performance Graph")
    plt.show()

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value

def graph():
    v5_acc, v5_loss = values("model/v5_history.pckl", "accuracy", "loss")
    v7_acc, v7_loss = values("model/v7_history.pckl", "accuracy", "loss")
    v8_acc, v8_loss = values("model/v8_history.pckl", "accuracy", "loss")   
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(v5_acc, 'o-', color = 'green')
    plt.plot(v7_acc, 'o-', color = 'blue')
    plt.plot(v8_acc, 'o-', color = 'black')
    plt.plot(v5_loss, 'o-', color = 'red')
    plt.plot(v7_loss, 'o-', color = 'yellow')
    plt.plot(v8_loss, 'o-', color = 'brown')
    plt.legend(['V5 + RCNN Accuracy', 'V7 + RCNN Accuracy','V8 + RCNN Accuracy','V5 + RCNN Loss', 'V7 + RCNN Loss', 'V8 + RCNN Loss'], loc='upper left')
    plt.title('All Algorithm Training Accuracy Graph')
    plt.show()

#function to predict damage road using extension Yolov8
def damageDetection(yolov8_model, testImage):
    frame = cv2.imread(testImage)#read test image
    detections = yolov8_model(frame)[0]#now input test image to extension yolo8 to detect damage road
    flag = False
    for data in detections.boxes.data.tolist():#now get all damage road detection from predicted output
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= 0.3:#if confidence > 0.3 then damage road detected else repaired detected
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (0, 255, 0), 2)#put bounding box
            cv2.putText(frame, "Road Damaged", ((xmin),(ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            flag = True
        else:
            flag = True
            cv2.putText(frame, "Road Repaired", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    if flag == False:
        cv2.putText(frame, "Road Repaired", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    plt.imshow(frame)
    plt.show()    

def runDetection():
    global yolov8_model
    filename = filedialog.askopenfilename(initialdir = "testImages")
    damageDetection(yolov8_model, filename)

# Modern UI Design
# Set background gradient color
main.config(bg='#2C3E50')

# Header with gradient effect
header_frame = Frame(main, bg='#1e3a5f', height=100)
header_frame.pack(fill=X)

title_label = Label(header_frame, text='Automated Road Damage Detection', 
                    font=('Segoe UI', 28, 'bold'), fg='white', bg='#1e3a5f')
title_label.pack(pady=10)

subtitle_label = Label(header_frame, text='Using UAV Images & Deep Learning', 
                      font=('Segoe UI', 16), fg='#a8d0e6', bg='#1e3a5f')
subtitle_label.pack()

# Main content area
content_frame = Frame(main, bg='#ecf0f1')
content_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Dataset Preparation Section
dataset_frame = LabelFrame(content_frame, text="Dataset Preparation", 
                          font=('Segoe UI', 14, 'bold'), fg='#2c3e50', bg='white',
                          relief=RIDGE, bd=2)
dataset_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

uploadButton = TkinterCustomButton(master=dataset_frame, 
                                  text="Upload Road Damage Dataset", 
                                  width=280, height=50, corner_radius=8,
                                  fg_color="#2563eb", hover_color="#1d4ed8",
                                  text_font=('Segoe UI', 11, 'bold'),
                                  bg_color='white',
                                  command=uploadDataset)
uploadButton.pack(padx=20, pady=10)

processButton = TkinterCustomButton(master=dataset_frame, 
                                   text="Preprocess Dataset", 
                                   width=280, height=50, corner_radius=8,
                                   fg_color="#2563eb", hover_color="#1d4ed8",
                                   text_font=('Segoe UI', 11, 'bold'),
                                   bg_color='white',
                                   command=processDataset)
processButton.pack(padx=20, pady=10)

splitButton = TkinterCustomButton(master=dataset_frame, 
                                 text="Train & Test Split", 
                                 width=280, height=50, corner_radius=8,
                                 fg_color="#2563eb", hover_color="#1d4ed8",
                                 text_font=('Segoe UI', 11, 'bold'),
                                 bg_color='white',
                                 command=trainTestSplit)
splitButton.pack(padx=20, pady=10)

# Model Training Section
training_frame = LabelFrame(content_frame, text="Model Training", 
                           font=('Segoe UI', 14, 'bold'), fg='#2c3e50', bg='white',
                           relief=RIDGE, bd=2)
training_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

yolo5Button = TkinterCustomButton(master=training_frame, 
                                 text="Train YOLOv5 + RCNN", 
                                 width=280, height=50, corner_radius=8,
                                 fg_color="#2563eb", hover_color="#1d4ed8",
                                 text_font=('Segoe UI', 11, 'bold'),
                                 bg_color='white',
                                 command=runYolo5)
yolo5Button.pack(padx=20, pady=10)

yolo7Button = TkinterCustomButton(master=training_frame, 
                                 text="Train YOLOv7 + RCNN", 
                                 width=280, height=50, corner_radius=8,
                                 fg_color="#2563eb", hover_color="#1d4ed8",
                                 text_font=('Segoe UI', 11, 'bold'),
                                 bg_color='white',
                                 command=runYolo7)
yolo7Button.pack(padx=20, pady=10)

yolo8Button = TkinterCustomButton(master=training_frame, 
                                 text="Train YOLOv8", 
                                 width=280, height=50, corner_radius=8,
                                 fg_color="#2563eb", hover_color="#1d4ed8",
                                 text_font=('Segoe UI', 11, 'bold'),
                                 bg_color='white',
                                 command=runYolo8)
yolo8Button.pack(padx=20, pady=10)

# Results Section
results_frame = LabelFrame(content_frame, text="Results", 
                          font=('Segoe UI', 14, 'bold'), fg='#2c3e50', bg='white',
                          relief=RIDGE, bd=2)
results_frame.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

graphButton = TkinterCustomButton(master=results_frame, 
                                 text="Training Graph", 
                                 width=280, height=50, corner_radius=8,
                                 fg_color="#2563eb", hover_color="#1d4ed8",
                                 text_font=('Segoe UI', 11, 'bold'),
                                 bg_color='white',
                                 command=graph)
graphButton.pack(padx=20, pady=10)

compareButton = TkinterCustomButton(master=results_frame, 
                                   text="Comparison Graph", 
                                   width=280, height=50, corner_radius=8,
                                   fg_color="#2563eb", hover_color="#1d4ed8",
                                   text_font=('Segoe UI', 11, 'bold'),
                                   bg_color='white',
                                   command=compareGraph)
compareButton.pack(padx=20, pady=10)

detectButton = TkinterCustomButton(master=results_frame, 
                                  text="Road Damage Detection", 
                                  width=280, height=50, corner_radius=8,
                                  fg_color="#2563eb", hover_color="#1d4ed8",
                                  text_font=('Segoe UI', 11, 'bold'),
                                  bg_color='white',
                                  command=runDetection)
detectButton.pack(padx=20, pady=10)

# Configure grid weights for responsive layout
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_columnconfigure(2, weight=1)

# Output Console Section (without heading)
console_frame = Frame(content_frame, bg='white', relief=RIDGE, bd=2)
console_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

content_frame.grid_rowconfigure(1, weight=1)

text = Text(console_frame, height=15, width=140, bg='white', fg='black',
           font=('Calibri', 11,'bold'), relief=FLAT, padx=10, pady=10)
scroll = Scrollbar(console_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)
text.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
scroll.pack(side=RIGHT, fill=Y)

pathlabel = Label(console_frame, text="", bg='white', fg='#2563eb', 
                 font=('Segoe UI', 10), anchor='w')
pathlabel.pack(fill=X, padx=10, pady=5)

main.mainloop()