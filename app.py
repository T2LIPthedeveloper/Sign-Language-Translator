from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

import cv2
import mediapipe as mp
import os
from torchvision.transforms import transforms
import numpy as np
import torch
import __main__

from collections import Counter
from functions import *
from PointNet import PointNet

model_path = "./saved_models/PointNet-LR0.0001/model_10.pth"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
setattr(__main__, "PointNet", PointNet)
model = torch.load(model_path, map_location=torch.device("cpu"))
# For webcam input:
# For webcam input

cap = cv2.VideoCapture(0)

value = None
class VideoThread(QThread):

    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_value_signal = pyqtSignal(str)
    full_word_signal = pyqtSignal(str)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if not ret:
                continue
            cv_img = cv2.cvtColor(cv2.flip(cv_img, 1), cv2.COLOR_BGR2RGB)
            label, raw_points = predict(model, cv_img)
            if label is not None:
                self.update_value_signal.emit(label)
            else:
                self.update_value_signal.emit("next")
                
            cv_img.flags.writeable = False
            results = hands.process(cv_img)
            
        # Draw the hand annotations on the image.
            cv_img.flags.writeable = True
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cv_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            self.change_pixmap_signal.emit(cv_img)
            
    def clear(self):
        self.stored_word = ""
        
    def del_letter(self):
        self.current_letter = "del"
        
    def space(self):
        self.current_letter = "space"
                
                


class App(QWidget):
    def __init__(self):
        super().__init__()
        
        self.labels_history = []
        self.current_letter = ""
        self.stored_word = ""
        
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 1000
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Learn Sign Language')
        self.letter = QLabel("")
        self.full_word = QLabel("")
        self.button_clear = QPushButton('Clear')
        self.button_Del = QPushButton('Del')
        self.button_Space = QPushButton('Space')
        
        self.button_clear.clicked.connect(self.clear_word)
        self.button_Del.clicked.connect(self.del_letter)
        self.button_Space.clicked.connect(self.space)
        
        hboxletter = QHBoxLayout()
        hboxletter.addWidget(QLabel("Letter:"))
        hboxletter.addWidget(self.letter)
        
        hboxword = QHBoxLayout()
        hboxword.addWidget(QLabel("Word:"))
        hboxword.addWidget(self.full_word)
        
        hbox = QHBoxLayout()
        hbox.addWidget(self.button_clear)
        hbox.addWidget(self.button_Del)
        hbox.addWidget(self.button_Space)
        
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addLayout(hboxletter)
        vbox.addLayout(hboxword)
        vbox.addLayout(hbox)
        
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        
        # # create the video capture thread
        self.thread = VideoThread()
        # # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_value_signal.connect(self.update_value)
        self.thread.full_word_signal.connect(self.update_word)
        # start the thread
        self.thread.start()



    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_value(self, value):
        self.labels_history.append(value)
        if len(self.labels_history) > 50:
            self.labels_history.pop(0)
            
        if len(self.labels_history) == 50:
            most_common_label = Counter(self.labels_history).most_common(1)[0][0]
            if most_common_label == "next":
                if self.current_letter == "del" and len(self.stored_word) > 0:
                    self.stored_word = self.stored_word[:-1]
                elif self.current_letter == "space" and len(self.stored_word) > 0:
                    self.stored_word += " "
                else:
                    self.stored_word += self.current_letter
                self.labels_history = []
                self.current_letter = ""
                self.letter.setText("")
                self.full_word.setText(self.stored_word)
            else:
                self.current_letter = most_common_label
                self.letter.setText(most_common_label)
        
    
    def update_word(self, word):
        self.full_word.setText(word)
        
    def clear_word(self):
        self.stored_word = ""
        self.full_word.setText(self.stored_word)
        
    def del_letter(self):
        if len(self.stored_word) > 0:
            self.stored_word = self.stored_word[:-1]
            self.full_word.setText(self.stored_word)
        
    def space(self):
        self.stored_word += " "
        self.full_word.setText(self.stored_word)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())