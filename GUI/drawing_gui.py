################################################
#      Point, Shapes, Painter Classes by      #
#    By Geoff Samuel www.GeoffSamuel.com       #
################################################
 
# Import Modules
from PyQt5 import QtGui,QtCore, uic, QtWidgets
import os.path, sys
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)+'/Neural Network/')
import numpy as np
from scipy import ndimage

from dnn import *
from plotting import *
from imagePreprocessing import *

# Load the UI File
gui_model = 'GUI.ui'
form, base = uic.loadUiType(gui_model)
image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir) + '/Images/'

# Point class for shapes      
class Point:
    x, y = 0, 0
    def __init__(self, nx=0, ny=0): 
        self.x = nx
        self.y = ny
        
# Single shape class
class Shape:
    location = Point()
    number = 0
    def __init__(self, L, S):
        self.location = L
        self.number = S

# Cotainer Class for all shapes
class Shapes:
    shapes = []
    def __init__(self):
        self.shapes = []
    # Returns the number of shapes
    def NumberOfShapes(self):
        return len(self.shapes)
    # Add a shape to the database, recording its position
    def NewShape(self, L, S):
        shape = Shape(L,S)
        self.shapes.append(shape)
    # Returns a shape of the requested data.
    def GetShape(self, Index):
        return self.shapes[Index]

# Class for painting widget
class Painter(QtWidgets.QWidget):
    ParentLink = 0
    MouseLoc = Point(0,0)  
    LastPos = Point(0,0)  
    def __init__(self, parent):
        super(Painter, self).__init__()
        self.ParentLink = parent
        self.MouseLoc = Point(0,0)
        self.LastPos = Point(0,0) 
    #Mouse down event
    def mousePressEvent(self, event): 
        self.ParentLink.IsPainting = True
        self.ParentLink.ShapeNum += 1
        self.LastPos = Point(0,0)    
    #Mouse Move event        
    def mouseMoveEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.MouseLoc = Point(event.x(),event.y())
            if((self.LastPos.x != self.MouseLoc.x) and (self.LastPos.y != self.MouseLoc.y)):
                self.LastPos =  Point(event.x(),event.y())
                self.ParentLink.DrawingShapes.NewShape(self.LastPos, self.ParentLink.ShapeNum)
            self.repaint()             
    #Mose Up Event         
    def mouseReleaseEvent(self, event):
        if(self.ParentLink.IsPainting == True):
            self.ParentLink.IsPainting = False
    # Paint Event
    def paintEvent(self,event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()
    # Draw the line       
    def drawLines(self, event, painter):   
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes()-1):     
            T = self.ParentLink.DrawingShapes.GetShape(i)
            T1 = self.ParentLink.DrawingShapes.GetShape(i+1) 
            if(T.number== T1.number):
                pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 7, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(T.location.x, T.location.y, T1.location.x, T1.location.y)
        
#Main UI Class
class CreateUI(base, form):
    DrawingShapes = Shapes()
    PaintPanel = 0
    IsPainting = False
    ShapeNum = 0
    def __init__(self):
        # Set up main window and widgets
        super(base,self).__init__()
        self.setupUi(self)
        self.setObjectName('Rig Helper')
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0,self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        # Set up Label for on hold picture
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(460, 70, 280, 280))
        self.pixmap = QtGui.QPixmap(image_path + str(-1) +".png")
        self.label.setPixmap(self.pixmap)
        self.Clear_Button.clicked.connect(self.ClearSlate)
        self.Predict_Button.clicked.connect(self.PredictNumber)
    # Reset Button
    def ClearSlate(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  
        self.pixmap = QtGui.QPixmap(image_path + str(-1) +".png")
        self.label.setPixmap(self.pixmap)
    # Predict Button
    def PredictNumber(self):
        image = get_image(self.DrawingFrame)
        pred = nn_predict(image=image)
        self.pixmap = QtGui.QPixmap(image_path + str(int(pred)) +".png")
        self.label.setPixmap(self.pixmap)


# Starting the GUI Application      
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = CreateUI()
    main_window.show()
    sys.exit(app.exec_())