# -*- coding: utf-8 -*-
#!/usr/bin/env python

import logging
import os
import sys
#import threading,time
import threading
import time
#from PyQt5 import QtSvg
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

import sprial
from Create_Drill import Create_Drill_fun
import math
from OCC.Core.BRepTools import breptools_Write
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Trsf, gp_Vec

from OCC.Display.OCCViewer import OffscreenRenderer
from OCC.Display.backend import load_backend, get_qt_modules
from OCC.Extend.TopologyUtils import TopologyExplorer
from PyQt5.QtWidgets import QHBoxLayout, QDockWidget, \
    QListWidget, QFileDialog

from PyQt5 import QtCore, QtWidgets
from graphics import GraphicsView, GraphicsPixmapItem
import Vision
from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
                             TopAbs_SHELL, TopAbs_SOLID)



#------------------------------------------------------------开始初始化环境
log = logging.getLogger(__name__)
def check_callable(_callable):
    if not callable(_callable):
        raise AssertionError("The function supplied is not callable")
backend_str=None
size=[850, 873]
display_triedron=True
background_gradient_color1=[206, 215, 222]
background_gradient_color2=[128, 128, 128]
if os.getenv("PYTHONOCC_OFFSCREEN_RENDERER") == "1":
    # create the offscreen renderer
    offscreen_renderer = OffscreenRenderer()


    def do_nothing(*kargs, **kwargs):
        """ takes as many parameters as you want,
        ans does nothing
        """
        pass


    def call_function(s, func):
        """ A function that calls another function.
        Helpfull to bypass add_function_to_menu. s should be a string
        """
        check_callable(func)
        log.info("Execute %s :: %s menu fonction" % (s, func.__name__))
        func()
        log.info("done")

    # returns empty classes and functions
used_backend = load_backend(backend_str)
log.info("GUI backend set to: %s", used_backend)
#------------------------------------------------------------初始化结束
from OCC.Display.qtDisplay import qtViewer3d
import MainGui
from PyQt5.QtGui import QPixmap
QtCore, QtGui, QtWidgets, QtOpenGL = get_qt_modules()
from OCC.Extend.DataExchange import read_step_file,write_step_file,read_stl_file,read_iges_file,read_step_file_with_names_colors
from OCC.Core.TopoDS import TopoDS_Shape,TopoDS_Builder,TopoDS_Compound,topods_CompSolid

class Vision(QtWidgets.QMainWindow,Vision.Ui_Form):
    def __init__(self,parent=None):
        super(Vision, self).__init__(parent)
        self.setupUi(self)

class Mywindown(QtWidgets.QMainWindow,MainGui.Ui_MainWindow):
    pass
    def __init__(self, parent=None):
        super(Mywindown,self).__init__(parent)
        self.setupUi(self)
        #3D显示设置
        self.canva = qtViewer3d(self.tab_4)#链接3D模块
        self.setWindowTitle("刀具参数化建模软件")
        #self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(self.width(), self.height());
        self.canva.setGeometry(QtCore.QRect(0, 0, 541, 400))
        self.centerOnScreen()
        self.version=Vision()
        #------------------------------------------------------------------视图设置
        self.quit.triggered.connect(self.Quit)
        self.actionView_Right.triggered.connect(self.View_Right)
        self.actionView_Left.triggered.connect(self.View_Left)
        self.actionView_Top.triggered.connect(self.View_Top)
        self.actionView_Bottom.triggered.connect(self.View_Bottom)
        self.actionView_Front.triggered.connect(self.View_Front)
        self.actionView_Iso.triggered.connect(self.View_Iso)
        #------------------------------------------------------------尺寸数据显示设置
        pix = QPixmap('Pic\\pic1.jpg')
        self.graphicsView = GraphicsView(pix, self.tab_5)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 541, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.item = GraphicsPixmapItem(pix)  # 创建像素图元
        self.scene = QtWidgets.QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)



        #--------------------------------------------------------------状态栏
        self.statusBar().showMessage('状态：软件运行正常')
        self.pushButton_3.clicked.connect(self.Create_shape)#生成钻头
        self.pushButton_5.clicked.connect(self.Copy_part_to_path)#导出
        self.pushButton_11.clicked.connect(self.Quit)#退出
        self.pushButton_4.clicked.connect(self.Parameter_Reset)#参数重置
        self.pushButton_10.clicked.connect(self.about)#关于
        self.pushButton_9.clicked.connect(self.Delete_shape)#删除
        self.pushButton_12.clicked.connect(self.drill_simulation)  # 钻孔仿真
        self.pushButton_13.clicked.connect(self.Clear_simulation)  # 清除仿真
        self.pushButton_15.clicked.connect(self.Get_select_shape)
        self.pushButton_16.clicked.connect(self.Move_shape_x)
        self.pushButton_17.clicked.connect(self.Move_shape_y)
        self.pushButton_18.clicked.connect(self.Move_shape_z)
        #self.setGeometry(300, 300, 250, 150)
        #---------------------------------------------------------------菜单栏
        self.import_part.triggered.connect(self.Import_part)
        self.tabWidget_2.currentChanged["int"].connect(self.Refresh)

        #-----------------------------------------------------------------------初始化变量
        self.shape = TopoDS_Shape
        self.filename=str()

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.WindowStateChange:
            if self.isMinimized():
                # print("窗口最小化")
                #self.canva._display.Repaint()
                pass
            elif self.isMaximized():
                pass
                # print("窗口最大化")
                #self.canva._display.Repaint()
            elif self.isFullScreen():
                # print("全屏显示")
                pass
            elif self.isActiveWindow():
                # print("活动窗口")
                self.canva._display.Repaint()
                pass

    def View_Bottom(self):
        pass
        self.canva._display.View_Bottom()
    def View_Front(self):
        pass
        self.canva._display.View_Front()
    def View_Iso(self):
        pass
        self.canva._display.View_Iso()

    def View_Left(self):
        pass
        self.canva._display.View_Left()
    def View_Right(self):
        pass
        self.canva._display.View_Right()

    def View_Top(self):
        pass
        self.canva._display.View_Top()

    def centerOnScreen(self):
        '''Centers the window on the screen.'''
        resolution = QtWidgets.QApplication.desktop().screenGeometry()
        x = (resolution.width() - self.frameSize().width()) / 2
        y = (resolution.height() - self.frameSize().height()) / 2
        #self.move(x, y)

    def Refresh(self):
        self.canva._display.Repaint()
        self.graphicsView.show()
    def updata_show(self):#更新界面参数
        pass

    def Parameter_Reset(self):
        self.lineEdit.setText("4")
        self.lineEdit_2.setText("8")
        self.lineEdit_3.setText("3.5")
        self.lineEdit_4.setText("1")
        self.lineEdit_5.setText("80")
        self.lineEdit_6.setText("3")
        self.lineEdit_7.setText("118")

    def updata_show_(self):#更新界面参数
        pass

        # -------------导入3D数据-----------------------------

    def Import_part(self):  # 导入数据
        pass
        # 清除之前数据
        try:
            self.chose_document = QFileDialog.getOpenFileName(self, '打开文件', './',
                                                              " STP files(*.stp , *.step);;(*.iges);;(*.stl)")  # 选择转换的文价夹
            filepath = self.chose_document[0]  # 获取打开文件夹路径
            # 判断文件类型 选择对应的导入函数
            end_with = str(filepath).lower()
            if end_with.endswith(".step") or end_with.endswith("stp"):
                self.import_shape = read_step_file(filepath)

            elif end_with.endswith("iges"):
                self.import_shape = read_iges_file(filepath)
            elif end_with.endswith("stl"):
                self.import_shape = read_stl_file(filepath)

            try:
                # self.canva._display.Context.Remove(self.show[0], True)
                self.acompound=self.import_shape
                self.canva._display.EraseAll()
                self.canva._display.hide_triedron()
                self.canva._display.display_triedron()
                self.show = self.canva._display.DisplayShape(self.import_shape, color="WHITE", update=True)
                self.canva._display.FitAll()
            except:
                pass
                #self.show = self.canva._display.DisplayShape(self.import_shape, color="WHITE", update=True)
                #self.canva._display.FitAll()
                pass
            self.statusbar.showMessage("状态：打开成功")  ###
            self.statusBar().showMessage('状态：软件运行正常')
        except:
            pass

    def Create_shape(self):
        try:

            R =float(self.lineEdit.text())
            self.lineEdit_2.setText(str(R*2))
            Rr =float(self.lineEdit_3.text())
            b = float(self.lineEdit_4.text())
            H = float(self.lineEdit_5.text())
            a = float(self.lineEdit_6.text())
            sigma =float(self.lineEdit_7.text())
            self.statusBar().showMessage('状态：模型生成中.........')
            self.drill_shape=Create_Drill_fun(R=R,Rr=Rr,b=b,H=H,sigma=sigma,a=a)

            self.Show_3d_model()
            self.statusBar().showMessage('状态：模型生成成功')
        except:
            self.statusBar().showMessage('状态：输入参数不正确')
    def Show_3d_model(self):

        pass
        #更新显示
        try:
            self.canva._display.EraseAll()
            self.canva._display.hide_triedron()
            self.canva._display.display_triedron()
            self.canva._display.Repaint()
            self.aCompound.Free()
        except:
            pass

        try:
            self.statusBar().showMessage('状态：模型显示中......')
            self.show=self.canva._display.DisplayShape(self.drill_shape,update=True)

            for i in TopologyExplorer(self.acompound).solids():
                self.show_list=self.canva._display.DisplayShape[i]

            self.canva._display.FitAll()
            self.statusBar().showMessage('状态：软件运行正常')
        except:
            pass


    def Copy_part_to_path(self):  #生成数据到指定路径
        try:

            self.filename = "drillr"
            path="D:\\"+self.filename
            fileName, ok = QFileDialog.getSaveFileName(self, "文件保存", path, "All Files (*);;Text Files (*.step)")
            write_step_file(self.drill_shape, fileName)
            self.statusBar().showMessage('状态：导出成功')
            #breptools_Write(self.aCompound, 'box.brep')
        except:
            self.statusBar().showMessage('错误：没有模型可以导出')
            pass
    def Delete_shape(self):
        self.canva._display.Context.Remove(self.show[0],True)
    def Clear_simulation(self):
        self.canva._display.Context.Remove(self.show_Blank[0],True)
    def Get_select_shape(self):
        self.measure_signal=1
    def Create_workplace(self):
        try:
            L=float(450)
            W = float(450)
            H = float(50)
            self.Blank=BRepPrimAPI_MakeBox(L, W, H).Shape()
            self.Blank = TopoDS_Shape(self.Blank)
            T = gp_Trsf()
            location_X = -L/2 # 把键槽移动到合适的位置
            location_Y = -W / 2  # 把键槽移动到合适的位置
            T.SetTranslation(gp_Vec(location_X,location_Y, H))
            loc = TopLoc_Location(T)
            self.Blank.Location(loc)
            #self.my_cylinder = BRepPrimAPI_MakeCylinder(10, 50).Shape()
            self.show_Blank=self.canva._display.DisplayShape(self.Blank,update=True)
            #self.show_my_cylinder = self.canva._display.DisplayShape(self.my_cylinder, update=True)
        except:
            pass
    def Axis_move(self,distance_x=None,distance_y=None,distance_z=None):
        pass
        try:
            self.X_Axis = gp_Trsf()  # 变换类
            self.X_Axis.SetTranslation(gp_Vec(distance_x, distance_y, distance_z))  # 设置变换类为平移
            self.X_Axis_Toploc = TopLoc_Location(self.X_Axis)
            self.canva._display.Context.SetLocation(self.show[0], self.X_Axis_Toploc)
            self.canva._display.Context.UpdateCurrentViewer()
        except  Exception as e:
            print(e)
    def Mill_cut(self,x=0,y=0,z=0):
        #钻孔点
        drill_hole_pnt=[[200,200,30],[200,200,-50],
                        [-200,200,30],[-200,200,-50],
                        [-200,-200,30],[-200,-200,-50],
                        [200,-200,30],[200,-200,-50]]
        try:
            for pnt in drill_hole_pnt:
                try:
                    time.sleep(0.2)
                    x = pnt[0]
                    y = pnt[1]
                    z = pnt[2]
                    self.Axis_move(distance_x=x, distance_y=y, distance_z=z)
                    dimater=float(self.lineEdit_2.text())
                    self.my_cylinder = BRepPrimAPI_MakeCylinder(dimater, 1000).Shape()
                    self.tool = TopoDS_Shape(self.my_cylinder)
                    T = gp_Trsf()
                    location_y = y  # 把刀具移动到合适的位置
                    location_x = x  # 把刀具移动到合适的位置
                    location_z = z  # 把刀具移动到合适的位置
                    T.SetTranslation(gp_Vec(location_x, location_y, location_z))
                    loc = TopLoc_Location(T)
                    self.tool.Location(loc)
                    self.Cutting_result = BRepAlgoAPI_Cut(self.Blank, self.tool).Shape()
                    self.canva._display.Context.Remove(self.show_Blank[0], True)
                    self.show_Blank = self.canva._display.DisplayShape(self.Cutting_result, update=True)
                    self.Blank = self.Cutting_result
                    QtWidgets.QApplication.processEvents()  # 一定加上这个功能，不然有卡顿
                except:
                    pass

        except Exception as e:
            pass
            self.statusBar().showMessage('错误：请确认刀具已经生成')
    def drill_simulation(self):
        self.Create_workplace()
        self.Mill_cut()
    def Move_shape_x(self):
        pass
        try:
            move_distance=float(self.lineEdit_8.text())
            T = gp_Trsf()
            T.SetTranslation(gp_Vec(move_distance, 0, 0))
            loc = TopLoc_Location(T)
            self.select_shape.Location(loc)
            show=self.canva._display.DisplayShape(self.select_shape)
            self.canva._display.Context.Remove(self.show_list[0], True)
            self.canva._display.Context.UpdateCurrentViewer()
            self.show_list = show
            self.canva._display.Repaint()

        except:
            pass

    def Move_shape_y(self):
        pass
        try:
            move_distance = float(self.lineEdit_12.text())
            print(move_distance)
            T = gp_Trsf()
            T.SetTranslation(gp_Vec(0, move_distance, 0))
            loc = TopLoc_Location(T)
            self.select_shape.Location(loc)
            show = self.canva._display.DisplayShape(self.select_shape)
            self.canva._display.Context.Remove(self.show_list[0], True)
            self.canva._display.Context.UpdateCurrentViewer()
            self.show_list = show
            self.canva._display.Repaint()
        except:
            pass
    def Move_shape_z(self):
        pass
        try:
            move_distance = float(self.lineEdit_15.text())
            print(move_distance)
            T = gp_Trsf()
            T.SetTranslation(gp_Vec(0, 0, move_distance))
            loc = TopLoc_Location(T)
            self.select_shape.Location(loc)
            show = self.canva._display.DisplayShape(self.select_shape)

            self.canva._display.Context.Remove(self.show_list[0], True)
            self.canva._display.Context.UpdateCurrentViewer()
            self.show_list = show
        except:
            pass

            self.canva._display.Repaint()

    def line_clicked(self, shp, *kwargs):
        try:
            if self.measure_signal == 1:
                self.canva._display.SetSelectionMode(TopAbs_SOLID)
                for i in shp:  # 获取最大尺寸
                    self.select_shape = i
                self.measure_signal = 0

        except:
            pass

    def Quit(self):#退出
        self.close()

    def Vision(self):#版本显示
        pass
    def about(self):
        self.version.show()








# following couple of lines is a tweak to enable ipython --gui='qt'
if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()  # checks if QApplication already exists
    if not app:  # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)
    #启动界面
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap("Pic\\setup_pic.jpg"))#启动图片设置
    splash.show()
    splash.showMessage("软件启动中......")
    time.sleep(0.5)
    #--------------------
    win = Mywindown()
    win_vision=Vision()
    win.vision.triggered.connect(win_vision.show)
    win.show()
    win.centerOnScreen()
    win.canva.InitDriver()
    win.resize(size[0], size[1])
    win.canva.qApp = app

    display = win.canva._display
    display.display_triedron()
    display.register_select_callback(win.line_clicked)
    if background_gradient_color1 and background_gradient_color2:
    # background gradient
        display.set_bg_gradient_color(background_gradient_color1, background_gradient_color2)

    win.raise_()  # make the application float to the top
    splash.finish(win)
    app.exec_()