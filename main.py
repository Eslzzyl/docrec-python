import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from ill_rec import rec_ill
from backend import GeoTr_Seg, reload_model
from imageviewer import ImageViewer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model_path = './model_pretrained/'
model_path = '../DocRec/model_pretrained/'

base_font = QFont('sans-serif', 12)

class MyMainWindow(QMainWindow):
    image_width = 500
    image_height = 700

    image_list = []
    image_index = 0

    GeoTr_Seg_model = GeoTr_Seg().to(device=device)
    is_model_loaded = False

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        if not os.path.exists('./geo_rec'):  # create save path
            os.mkdir('./geo_rec')
        if not os.path.exists('./ill_rec'):  # create save path
            os.mkdir('./ill_rec')
        # 选择按钮
        self.choose_img_button = QPushButton()
        self.choose_img_button.setFont(base_font)
        self.choose_img_button.setText('选择图片')
        # self.choose_img_button.setFixedSize(QFontMetrics(base_font).boundingRect('选择图片').size() + QSize(20, 20))
        self.choose_folder_button = QPushButton()
        self.choose_folder_button.setFont(base_font)
        self.choose_folder_button.setText('选择文件夹')
        # self.choose_folder_button.setFixedSize(QFontMetrics(base_font).boundingRect('选择文件夹').size() + QSize(20, 20))
        self.process_button = QPushButton()
        self.process_button.setFont(base_font)
        self.process_button.setText('开始处理')
        self.previous_button = QPushButton()
        self.previous_button.setFont(base_font)
        self.previous_button.setText('上一张')
        self.next_button = QPushButton()
        self.next_button.setFont(base_font)
        self.next_button.setText('下一张')
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        # 操作分组框
        self.operation_group_box = QGroupBox('操作')
        self.operation_group_box.setFont(base_font)
        self.operation_group_layout = QHBoxLayout()
        self.operation_group_layout.addWidget(self.choose_img_button)
        self.operation_group_layout.addWidget(self.choose_folder_button)
        self.operation_group_layout.addWidget(self.process_button)
        self.operation_group_layout.addWidget(self.previous_button)
        self.operation_group_layout.addWidget(self.next_button)
        self.operation_group_layout.setAlignment(Qt.AlignLeft)
        self.operation_group_box.setLayout(self.operation_group_layout)

        # 6个标签显示图片
        self.current_image_label = QLabel()
        self.current_image_label.setText('当前尚未加载图片')
        self.origin_img_label = QLabel()
        self.geo_img_label = QLabel()
        self.ill_img_label = QLabel()
        self.origin_img_label.setFixedWidth(self.image_width)
        self.origin_img_label.setMinimumHeight(self.image_height)
        self.geo_img_label.setFixedWidth(self.image_width)
        self.geo_img_label.setMinimumHeight(self.image_height)
        self.ill_img_label.setFixedWidth(self.image_width)
        self.ill_img_label.setMinimumHeight(self.image_height)
        self.origin_label = QLabel('原图')
        self.origin_label.setFont(base_font)
        self.geo_label = QLabel('几何矫正结果')
        self.geo_label.setFont(base_font)
        self.ill_label = QLabel('光照修复结果')
        self.ill_label.setFont(base_font)
        self.current_image_label.setAlignment(Qt.AlignCenter)
        self.origin_label.setAlignment(Qt.AlignCenter)
        self.geo_label.setAlignment(Qt.AlignCenter)
        self.ill_label.setAlignment(Qt.AlignCenter)

        self.origin_pixmap = QPixmap()
        self.geo_pixmap = QPixmap()
        self.ill_pixmap = QPixmap()

        # 设置中间布局
        self.image_group_layout = QGridLayout()
        self.image_group_layout.addWidget(self.current_image_label, 0, 0, 1, 3)
        self.image_group_layout.addWidget(self.origin_img_label, 1, 0)
        self.image_group_layout.addWidget(self.geo_img_label, 1, 1)
        self.image_group_layout.addWidget(self.ill_img_label, 1, 2)
        self.image_group_layout.addWidget(self.origin_label, 2, 0)
        self.image_group_layout.addWidget(self.geo_label, 2, 1)
        self.image_group_layout.addWidget(self.ill_label, 2, 2)

        #图片分组框
        self.image_group_box = QGroupBox('图片展示')
        self.image_group_box.setFont(base_font)
        self.image_group_box.setLayout(self.image_group_layout)

        # 日志区
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(200)
        self.log_area.setReadOnly(True)         # 设置不可编辑
        self.clear_log_button = QPushButton()
        self.clear_log_button.setText('清空日志')
        self.log_area_layout = QHBoxLayout()
        self.log_area_layout.addWidget(self.log_area)
        self.log_area_layout.addWidget(self.clear_log_button)
        self.log_group_box = QGroupBox('日志')   # 添加一个QGroupBox
        self.log_group_box.setFont(base_font)
        self.clear_log_button.setFont(base_font)
        self.log_group_box.setLayout(self.log_area_layout)  # 设置QGroupBox的布局为之前的QGridLayout

        # 设置主布局
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.operation_group_box)
        self.main_layout.addWidget(self.image_group_box)
        self.main_layout.addWidget(self.log_group_box)
        # 设置主Widget
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 设置状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('就绪')
        self.setStatusBar(self.status_bar)

        # 设置菜单栏
        self.init_menu()

        # 绑定信号槽
        self.choose_img_button.clicked.connect(self.on_image_choose)
        self.choose_folder_button.clicked.connect(self.on_folder_choose)
        self.process_button.clicked.connect(self.on_rec)
        self.previous_button.clicked.connect(self.on_previous)
        self.next_button.clicked.connect(self.on_next)
        self.clear_log_button.clicked.connect(self.log_area.clear)

        # 在图片上双击，显示原图
        self.origin_img_label.mouseDoubleClickEvent = lambda _: self.on_img_double_click(self.origin_pixmap, self.get_current_image_name())
        self.geo_img_label.mouseDoubleClickEvent = lambda _: self.on_img_double_click(self.geo_pixmap, self.get_current_image_name())
        self.ill_img_label.mouseDoubleClickEvent = lambda _: self.on_img_double_click(self.ill_pixmap, self.get_current_image_name())

        if check_model() == 0:
            self.log('模型位置：' + model_path)
        else:
            self.log('模型文件不存在：' + model_path)
            self.choose_img_button.setEnabled(False)
            self.choose_folder_button.setEnabled(False)
        
        self.setWindowTitle('DocRec文档矫正')
        self.show()
    
    def get_current_image_name(self):
        return self.image_list[self.image_index]

    def init_menu(self):
        self.menu_bar = self.menuBar()
        # 修改菜单栏的字体
        self.menu_bar.setFont(base_font)
        # 设置文件菜单
        self.file_menu = self.menu_bar.addMenu('文件')
        self.file_menu.setFont(base_font)
        self.file_menu.addAction('选择图片...', self.on_image_choose)
        self.file_menu.addAction('选择文件夹...', self.on_folder_choose)
        self.file_menu.addAction('设置...', self.on_setting)
        self.file_menu.addAction('退出', sys.exit)
        # 设置编辑菜单
        self.edit_menu = self.menu_bar.addMenu('编辑')
        self.edit_menu.setFont(base_font)
        self.edit_menu.addAction('清空日志', self.log_area.clear)
        self.edit_menu.addAction('上一张', self.on_previous)
        self.edit_menu.addAction('下一张', self.on_next)
        # 程序启动时，上一张和下一张菜单不可用
        self.edit_menu.actions()[1].setEnabled(False)
        self.edit_menu.actions()[2].setEnabled(False)
        # 设置关于菜单
        self.about_menu = self.menu_bar.addMenu('帮助')
        self.about_menu.setFont(base_font)
        self.about_menu.addAction('关于DocRec...', self.on_about)

    def log(self, text: str):
        # 添加系统时间
        self.log_area.append(time.strftime('[%H:%M:%S] ', time.localtime(time.time())) + text)
        self.log_area.moveCursor(QTextCursor.End)

    def setLabelImage(self, label: QLabel, pixmap: QPixmap):
        # 若图片宽度大于长度，则顺时针旋转90度
        if pixmap.width() > pixmap.height():
            pixmap = pixmap.transformed(QTransform().rotate(90))
        # 若图片宽度大于self.image_width，则按比例缩放到self.image_width
        if pixmap.width() > self.image_width:
            pixmap = pixmap.scaledToWidth(self.image_width)
        label.setPixmap(pixmap)
    
    def on_image_choose(self):
        # 选择图片
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图片文件(*.jpg *.png)')
        self.status_bar.showMessage('正在加载：' + file_name)
        if file_name:
            self.log_area.clear()
            self.image_list.clear()
            self.origin_pixmap.load(file_name)
            self.setLabelImage(self.origin_img_label, self.origin_pixmap)
            self.log('加载图片：' + file_name)
            self.image_list.append(file_name)
            self.image_index = 0
            self.status_bar.showMessage('就绪')
    
    def on_folder_choose(self):
        # 选择文件夹
        folder_name = QFileDialog.getExistingDirectory(self, '选择文件夹', '')
        self.status_bar.showMessage('正在加载：' + folder_name)
        if folder_name:
            self.log_area.clear()
            self.image_list.clear()
            self.image_list = os.listdir(folder_name)
            for i in self.image_list:
                if i.split('.')[-1] not in ['jpg', 'png']:
                    self.image_list.remove(i)
            self.image_list = [os.path.join(folder_name, x).replace('\\', '/') for x in self.image_list]
            self.log('加载文件夹：' + folder_name)
            self.log('共加载了' + str(len(self.image_list)) + '张图片')
            self.image_index = 0
            # 如果加载图片大于1张，则启用下一张按钮
            if len(self.image_list) > 1:
                self.next_button.setEnabled(True)
                self.edit_menu.actions()[2].setEnabled(True)
            origin_name = self.image_list[self.image_index].split('/')[-1]
            self.status_bar.showMessage('正在加载：' + origin_name)
            self.origin_pixmap.load(self.image_list[self.image_index])
            self.setLabelImage(self.origin_img_label, self.origin_pixmap)
            self.log('当前图片：' + origin_name)
            geo_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_geo.png'
            geo_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'geo_rec/' + geo_name
            ill_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_ill.png'
            ill_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'ill_rec/' + ill_name
            if os.path.exists(geo_path):
                self.geo_pixmap.load(geo_path)
                self.setLabelImage(self.geo_img_label, self.geo_pixmap)
                self.log('已找到现存的几何矫正结果，已加载。如果要重新矫正，请点击“矫正”按钮。')
            if os.path.exists(ill_path):
                self.ill_pixmap.load(ill_path)
                self.setLabelImage(self.ill_img_label, self.ill_pixmap)
                self.log('已找到现存的光照修复结果，已加载。如果要重新矫正，请点击“矫正”按钮。')
            self.status_bar.showMessage('就绪')
    
    def on_img_double_click(self, pixmap: QPixmap, name: str):
        # 开辟新窗口显示图片
        self.imageview = ImageViewer()
        self.imageview.setPixmap(pixmap, name)
        self.imageview.show()

    def on_about(self):
        about = QMessageBox()
        about.setWindowTitle('关于DocRec')
        about.setStandardButtons(QMessageBox.Ok)
        about.setDefaultButton(QMessageBox.Ok)
        about.setIcon(QMessageBox.Information)
        about.setText('DocRec是一个基于深度学习技术的扭曲文档矫正工具，支持曲线边界矫正。')
        about.exec_()

    def on_setting(self):
        setting_widget = SettingDialog()
        setting_widget.exec()
        self.log_area.clear()
        self.log('由于设置更改，日志已清空')
        self.log('模型位置：' + model_path)
    
    def on_previous(self):
        if self.image_index > 0:
            self.image_index -= 1
            if self.image_index == 0:
                self.previous_button.setEnabled(False)
                self.edit_menu.actions()[1].setEnabled(False)
            origin_name = self.image_list[self.image_index].split('/')[-1]
            self.status_bar.showMessage('正在加载：' + origin_name)
            geo_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_geo.png'
            geo_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'geo_rec/' + geo_name
            ill_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_ill.png'
            ill_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'ill_rec/' + ill_name
            self.origin_pixmap.load(self.image_list[self.image_index])
            self.setLabelImage(self.origin_img_label, self.origin_pixmap)
            # self.log('加载图片：' + self.image_list[self.image_index])
            if os.path.exists(geo_path):
                self.geo_pixmap.load(geo_path)
                self.setLabelImage(self.geo_img_label, self.geo_pixmap)
            if os.path.exists(ill_path):
                self.ill_pixmap.load(ill_path)
                self.setLabelImage(self.ill_img_label, self.ill_pixmap)
            # self.log('已找到现存的矫正结果，已加载。如果要重新矫正，请点击“开始处理”按钮。')
            if self.next_button.isEnabled() == False:
                self.next_button.setEnabled(True)
                self.edit_menu.actions()[2].setEnabled(True)
            self.log('当前图片：' + origin_name)
            self.status_bar.showMessage('就绪')
        else:
            self.previous_button.setEnabled(False)
            self.edit_menu.actions()[1].setEnabled(False)
    
    def on_next(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            if self.image_index == len(self.image_list) - 1:
                self.next_button.setEnabled(False)
                self.edit_menu.actions()[2].setEnabled(False)
            origin_name = self.image_list[self.image_index].split('/')[-1]
            self.status_bar.showMessage('正在加载：' + origin_name)
            geo_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_geo.png'
            geo_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'geo_rec/' + geo_name
            ill_name = self.image_list[self.image_index].split('/')[-1].split('.')[0] + '_ill.png'
            ill_path = self.image_list[self.image_index].replace('distorted/' + origin_name, '') + 'ill_rec/' + ill_name
            self.origin_pixmap.load(self.image_list[self.image_index])
            self.setLabelImage(self.origin_img_label, self.origin_pixmap)
            # self.log('加载图片：' + self.image_list[self.image_index])
            if os.path.exists(geo_path):
                self.geo_pixmap.load(geo_path)
                self.setLabelImage(self.geo_img_label, self.geo_pixmap)
            if os.path.exists(ill_path):
                self.ill_pixmap.load(ill_path)
                self.setLabelImage(self.ill_img_label, self.ill_pixmap)
            # self.log('已找到现存的矫正结果，已加载。如果要重新矫正，请点击“矫正”按钮。')
            if self.previous_button.isEnabled() == False:
                self.previous_button.setEnabled(True)
                self.edit_menu.actions()[1].setEnabled(True)
            self.log('当前图片：' + origin_name)
            self.status_bar.showMessage('就绪')
        else:
            self.next_button.setEnabled(False)
            self.edit_menu.actions()[2].setEnabled(False)

    def on_rec(self):
        if not self.is_model_loaded:
            self.status_bar.showMessage('正在加载模型')
            self.log('正在加载模型')
            reload_model(self.GeoTr_Seg_model, model_path)
            self.GeoTr_Seg_model.eval()
            self.is_model_loaded = True

        for image_path in self.image_list:
            name = image_path.split('.')[-2]

            print('开始处理：', image_path)
            print('读取图片：', end='')
            image_original = np.array(Image.open(image_path))[:, :, :3] / 255.
            print('Done. Image resolution: ', image_original.shape)
            h, w, _ = image_original.shape

            image = cv2.resize(image_original, (288, 288))
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float().unsqueeze(0)

            with torch.no_grad():
                bm = self.GeoTr_Seg_model(image.to(device=device))

                bm = bm.cpu()
                bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
                bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
                bm0 = cv2.blur(bm0, (3, 3))
                bm1 = cv2.blur(bm1, (3, 3))
                lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
                
                out = F.grid_sample(torch.from_numpy(image_original).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)

                image_geo = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)
                print('Done.')
                print('Saving geo images...', end='')
                cv2.imwrite('./geo_rec' + name + '_geo' + '.png', image_geo)
                print('Done.')

                ill_save_path = 'ill_rec' + name + '_ill' + '.png'
                print('Illumination correction working...', end='')
                rec_ill(image_geo, ill_save_path)
                print('Done.')
        print('Done: ', image_path + '\n')

        self.log('开始矫正')
        self.status_bar.showMessage('正在矫正')
        self.rec_pixmap = self.origin_pixmap
        self.setLabelImage(self.rec_img_label, self.rec_pixmap)
        self.log('矫正完成')
        self.status_bar.showMessage('就绪')

class SettingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        path_label = QLabel('模型位置：')
        path_label.setFont(base_font)
        path_line_edit = QLineEdit()
        path_line_edit.setReadOnly(True)
        path_line_edit.setFont(base_font)
        path_line_edit.setFixedWidth(200)
        path_line_edit.setText(model_path)
        path_button = QPushButton('选择...')
        path_button.setFont(base_font)
        path_button.clicked.connect(self.on_model_choose)
        path_line_edit.setText(model_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(path_label)
        path_layout.addWidget(path_line_edit)
        path_layout.addWidget(path_button)

        # 在这里添加后续的设置选项

        close_button = QPushButton('关闭')
        close_button.setFont(base_font)
        close_button.clicked.connect(self.close)
        operate_layout = QHBoxLayout()
        operate_layout.addWidget(close_button)

        setting_layout = QVBoxLayout()
        setting_layout.addLayout(path_layout)
        setting_layout.addLayout(operate_layout)

        self.setWindowTitle('设置')
        self.setLayout(setting_layout)
        self.resize(600, 400)

    def on_model_choose(self):
        global model_path
        # 弹出选择模型文件的对话框
        dir_name = QFileDialog.getExistingDirectory(self, '选择模型位置', '', QFileDialog.ShowDirsOnly)
        if dir_name:
            if check_model() == 1:
                QMessageBox.critical(self, '错误', '未找到' + model_path + 'geotr.pth')
            if check_model() == 2:
                QMessageBox.critical(self, '错误', '未找到' + model_path + 'seg.pth')
            model_path = dir_name
        else:
            QMessageBox.critical(self, '错误', '未选择模型位置')

def check_model() -> int:
    if not os.path.exists(model_path + 'geotr.pth'):
        return 1
    if not os.path.exists(model_path + 'seg.pth'):
        return 2
    return 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    sys.exit(app.exec_())