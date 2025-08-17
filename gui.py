
import sys
import json
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Recognizer:
    def __init__(self, model_path='best_model.pth', map_path='char_map.json', data_dir=''):
        if not os.path.exists(model_path) or not os.path.exists(map_path):
            raise FileNotFoundError("模型文件或字符映射表不存在，请先运行 train_model.py 训练模型。")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir

        with open(map_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        num_classes = len(self.char_to_idx)

        self.model = self.get_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_model(self, num_classes):
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        return model

    def recognize(self, image_path, top_k=5):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"无法打开图片 {image_path}: {e}")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()
        for i in range(top_k):
            char_idx = top_indices[i]
            char_name = self.idx_to_char.get(char_idx, '?')
            probability = top_probs[i]
            #example_img_path = self.find_sample_image(char_name)
            results.append({
                'char': char_name,
                'prob': f'{probability:.2%}'
            })
        return results

'''
    def find_sample_image(self, char_name):
        if not self.data_dir:
            return None
        folder_path = os.path.join(self.data_dir, char_name)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    return os.path.join(folder_path, file)
        return None
'''

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('汉字书法识别器')
        self.resize(1000, 720)

        try:
            self.recognizer = Recognizer(data_dir='chinese_fonts')
        except FileNotFoundError as e:
            QMessageBox.critical(self, "错误", str(e))
            QTimer.singleShot(100, self.close)
            return

        self.init_ui()
        self.qimg_path = None

    def init_ui(self):
        w = QWidget(self)
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)

        self.input_lbl = QLabel('请选择一张汉字图片')
        self.input_lbl.setAlignment(Qt.AlignCenter)
        self.input_lbl.setMinimumSize(400, 300)
        self.input_lbl.setStyleSheet('border: 2px dashed #aaa;')
        layout.addWidget(self.input_lbl)

        self.select_btn = QPushButton('选择图片')
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D4; 
				height: 28px;
                color: white;                
                border: none;
				font-weight: bold;
                padding: 6px 15px;
                font-size: 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
        """)
        self.select_btn.clicked.connect(self.select_img)
        layout.addWidget(self.select_btn)

        self.result_table = QTableWidget(5, 2)
        self.result_table.setHorizontalHeaderLabels(['识别结果', '置信度'])
        self.result_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #e0e0e0;   
                color: black;                
                font-weight: bold;           
                border: 1px solid #c0c0c0;
                padding: 4px;
            }
        """)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.result_table)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('准备就绪。请先运行 train_model.py 进行模型训练。')

    def select_img(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图片 (*.jpg *.jpeg *.png *.gif *.bmp)')
        if not path:
            return
        self.qimg_path = path
        pixmap = QPixmap(path)
        self.input_lbl.setPixmap(pixmap.scaled(self.input_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.do_recognize()

    def do_recognize(self):
        if not self.qimg_path:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = self.recognizer.recognize(self.qimg_path)
            self.display_results(results)
            self.status_bar.showMessage(f"识别完成，最高匹配: {results[0]['char']}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "识别错误", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def display_results(self, results):
        self.result_table.setRowCount(len(results))
        for i, res in enumerate(results):
            self.result_table.setItem(i, 0, QTableWidgetItem(res['char']))
            self.result_table.setItem(i, 1, QTableWidgetItem(res['prob']))
            self.result_table.item(i, 0).setTextAlignment(Qt.AlignCenter)
            self.result_table.item(i, 1).setTextAlignment(Qt.AlignCenter)
'''
            if res['img'] and os.path.exists(res['img']):
                img_item = QLabel()
                pixmap = QPixmap(res['img'])
                pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_item.setPixmap(pixmap)
                img_item.setAlignment(Qt.AlignCenter)
                self.result_table.setCellWidget(i, 2, img_item)
            else:
                self.result_table.setItem(i, 2, QTableWidgetItem("无示例图"))
			'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("微软雅黑", 9)
    QApplication.setFont(font)
    win = MainWindow()
    if hasattr(win, 'recognizer'):
        win.show()
        sys.exit(app.exec_())
