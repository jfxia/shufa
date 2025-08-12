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

# SE注意力模块 (与训练代码一致)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 双任务识别模型
class DMTLRecognizer:
    def __init__(self, model_path='best_model.pth', char_map_path='char_map.json', 
                 style_map_path='style_map.json', data_dir=''):
        # 检查文件是否存在
        if not all(map(os.path.exists, [model_path, char_map_path, style_map_path])):
            raise FileNotFoundError("模型或映射文件不存在，请先训练模型")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        
        # 加载字符和风格映射
        with open(char_map_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        with open(style_map_path, 'r', encoding='utf-8') as f:
            self.style_to_idx = json.load(f)
        self.idx_to_style = {v: k for k, v in self.style_to_idx.items()}
        
        # 初始化模型
        self.model = self._build_model(len(self.char_to_idx), len(self.style_to_idx))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _build_model(self, num_chars, num_styles):
        # 创建与训练一致的模型结构
        class DMTLModel(nn.Module):
            def __init__(self, num_chars, num_styles):
                super(DMTLModel, self).__init__()
                self.base_model = resnet50(weights=None)
                num_ftrs = self.base_model.fc.in_features
                self.base_model.fc = nn.Identity()
                self.se = SEBlock(2048)
                self.char_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_ftrs, num_chars)
                )
                self.style_head = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_ftrs, num_styles)
                )
            
            def forward(self, x):
                features = self.base_model(x)
                features = features.unsqueeze(-1).unsqueeze(-1)
                features = self.se(features)
                features = features.squeeze(-1).squeeze(-1)
                char_output = self.char_head(features)
                style_output = self.style_head(features)
                return char_output, style_output
        
        return DMTLModel(num_chars, num_styles)
    
    def recognize(self, image_path, top_k=5):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"无法打开图片: {e}")
        
        # 预处理并预测
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            char_output, style_output = self.model(image_tensor)
            char_probs = torch.nn.functional.softmax(char_output, dim=1)
            style_probs = torch.nn.functional.softmax(style_output, dim=1)
            
            # 获取字符的top-k结果
            top_char_probs, top_char_indices = torch.topk(char_probs, top_k)
            char_results = []
            for i in range(top_k):
                char_idx = top_char_indices[0][i].item()
                char_name = self.idx_to_char.get(char_idx, '?')
                prob = top_char_probs[0][i].item()
                char_results.append({
                    'char': char_name,
                    'prob': f'{prob:.2%}'
                })
            
            # 获取风格结果
            _, top_style_idx = torch.max(style_probs, 1)
            style_idx = top_style_idx[0].item()
            style_name = self.idx_to_style.get(style_idx, '?')
            style_prob = style_probs[0][style_idx].item()
            style_result = {
                'style': style_name,
                'prob': f'{style_prob:.2%}'
            }
        
        return char_results, style_result

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('汉字书法识别器 - DMTL框架')
        self.resize(1000, 800)
        
        try:
            self.recognizer = DMTLRecognizer(data_dir='chinese_fonts')
        except FileNotFoundError as e:
            QMessageBox.critical(self, "错误", str(e))
            QTimer.singleShot(100, self.close)
            return
        
        self.init_ui()
        self.image_path = None
    
    def init_ui(self):
        # 主窗口设置
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 图像显示区域
        self.image_label = QLabel('请选择书法图片')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setStyleSheet('border: 2px dashed #aaa; font-size: 16px;')
        layout.addWidget(self.image_label)
        
        # 选择按钮
        select_btn = QPushButton('选择图片')
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A86E8;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3A76D8;
            }
            QPushButton:pressed {
                background-color: #2A66C8;
            }
        """)
        select_btn.clicked.connect(self.select_image)
        layout.addWidget(select_btn)
        
        # 风格识别结果
        style_group = QGroupBox("字体风格识别结果")
        style_layout = QVBoxLayout()
        
        self.style_label = QLabel('尚未识别')
        self.style_label.setAlignment(Qt.AlignCenter)
        self.style_label.setStyleSheet('font-size: 18px; font-weight: bold; color: #C46210;')
        style_layout.addWidget(self.style_label)
        
        self.style_prob = QLabel('')
        self.style_prob.setAlignment(Qt.AlignCenter)
        self.style_prob.setStyleSheet('font-size: 16px; color: #444;')
        style_layout.addWidget(self.style_prob)
        
        style_group.setLayout(style_layout)
        layout.addWidget(style_group)
        
        # 字符识别结果
        char_group = QGroupBox("字符识别结果 (Top-5)")
        char_layout = QVBoxLayout()
        
        self.result_table = QTableWidget(5, 2)
        self.result_table.setHorizontalHeaderLabels(['字符', '置信度'])
        self.result_table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #E0E0E0;
                color: #333;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setStyleSheet("font-size: 16px;")
        
        # 初始化表格
        for i in range(5):
            self.result_table.setItem(i, 0, QTableWidgetItem(""))
            self.result_table.setItem(i, 1, QTableWidgetItem(""))
            self.result_table.item(i, 0).setTextAlignment(Qt.AlignCenter)
            self.result_table.item(i, 1).setTextAlignment(Qt.AlignCenter)
        
        char_layout.addWidget(self.result_table)
        char_group.setLayout(char_layout)
        layout.addWidget(char_group)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('准备就绪')
    
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择书法图片', '', 
            '图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)'
        )
        
        if not path:
            return
        
        self.image_path = path
        pixmap = QPixmap(path)
        
        # 调整图片大小以适应显示区域
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(), 
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
            self.recognize()
    
    def recognize(self):
        if not self.image_path:
            return
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 执行识别
            char_results, style_result = self.recognizer.recognize(self.image_path)
            
            # 更新风格结果
            self.style_label.setText(f"识别字体: {style_result['style']}")
            self.style_prob.setText(f"置信度: {style_result['prob']}")
            
            # 更新字符结果
            for i, result in enumerate(char_results):
                self.result_table.setItem(i, 0, QTableWidgetItem(result['char']))
                self.result_table.setItem(i, 1, QTableWidgetItem(result['prob']))
            
            self.status_bar.showMessage(
                f"识别完成 - 字符: {char_results[0]['char']}, 风格: {style_result['style']}", 
                5000
            )
            
        except Exception as e:
            QMessageBox.critical(self, "识别错误", str(e))
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())