# main.py
import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image, ImageOps, ImageEnhance
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import hashlib
import logging
from PIL.ExifTags import TAGS

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置部分 ---
MODEL_PATH = "best_model.pth"
MAP_PATH = "char_map.json"

# --- 全局初始化模型（启动时加载一次） ---
class Recognizer:
    def __init__(self, model_path: str = MODEL_PATH, map_path: str = MAP_PATH):
        if not os.path.exists(model_path) or not os.path.exists(map_path):
            raise FileNotFoundError(
                f"模型文件 '{model_path}' 或字符映射表 '{map_path}' 不存在，"
                "请先运行 train_model.py 进行模型训练。"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载字符映射
        with open(map_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        num_classes = len(self.char_to_idx)

        # 构建并加载模型
        self.model = self._get_model(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_model(self, num_classes: int) -> nn.Module:
        model = resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        return model

    def preprocess_image(self, image_bytes: bytes, user_agent: str = "") -> Image.Image:
        """统一的图像预处理，特别处理移动设备上传的图片"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # 记录原始图像信息用于调试
            logger.info(f"原始图像 - 格式: {image.format}, 模式: {image.mode}, 尺寸: {image.size}")
            
            # 处理EXIF方向信息（手机照片常有旋转问题）
            try:
                exif = image._getexif()
                if exif:
                    for tag, value in exif.items():
                        decoded = TAGS.get(tag, tag)
                        if decoded == 'Orientation':
                            if value == 3:
                                image = image.rotate(180, expand=True)
                            elif value == 6:
                                image = image.rotate(270, expand=True)
                            elif value == 8:
                                image = image.rotate(90, expand=True)
                            break
            except Exception as e:
                logger.warning(f"EXIF处理失败: {e}")
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 检测是否为移动设备并应用增强处理
            is_mobile = any(mobile_indicator in user_agent.lower() 
                           for mobile_indicator in ['mobile', 'iphone', 'android', 'ipad'])
            
            if is_mobile:
                logger.info("检测到移动设备，应用增强预处理")
                # 增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                # 轻微锐化
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            raise ValueError(f"无法处理图片: {e}")

    def assess_image_quality(self, image: Image.Image) -> Dict:
        """评估图像质量"""
        # 转换为numpy数组进行处理
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # 计算清晰度（拉普拉斯方差）
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算亮度和对比度
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        quality_info = {
            'clarity': float(clarity),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_acceptable': clarity > 50 and 30 < brightness < 220
        }
        
        logger.info(f"图像质量评估: {quality_info}")
        return quality_info

    def recognize(self, image_bytes: bytes, top_k: int = 5, user_agent: str = "") -> List[Dict[str, str]]:
        """
        识别上传的图像。

        Args:
            image_bytes: 图片的二进制数据。
            top_k: 返回前k个结果。
            user_agent: 用户代理字符串，用于设备检测

        Returns:
            一个字典列表，每个字典包含 `char` 和 `prob` 键。
        """
        # 记录图像哈希用于调试
        image_hash = hashlib.md5(image_bytes).hexdigest()
        logger.info(f"处理图像 - 哈希: {image_hash}, 设备: {user_agent}")

        # 预处理图像
        image = self.preprocess_image(image_bytes, user_agent)
        
        # 评估图像质量
        quality_info = self.assess_image_quality(image)
        if not quality_info['is_acceptable']:
            logger.warning(f"图像质量可能影响识别: {quality_info}")

        # 应用模型预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        top_probs_np = top_probs.cpu().numpy().flatten()
        top_indices_np = top_indices.cpu().numpy().flatten()

        for i in range(top_k):
            char_idx = top_indices_np[i]
            char_name = self.idx_to_char.get(char_idx, '?')
            probability = top_probs_np[i]
            results.append({
                'char': char_name,
                'prob': f'{probability:.2%}'
            })

        logger.info(f"识别结果: {results}")
        return results

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="汉字书法字体识别",
    description="上传一张汉字图片，返回识别出的汉字及其置信度。",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化识别器（全局单例）
try:
    recognizer = Recognizer()
    logger.info("模型加载成功，服务已启动")
except FileNotFoundError as e:
    logger.error(f"启动失败: {e}")
    recognizer = None

# --- API 路由 ---
@app.post("/upload", response_model=List[Dict[str, str]])
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    上传图片进行汉字识别。

    - **file**: 需要识别的图片文件 (jpg, png, etc.)
    """
    if not recognizer:
        raise HTTPException(
            status_code=503,
            detail="服务暂不可用，模型文件未找到。"
        )

    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上传的文件必须是图片格式。")

    try:
        image_bytes = await file.read()
        user_agent = request.headers.get('User-Agent', '')
        results = recognizer.recognize(image_bytes, top_k=5, user_agent=user_agent)
        return results
    except ValueError as ve:
        logger.error(f"图片处理错误: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"识别错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

# 调试接口
@app.post("/debug_upload")
async def debug_upload(request: Request, file: UploadFile = File(...)):
    """调试接口，返回上传图片的详细信息"""
    try:
        image_bytes = await file.read()
        user_agent = request.headers.get('User-Agent', '')
        
        # 使用recognizer的预处理方法来分析图片
        image = recognizer.preprocess_image(image_bytes, user_agent)
        quality_info = recognizer.assess_image_quality(image)
        
        debug_info = {
            'file_size': len(image_bytes),
            'file_hash': hashlib.md5(image_bytes).hexdigest(),
            'image_size': image.size,
            'image_mode': image.mode,
            'quality_assessment': quality_info,
            'user_agent': user_agent
        }
        
        return JSONResponse(content=debug_info)
    except Exception as e:
        logger.error(f"调试接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="web")

@app.get("/")
async def root():
    return {"message": "欢迎使用汉字书法识别模型，请使用 POST /upload 接口上传图片。"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    status = "healthy" if recognizer else "unhealthy"
    return {
        "status": status,
        "model_loaded": recognizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
