# main.py
import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io

# --- 配置部分 ---
MODEL_PATH = "best_model.pth"
MAP_PATH = "char_map.json"
DATA_DIR = ""  # 原始代码中未使用，保留为空


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

    def recognize(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, str]]:
        """
        识别上传的图像。

        Args:
            image_bytes: 图片的二进制数据。
            top_k: 返回前k个结果。

        Returns:
            一个字典列表，每个字典包含 `char` 和 `prob` 键。
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            raise ValueError(f"无法打开图片: {e}")

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

        return results

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="汉字书法字体识别 API",
    description="上传一张汉字图片，返回识别出的汉字及其置信度。",
    version="1.0.0"
)

# 添加 CORS 中间件（可选，方便前端调试）
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
except FileNotFoundError as e:
    # 如果模型文件不存在，启动时报错
    print(f"[ERROR] 启动失败: {e}")
    recognizer = None

# --- API 路由 ---
@app.post("/upload", response_model=List[Dict[str, str]])
async def upload_image(file: UploadFile = File(...)):
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
        results = recognizer.recognize(image_bytes, top_k=5)
        return results
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="web")

@app.get("/")
async def root():
    return {"message": "欢迎使用汉字书法识别 API！请使用 POST /upload 接口上传图片。"}


# --- 主程序入口 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)