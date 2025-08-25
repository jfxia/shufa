import os
import logging
from PIL import Image
import argparse
from pathlib import Path
from typing import List

class ImageCleaner:
    def __init__(self, log_file: str = "image_cleanup.log"):
        self.log_file = log_file
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
    def is_animated_gif(self, image_path: str) -> bool:
        """检查是否为动态GIF"""
        try:
            with Image.open(image_path) as img:
                if img.format == 'GIF':
                    # 检查是否有多个帧
                    try:
                        img.seek(1)
                        return True
                    except EOFError:
                        return False
            return False
        except Exception as e:
            logging.debug(f"无法检查GIF动画: {image_path}, 错误: {e}")
            return False
            
    def is_vector_image(self, image_path: str) -> bool:
        """检查是否为矢量图（基于文件扩展名）"""
        vector_extensions = {'.svg', '.ai', '.eps', '.pdf'}
        return Path(image_path).suffix.lower() in vector_extensions
        
    def is_small_image(self, image_path: str) -> bool:
        """检查图片尺寸是否小于50px（只针对可读取的位图）"""
        # 如果是矢量图，跳过尺寸检查
        if self.is_vector_image(image_path):
            return False
            
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return width < 50 or height < 50
        except Exception as e:
            logging.debug(f"无法检查图片尺寸: {image_path}, 错误: {e}")
            return False
            
    def get_image_info(self, image_path: str) -> dict:
        """获取图片的详细信息"""
        info = {
            'path': image_path,
            'is_animated_gif': False,
            'is_vector': False,
            'is_small': False,
            'reasons': []
        }
        
        file_ext = Path(image_path).suffix.lower()
        
        # 检查矢量图
        if self.is_vector_image(image_path):
            info['is_vector'] = True
            info['reasons'].append("矢量图")
        
        # 检查动态GIF（只对GIF文件）
        if file_ext == '.gif':
            if self.is_animated_gif(image_path):
                info['is_animated_gif'] = True
                info['reasons'].append("动态GIF")
        
        # 检查小尺寸图片（只对可读取的位图）
        if not info['is_vector']:  # 矢量图不检查尺寸
            if self.is_small_image(image_path):
                info['is_small'] = True
                info['reasons'].append("小尺寸图片")
        
        return info
    
    def process_directory(self, directory_path: str) -> List[str]:
        """处理目录中的所有图片文件"""
        target_images = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', 
                               '.svg', '.ai', '.eps', '.pdf', '.webp', '.ico'}
        
        print(f"开始扫描目录: {directory_path}")
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in supported_extensions:
                    image_info = self.get_image_info(file_path)
                    
                    if image_info['reasons']:
                        target_images.append(file_path)
                        reasons_str = ", ".join(image_info['reasons'])
                        logging.info(f"找到目标图片: {file_path} - 原因: {reasons_str}")
                        print(f"找到: {file} - {reasons_str}")
        
        return target_images
    
    def save_results(self, image_list: List[str], output_file: str = None):
        """保存结果到文件"""
        if output_file is None:
            output_file = self.log_file.replace('.log', '_results.txt')
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("符合要求的图片列表:\n")
            f.write("=" * 50 + "\n")
            for img_path in image_list:
                f.write(f"{img_path}\n")
                
        logging.info(f"结果已保存到: {output_file}, 共找到 {len(image_list)} 个符合要求的图片")
        print(f"结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='图片清理程序 - 识别特定类型的图片')
    parser.add_argument('directory', help='要扫描的目录路径')
    parser.add_argument('--output', '-o', default='cleaned_images.log', 
                       help='输出日志文件名')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在!")
        return
    
    cleaner = ImageCleaner(args.output)
    
    target_images = cleaner.process_directory(args.directory)
    
    cleaner.save_results(target_images)
    
    print(f"扫描完成! 共找到 {len(target_images)} 个符合要求的图片")

if __name__ == "__main__":
    main()