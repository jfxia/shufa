import os
import re
import time
import json
import logging
import random
import requests
import sqlite3
from bs4 import BeautifulSoup
from threading import Thread, Lock
from threading import local
from queue import Queue
from urllib.parse import urljoin
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("font_scraper.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FontScraper:
    def __init__(self, output_dir="chinese_fonts", db_path="font_database.db", thread_count=2, min_interval=5.5):
        """
        初始化爬虫
        
        :param output_dir: 输出目录
        :param db_path: 数据库文件路径
        :param thread_count: 线程数量
        :param min_interval: 最小请求间隔（秒）
        """
        self.base_url = "https://sf.zdic.net"
        self.home_url = "https://sf.zdic.net"
        self.search_url = "https://sf.zdic.net/e/search/index.php"  # 正确的搜索URL
        self.font_types = {
            "zs": "篆书",  # 篆书
            "ls": "隶书",  # 隶书
            "ks": "楷书",  # 楷书
            "xs": "行书",  # 行书
            "cs": "草书"   # 草书
        }
        self.font_classids = {
            "zs": "31",  # 篆书
            "ls": "32",  # 隶书
            "ks": "33",  # 楷书
            "xs": "34",  # 行书
            "cs": "35"   # 草书
        }
        
        self.output_dir = output_dir
        self.db_path = db_path
        self.thread_count = thread_count
        self.min_interval = min_interval
        self.last_request_time = 0
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 线程安全的队列
        self.task_queue = Queue()
        self.lock = Lock()
        self.thread_local = local()
        # 创建会话，保持cookies
        self.session = requests.Session()
        
        # 初始化会话和数据库
        self._init_session()
        self._init_database()
    
    def _get_headers(self, referer=None):
        """获取请求头"""
        headers = {
            "User-Agent": random.choice([
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
            ]),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Content-Type": "application/x-www-form-urlencoded",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
        }
        
        if referer:
            headers["Referer"] = referer
            
        return headers
    
    def _init_database(self):
        """初始化数据库，创建表"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # 创建字体图片表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS font_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character TEXT NOT NULL,
                font_type TEXT NOT NULL,
                font_name TEXT NOT NULL,
                image_url TEXT NOT NULL,
                local_path TEXT NOT NULL,
                download_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'downloaded'
            )
            ''')
            
            # 创建已处理汉字表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS completed_characters (
                character TEXT PRIMARY KEY,
                completed_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                success_fonts INTEGER DEFAULT 0,
                total_fonts INTEGER DEFAULT 5
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_status (
                character TEXT PRIMARY KEY,
                status TEXT CHECK(status IN ('pending', 'completed', 'not_found')) DEFAULT 'pending',
                last_checked DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')       

            # 创建索引以提高查询速度
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_char_font ON font_images (character, font_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_url ON font_images (image_url)')
            
            self.conn.commit()
            logger.info(f"数据库初始化成功，路径: {self.db_path}")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            raise

    def _get_db_connection(self):
        """获取当前线程的数据库连接"""
        if not hasattr(self.thread_local, 'conn'):
            self.thread_local.conn = sqlite3.connect(self.db_path)
        return self.thread_local.conn

    def is_image_downloaded(self, image_url):
        """检查图片是否已下载过"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM font_images WHERE image_url = ?", (image_url,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.warning(f"检查图片下载状态失败: {str(e)}")
            return False

    def is_character_completed(self, character):
        """检查汉字是否已处理完成"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM completed_characters WHERE character = ?", (character,))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.warning(f"检查汉字处理状态失败: {str(e)}")
            return False
    
    def mark_character_completed(self, character, success_fonts, total_fonts=5):
        """标记汉字为已完成"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO completed_characters (character, success_fonts, total_fonts) VALUES (?, ?, ?)", 
                (character, success_fonts, total_fonts)
            )
            conn.commit()
            logger.debug(f"标记汉字 '{character}' 为已完成")
        except Exception as e:
            logger.error(f"标记汉字为已完成失败: {str(e)}")

    def is_character_not_found(self, character):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM character_status WHERE character = ? AND status = 'not_found'", (character,))
        return cursor.fetchone() is not None
    
    def mark_character_not_found(self, character):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO character_status (character, status) VALUES (?, 'not_found')",
            (character,)
        )
        conn.commit()
        logger.info(f"记录汉字【{character}】状态为：未找到字体")


    def save_image_record(self, character, font_type, font_name, image_url, local_path):
        """保存图片记录到数据库"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO font_images (character, font_type, font_name, image_url, local_path) VALUES (?, ?, ?, ?, ?)",
                (character, font_type, font_name, image_url, local_path)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"保存图片记录失败: {str(e)}")
            return False
    
    def get_characters(self, file_path="ciyun.txt"):
        """从文件中获取汉字列表，过滤重复项"""
        try:
            # 获取绝对路径以便于诊断
            abs_path = os.path.abspath(file_path)
            logger.info(f"尝试读取汉字文件: {abs_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"汉字文件 {abs_path} 不存在。请确保文件位于当前工作目录或提供正确的文件路径。")
                logger.error("当前工作目录: " + os.getcwd())
                return []
                
            # 尝试多种编码读取文件
            encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']  # 添加utf-8-sig处理BOM
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.debug(f"成功使用 {encoding} 编码读取文件")
                    break
                except UnicodeDecodeError:
                    continue
                    
            if content is None:
                logger.error(f"无法用常见编码读取文件 {file_path}")
                return []
            
            # 打印文件开头100字符用于调试
            logger.debug(f"文件前100字符: {content[:100].replace('\n', ' ')}")
            
            # 提取所有汉字
            characters = []
            line_count = 0
            valid_line_count = 0
            colon_found = 0
            
            for line in content.splitlines():
                line_count += 1
                stripped_line = line.strip()
                
                # 跳过空行和注释行
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                    
                valid_line_count += 1
                
                # 检查是否有冒号
                has_chinese_colon = '：' in stripped_line
                has_english_colon = ':' in stripped_line
                
                if has_chinese_colon or has_english_colon:
                    colon_found += 1
                    # 提取冒号后的部分
                    if has_chinese_colon:
                        char_part = stripped_line.split('：')[1]
                    else:
                        char_part = stripped_line.split(':')[1]
                    
                    # 提取所有汉字
                    chars = re.findall(r'[\u4e00-\u9fff]', char_part)
                    if chars:
                        characters.extend(chars)
                    else:
                        logger.debug(f"第{line_count}行有冒号但未找到汉字: {stripped_line}")
                else:
                    logger.debug(f"第{line_count}行无冒号，跳过: {stripped_line}")
            
            logger.info(f"共处理 {line_count} 行，其中 {valid_line_count} 行有效，{colon_found} 行包含冒号")
            
            # 去重并保留顺序
            seen = set()
            unique_chars = []
            for char in characters:
                if char not in seen:
                    seen.add(char)
                    unique_chars.append(char)
            
            if len(unique_chars) == 0:
                logger.error("未找到任何汉字！请检查 ciyun.txt 文件格式是否符合预期。")
                logger.error("预期格式示例：'一东：东同铜桐峒 etc.' 或 '二冬：冬农宗钟 etc.'")
                logger.error("请确认：")
                logger.error("1. 文件确实包含汉字")
                logger.error("2. 汉字位于冒号(：或:)之后")
                logger.error("3. 文件不是空文件")
            else:
                logger.info(f"共找到 {len(unique_chars)} 个唯一汉字需要处理")
            
            return unique_chars
        except Exception as e:
            logger.exception(f"读取汉字文件失败: {str(e)}")
            return []
    
    def _ensure_request_interval(self):
        """确保请求间隔大于最小间隔"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"请求间隔不足，等待 {sleep_time:.2f} 秒")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _init_session(self):
        """初始化会话，访问主页获取必要的cookies"""
        try:
            self._ensure_request_interval()
            logger.info("初始化会话: 访问主页获取cookies")
            
            # 先访问主页
            response = self.session.get(
                self.home_url, 
                headers=self._get_headers(),
                timeout=15
            )
            # 明确设置响应编码
            response.encoding = 'utf-8'
            logger.info(f"访问主页状态码: {response.status_code}")
            
            # 检查是否成功
            if response.status_code != 200:
                logger.error(f"无法访问主页，状态码: {response.status_code}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"初始化会话失败: {str(e)}")
            return False
    
    def _search_character(self, character):
        """搜索汉字，获取各字体页面URL"""
        self._ensure_request_interval()
        
        try:
            logger.info(f"搜索汉字: {character}")
            
            # 访问主页作为Referer
            home_response = self.session.get(
                self.home_url, 
                headers=self._get_headers(),
                timeout=15
            )
            home_response.encoding = 'utf-8'
            logger.debug(f"访问主页作为Referer状态码: {home_response.status_code}")
            
            # 检查主页是否成功加载
            if home_response.status_code != 200:
                logger.error(f"无法访问主页，状态码: {home_response.status_code}")
                return None
            
            # 准备搜索数据 - 从HTML代码中提取的正确参数
            data = {
                "keyboard": character,
                "show": "simtra",
                "classid": "33",  # 默认搜索楷书
                "tp": "tp1"
            }
            
            # 发送搜索请求
            response = self.session.post(
                self.search_url, 
                data=data, 
                headers=self._get_headers(referer=self.home_url),
                timeout=15,
                allow_redirects=True  # 必须允许重定向
            )
            # 明确设置响应编码
            response.encoding = 'utf-8'
            
            logger.info(f"搜索 {character} 返回状态码: {response.status_code}")
            logger.debug(f"搜索 {character} 的最终URL: {response.url}")
            
            # 检查响应内容片段
            debug_content = response.text[:500].replace('\n', ' ')
            logger.debug(f"响应内容片段: {debug_content}")
            
            # 检查是否是错误页面
            if "没有搜索到相关的内容" in response.text or "信息提示" in response.text:
                logger.warning(f"搜索 {character} 时网站返回'没有搜索到相关的内容'")
                return None
            
            # 检查是否重定向到结果页面
            if "/sf/ks/" in response.url and ".html" in response.url:
                # 从URL提取通用部分
                match = re.search(r'/sf/ks/(\d{4})/([a-f0-9]+)\.html', response.url)
                if match:
                    date_part = match.group(1)
                    hash_part = match.group(2)
                    
                    # 构建所有字体的URL
                    return {
                        "ks": f"{self.base_url}/sf/ks/{date_part}/{hash_part}.html",
                        "zs": f"{self.base_url}/sf/zs/{date_part}/{hash_part}.html",
                        "ls": f"{self.base_url}/sf/ls/{date_part}/{hash_part}.html",
                        "xs": f"{self.base_url}/sf/xs/{date_part}/{hash_part}.html",
                        "cs": f"{self.base_url}/sf/cs/{date_part}/{hash_part}.html"
                    }
            
            # 如果没有重定向到特定字体页面，尝试从搜索结果中提取
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尝试查找导航栏中的字体链接
            font_links = {}
            for font_code in self.font_types.keys():
                # 查找该字体类型的链接
                link = soup.find('a', href=re.compile(f'/sf/{font_code}/\\d{{4}}/[a-f0-9]+\\.html'))
                if link and link.get('href'):
                    full_url = urljoin(self.base_url, link['href'])
                    font_links[font_code] = full_url
                    logger.debug(f"找到 {self.font_types[font_code]} 链接: {full_url}")
            
            # 如果找到了5个字体链接，就返回
            if len(font_links) == 5:
                return font_links
            
            logger.warning(f"无法获取 {character} 的所有字体链接")
            return None
            
        except Exception as e:
            logger.exception(f"搜索 {character} 时出错")
            return None
    
    def _fetch_async_fonts(self, font_code, purl):
        """获取异步加载的字体图片"""
        async_url = f"{self.base_url}/sftplb/{font_code}/{purl}.php"
        logger.info(f"获取异步字体图片: {async_url}")
        
        try:
            self._ensure_request_interval()
            response = self.session.get(
                async_url, 
                headers=self._get_headers(referer=self.home_url),
                timeout=15
            )
            # 明确设置响应编码
            response.encoding = 'utf-8'
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"获取异步字体图片失败: {str(e)}")
            return None
    
    def _fetch_page(self, url, max_retries=3):
        """获取页面内容，带重试机制"""
        for attempt in range(max_retries):
            try:
                self._ensure_request_interval()
                response = self.session.get(
                    url, 
                    headers=self._get_headers(referer=self.home_url),
                    timeout=15
                )
                # 明确设置响应编码
                response.encoding = 'utf-8'
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"获取页面 {url} 时出错 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # 指数退避
                else:
                    return None
        return None
    
    def _parse_images(self, html_content, is_async=False):
        """解析HTML，提取图片链接"""
        if not html_content:
            return []
        
        try:
            image_urls = []
            
            if is_async:
                # 处理异步加载的图片
                # 1. 尝试直接查找img标签
                img_tags = re.findall(r'<img[^>]+src="([^"]+)"', html_content)
                for src in img_tags:
                    # 清理URL（去除多余空格等）
                    src = src.strip()
                    # 确保是完整URL
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.base_url, src)
                    image_urls.append(src)
                
                # 2. 如果没找到，尝试处理纯文本格式
                if not image_urls:
                    # 从知识库中看到，有时返回的是纯文本，但图片URL可能在其他地方
                    # 尝试查找可能的图片URL模式
                    urls = re.findall(r'https://sf\.zdic\.net/d/file/[^"\']+\.gif|https://sf\.zdic\.net/d/file/[^"\']+\.jpg', html_content)
                    for url in urls:
                        image_urls.append(url.strip())
                
                # 3. 如果还是没找到，检查是否有其他格式的图片URL
                if not image_urls:
                    # 尝试查找可能的图片URL
                    urls = re.findall(r'https?://[^"\']+?\.(?:gif|jpg|jpeg|png|svg)', html_content)
                    for url in urls:
                        image_urls.append(url.strip())
            else:
                # 处理常规页面
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # 尝试多种选择器来查找图片
                font_container = soup.select_one('#fontlist, .fontlist, #font-imgs, .font-imgs, .font-content, #content, .i_sfdiv, .sft')
                if font_container:
                    img_tags = font_container.find_all('img')
                    for img in img_tags:
                        src = img.get('src', '') or img.get('data-src', '')
                        if src:
                            # 确保是完整URL
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = urljoin(self.base_url, src)
                            # 清理URL中的多余空格
                            src = src.strip()
                            image_urls.append(src)
                
                # 方法2：如果方法1没找到，尝试查找所有图片
                if not image_urls:
                    img_tags = soup.find_all('img')
                    for img in img_tags:
                        src = img.get('src', '') or img.get('data-src', '')
                        if src and any(ext in src.lower() for ext in ['.gif', '.jpg', '.jpeg', '.png', '.svg']):
                            if src.startswith('//'):
                                src = 'https:' + src
                            elif src.startswith('/'):
                                src = urljoin(self.base_url, src)
                            # 清理URL
                            src = src.strip()
                            image_urls.append(src)
                
                # 方法3：如果还是没找到，尝试从异步加载参数中提取
                if not image_urls:
                    gsft_match = re.search(r'gsft\("([a-z]{2})","([\d/]+)"\)', html_content)
                    if gsft_match:
                        font_code, purl = gsft_match.groups()
                        async_content = self._fetch_async_fonts(font_code, purl)
                        if async_content:
                            async_image_urls = self._parse_images(async_content, is_async=True)
                            image_urls.extend(async_image_urls)
            
            # 去重
            image_urls = list(set(image_urls))
            
            return image_urls
        except Exception as e:
            logger.error(f"解析图片时出错: {str(e)}")
            return []
    
    def _download_image(self, url, save_path, character, font_code, font_name):
        """下载图片，带重试机制，先检查数据库"""
        # 先检查数据库，看是否已经下载过
        if self.is_image_downloaded(url):
            logger.debug(f"图片已存在，跳过下载: {url}")
            return True
        
        for attempt in range(3):
            try:
                self._ensure_request_interval()
                response = self.session.get(
                    url, 
                    headers=self._get_headers(referer=self.home_url),
                    timeout=20
                )
                response.raise_for_status()
                
                # 确保目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 保存图片
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                # 保存记录到数据库
                if self.save_image_record(character, font_code, font_name, url, save_path):
                    logger.debug(f"保存图片记录到数据库: {url} -> {save_path}")
                
                return True
            except Exception as e:
                logger.warning(f"下载图片 {url} 失败 (尝试 {attempt+1}/3): {str(e)}")
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))  # 指数退避
                else:
                    return False
        return False

    def get_font_image_count(self, character, font_code):
        """从数据库查询某汉字某字体的已下载图片数量"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM font_images WHERE character = ? AND font_type = ?",
                (character, font_code)
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"查询字体图片数量失败: {e}")
            return 0

    def _generate_index_html(self, char, char_dir):
        """生成汉字索引页面"""
        index_path = os.path.join(char_dir, "index.html")
        
        # 收集各字体目录信息
        font_dirs = []
        for font_code, font_name in self.font_types.items():
            font_dir = os.path.join(char_dir, font_code)
            if os.path.exists(font_dir):
                # 统计图片数量
                #img_count = len([f for f in os.listdir(font_dir) 
                #                if f.endswith('.gif') or f.endswith('.jpg') or f.endswith('.png') or f.endswith('.svg')])
				#
				# ✅ 用数据库统计图片数量，而不是 os.listdir
                img_count = self.get_font_image_count(char, font_code)
                font_dirs.append({
                    "code": font_code,
                    "name": font_name,
                    "count": img_count,
                    "path": f"{font_code}/index.html"
                })
        
        # 生成HTML内容
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{char} 书法字体展示</title>
    <style>
        body {{ 
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; 
            line-height: 1.6; 
            margin: 20px; 
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
        }}
        .font-list {{ 
            list-style: none; 
            padding: 0; 
        }}
        .font-item {{ 
            margin: 15px 0; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
            background-color: #fff;
            transition: transform 0.2s;
        }}
        .font-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }}
        .font-title {{ 
            font-size: 1.2em; 
            color: #2980b9; 
            text-decoration: none;
            display: block;
            font-weight: bold;
        }}
        .font-count {{ 
            color: #7f8c8d; 
            font-size: 0.9em; 
            margin-top: 5px;
        }}
        .stats {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{char} 书法字体展示</h1>
        <ul class="font-list">'''
        
        for font in font_dirs:
            html += f'''
        <li class="font-item">
            <a href="{font['path']}" class="font-title">{font['name']}</a>
            <span class="font-count">（共 {font['count']} 种写法）</span>
        </li>'''
        
        html += f'''
        </ul>
        <div class="stats">
            <p>本页面由书法字体抓取程序生成 | 汉字：{char}</p>
        </div>
    </div>
</body>
</html>'''
        
        # 保存HTML文件（使用UTF-8编码）
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return index_path
    
    def _generate_font_index_html(self, char, font_code, font_name, image_files, font_dir):
        """生成字体索引页面"""
        index_path = os.path.join(font_dir, "index.html")
        
        # 生成HTML内容
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{char} - {font_name}</title>
    <style>
        body {{ 
            font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; 
            line-height: 1.6; 
            margin: 20px; 
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 15px;
        }}
        .breadcrumb {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .breadcrumb a {{
            color: #2980b9;
            text-decoration: none;
        }}
        .breadcrumb span {{
            color: #7f8c8d;
        }}
        .image-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
            gap: 15px; 
        }}
        .image-item {{ 
            text-align: center; 
            background: white;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-item img {{ 
            max-width: 100%; 
            height: auto; 
            border: 1px solid #ddd; 
            border-radius: 4px;
            transition: transform 0.2s;
        }}
        .image-item img:hover {{
            transform: scale(1.05);
        }}
        .image-caption {{ 
            margin-top: 5px; 
            color: #666; 
            font-size: 0.9em; 
        }}
        .stats {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            <a href="../index.html">< 返回 {char} 首页</a> | <span>{font_name}</span>
        </div>
        <h1>{char} - {font_name}</h1>
        <div class="image-grid">'''
        
        fcount = 0
        for i, img_file in enumerate(image_files, 1):
            if os.path.exists(img_file):
                img_name = os.path.basename(img_file)
                fcount += 1
                html += f'''
            <div class="image-item">
                <img src="{img_name}" alt="{char} {font_name} 写法 {i}">
                <div class="image-caption">写法 {i}</div>
            </div>'''
        
        html += f'''
        </div>
        <div class="stats">
            <p>共 {fcount} 种 {font_name} 写法 | 汉字：{char}</p>
        </div>
    </div>
</body>
</html>'''
        
        # 保存HTML文件（使用UTF-8编码）
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return index_path
    
    def process_character(self, character):
        """处理单个汉字"""
        logger.info(f"===== 开始处理汉字: {character} =====")
        
        # 检查是否已处理过
        if self.is_character_completed(character):
            logger.info(f"汉字【 {character} 】已处理过，跳过")
            return False
        
        # 搜索汉字，获取各字体页面URL
        font_urls = self._search_character(character)
        if not font_urls:
            logger.warning(f"xx 无法获取【 {character} 】的字体链接，跳过")
            self.mark_character_not_found(character)
            return False
        
        # 创建汉字目录
        char_dir = os.path.join(self.output_dir, character)
        os.makedirs(char_dir, exist_ok=True)
        
        # 保存各字体页面和图片
        all_images_downloaded = True
        success_fonts = 0
        total_fonts = len(self.font_types)
        
        for font_code, font_name in self.font_types.items():
            logger.info(f"处理【 {character} 】 {font_name}...")
            
            # 检查是否有该字体的URL
            if font_code not in font_urls:
                logger.warning(f"xx 没有【 {character} 】 {font_name} 的URL，跳过")
                all_images_downloaded = False
                continue
            
            # 创建字体目录
            font_dir = os.path.join(char_dir, font_code)
            os.makedirs(font_dir, exist_ok=True)
            
            # 获取字体页面内容
            html_content = self._fetch_page(font_urls[font_code])
            if not html_content:
                logger.warning(f"xx 无法获取【 {character} 】 {font_name} 页面，跳过")
                all_images_downloaded = False
                continue
            '''
            # 保存HTML页面
            html_path = os.path.join(font_dir, "index.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content) '''
            
            # 尝试从页面中提取异步加载参数
            async_image_urls = []
            gsft_match = re.search(r'gsft\("([a-z]{2})","([\d/]+)"\)', html_content)
            if gsft_match and gsft_match.group(1) == font_code:
                purl = gsft_match.group(2)
                async_content = self._fetch_async_fonts(font_code, purl)
                if async_content:
                    async_image_urls = self._parse_images(async_content, is_async=True)
            
            # 解析常规图片
            regular_image_urls = self._parse_images(html_content)
            
            # 合并图片URL列表
            image_urls = list(set(async_image_urls + regular_image_urls))
            logger.info(f"找到【 {character} 】{len(image_urls)}张 {font_name} 图片")
            
            if not image_urls:
                logger.warning(f"未找到【 {character} 】{font_name}片，跳过")
                all_images_downloaded = False
                continue
            
            # 下载图片
            image_files = []
            for i, img_url in enumerate(image_urls, 1):
                # 生成文件名：字体代码+3位序号+.扩展名
                ext = os.path.splitext(img_url)[1]
                if not ext or len(ext) > 5:  # 防止无效扩展名
                    ext = '.gif'  # 默认使用gif
                filename = f"{font_code}{i:03d}{ext}"
                save_path = os.path.join(font_dir, filename)
                
                if self._download_image(img_url, save_path, character, font_code, font_name):
                    image_files.append(save_path)
                    logger.debug(f"      下载成功: {filename}")
                else:
                    logger.warning(f"      下载失败: {filename}")
                    all_images_downloaded = False
            
            # 重新生成字体索引页面，确保图片顺序正确
            if image_files:
                self._generate_font_index_html(character, font_code, font_name, image_files, font_dir)
                success_fonts += 1
                logger.info(f" 【 {character} 】{font_name} 处理完成，成功下载 {len(image_files)} 张图片")
            else:
                logger.warning(f"【 {character} 】 {font_name} 处理完成，但无有效图片")
        
        # 生成汉字索引页面
        self._generate_index_html(character, char_dir)
        
        # 标记为已完成
        if success_fonts > 0:
            self.mark_character_completed(character, success_fonts, total_fonts)
            status = "成功" if all_images_downloaded else "部分成功"
            logger.info(f"===== 汉字【 {character} 】处理完成 [{status}] ({success_fonts}/{total_fonts} 种字体) =====")
            return True
        else:
            logger.warning(f"===== 汉字【 {character} 】处理失败 (0/{total_fonts} 种字体) =====")
            return False
    
    def worker(self):
        """工作线程函数"""
        while True:
            try:
                character = self.task_queue.get(timeout=10)
                self.process_character(character)
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"工作线程出错: {str(e)}")
                self.task_queue.task_done()
    
    def start(self, characters=None):
        """启动爬虫"""
        if characters is None:
            # 从文件获取汉字列表
            characters = self.get_characters()
        
        if not characters:
            logger.error("没有汉字需要处理")
            return
        
        # 将任务添加到队列
        remaining_chars = []
        for char in characters:
            if self.is_character_completed(char) or self.is_character_not_found(char):
                continue
            else:
                self.task_queue.put(char)
                remaining_chars.append(char)
        
        logger.info(f"共 {len(remaining_chars)} 个汉字需要处理（总共 {len(characters)} 个）")
        
        if not remaining_chars:
            logger.info("所有汉字都已处理完毕")
            return
        
        # 启动工作线程
        threads = []
        for _ in range(self.thread_count):
            thread = Thread(target=self.worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        logger.info(f"启动 {self.thread_count} 个工作线程")
        
        # 等待所有任务完成
        self.task_queue.join()
        
        logger.info("===== 所有任务处理完成 =====")
        logger.info(f"总计处理 {len(remaining_chars)} 个汉字")

if __name__ == "__main__":
    # 创建爬虫实例
    scraper = FontScraper(
        output_dir="chinese_fonts",
        db_path="font_database.db",  
        thread_count=3,  # 由于反爬虫机制，线程数不宜过多
        min_interval=5.5  # 确保大于5秒
    )
    scraper.start()

'''    
    test_char = "无"
    logger.info(f"先测试单个已知存在的汉字'{test_char}'")
    if scraper.process_character(test_char):
        logger.info(f"测试成功！'{test_char}'字处理完成")
        logger.info("现在开始处理ciyun.txt中的所有汉字")
        scraper.start()
    else:
        logger.error(f"测试失败！请检查网络连接和网站状态")
        logger.error("建议手动访问 https://sf.zdic.net 并搜索'{test_char}'字，确认网站正常工作")
        
        # 尝试诊断问题
        logger.info("尝试诊断搜索问题...")
        try:
            # 检查能否访问主页
            home_response = scraper.session.get(scraper.home_url, headers=scraper._get_headers(), timeout=10)
            home_response.encoding = 'utf-8'
            logger.info(f"主页访问状态: {home_response.status_code}")
            
            # 尝试发送搜索请求
            data = {
                "keyboard": test_char,
                "show": "simtra",
                "classid": "33",
                "tp": "tp1"
            }
            search_response = scraper.session.post(
                scraper.search_url, 
                data=data, 
                headers=scraper._get_headers(referer=scraper.home_url),
                timeout=15,
                allow_redirects=True
            )
            search_response.encoding = 'utf-8'
            logger.info(f"搜索请求状态: {search_response.status_code}")
            logger.info(f"搜索结果URL: {search_response.url}")
            
            # 检查是否有错误信息
            if "没有搜索到相关的内容" in search_response.text:
                logger.error("网站返回'没有搜索到相关的内容'，可能的原因:")
                logger.error("1. 网站结构可能已改变")
                logger.error("2. 需要额外的请求头或cookies")
                logger.error("3. 网站可能有IP限制或验证码")
            
            # 保存响应内容供分析
            with open("search_response.html", "w", encoding="utf-8") as f:
                f.write(search_response.text)
            logger.info("已保存搜索响应内容到 search_response.html 供进一步分析")
        except Exception as e:
            logger.error(f"诊断过程中出错: {str(e)}")
'''