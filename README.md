# 汉字书法识别器

![screenshot](/assets/screenshot.png)

**缘起**: 经常遇到龙飞凤舞的书法作品，很多字不认识。于是计划创建一个汉字书法识别器，用户输入一张汉字图片，软件可以识别出对应的汉字。

**设计思路** ：利用爬虫程序在网上抓取汉字书法图片文件，对这些图片清洗整理之后，训练一个基于卷积神经网络CNN的模型，而后即可识别。

**关于数据**： （1）汉字集合来自《晚翠轩词韵》与《通用汉字规范表》；（2）数据采集的工作量最大，书法字体包括有篆书（zs）、隶书（ls）、草书（cs）、行书（xs）、楷书（ks），目前已抓取了6764个汉字的书法图片，另有1174个生僻汉字的图片无法找到，已抓取的图片数量在20万张左右，其中经过清洗后实际可以使用的图片近18万张。

**关于模型**：采用了ResNet50模型，该模型具有比较好的精准度，但模型训练也非常耗时。样本数据测试表明，对于汉字书法的识别率超过85%。模型文件可在HF下载 

https://huggingface.co/xiajingfeng/chinese-calligraphy-recognition-v1

**程序文件**：
```
scraper.py： 爬虫程序，抓取书法字体图片，图片保存在chinese_fonts目录，汉字信息保存在SQLite数据库文件font_database.db。

clean.py：图片清理程序，找出矢量图、动态gif、小尺寸图片(<50px)，可以将这些图片删除。

train_model.py：训练ResNet50模型，50轮计算，生成best_model.pth模型文件。

gui.py：用户程序，输入图片，检索出置信度最高的5个汉字。
```
**前置要求**

开始使用前，请确保已安装以下依赖项：

```
Python 3.11+

PyTorch（建议使用CUDA支持以加速训练）

PyQt5（用于图形界面）

其他Python包：PIL, torchvision, requests, beautifulsoup4, sqlite3
```

**模型训练**

```
python train_model.py --data-dir chinese_fonts --epochs 50
```

**用户程序运行**

若已有预训练模型（best_model.pth）和字符映射（char_map.json），可以立即开始识别字符：

```
python gui.py
```


## Web访问

**程序文件**

```
main.py：基于FastAPI的后端程序

static/index.html：用户前端页面
```

**启动Web服务**

请安装uvicorn，而后执行如下命令：

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

从而在浏览器中访问127.0.0.1:8000

或者可以访问部署在HuggingFace的web服务 https://huggingface.co/spaces/xiajingfeng/shufa

![screenshot](/assets/screenshot2.png)
