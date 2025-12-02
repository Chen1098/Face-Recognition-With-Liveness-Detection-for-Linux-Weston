# 人脸识别系统 (Face Recognition System with Anti-Spoofing)



## 系统概述 (System Overview)

这是一个具有防伪功能的人脸识别系统，可以识别登记用户的身份和职位，并能检测照片或视频欺骗。

本系统具有以下功能：

- 用户注册和管理
- 实时人脸识别
- 职位显示（机电科长、机电副科长、机电队长、机电副队长）
- 防照片/视频欺骗检测
- 距离检测（人脸需足够接近摄像头）
- 完全中文界面

## Windows 安装指南 (Windows Installation Guide)

### 系统要求 (System Requirements)

- Windows 7/8/10/11
- Python 3.9+
- 网络摄像头
- 最小 2GB RAM
- 50MB 磁盘空间

### 步骤 1: 安装 Python

1. 从 [Python 官网](https://www.python.org/downloads/) 下载并安装 Python 3.9 或更高版本
2. 安装时勾选 "Add Python to PATH" 选项

### 步骤 2: 下载项目

1. 创建一个文件夹用于存放项目，例如 `C:\FaceRecognition`
2. 将项目文件复制到该文件夹中，确保保持原有的目录结构

### 步骤 3: 创建虚拟环境并安装依赖

打开命令提示符 (CMD)，然后运行以下命令：

cd C:\FaceRecognitionpython -m venv venvvenv\Scripts\activatepip install opencv-python numpy pillow scikit-learn

### 步骤 4: 安装中文字体（如果需要）

    如果系统中没有安装 SimHei 或 SimSun 字体，可以从以下位置下载：
    - SimHei: [下载链接](https://www.fontpalace.com/font-details/SimHei/)
    - SimSun: [下载链接](https://www.fontpalace.com/font-details/SimSun/)
    
    
    下载后，双击字体文件安装。

### 步骤 5: 运行程序

python face_recognition_gui.py

## 使用指南 (Usage Guide)

### 添加新用户

    1. 点击 "添加用户" 按钮
    2. 输入用户 ID 和姓名
    3. 选择职位
    4. 确认后跟随屏幕指示，采集人脸样本



### 识别用户

    1. 点击 "开始识别" 按钮
    2. 将人脸对准摄像头
    3. 系统将显示识别结果
    
    ### 管理用户
    1. 点击 "管理用户" 按钮
    2. 可查看、删除用户
    
    ### 重置系统
    1. 点击 "重置系统" 按钮
    2. 确认后将删除所有用户数据
    
    ## 调整系统参数 (System Adjustments)
    
    ### 调整防伪检测灵敏度
    
    如果需要增加或降低防伪检测灵敏度，编辑 `utils/anti_spoof.py` 文件：
    
    #### 提高灵敏度（更容易检测欺骗）
    ```python
    # 找到 check_liveness 方法中的这行
    is_live = combined_score > 0.4
    # 修改为
    is_live = combined_score > 0.5  # 增大阈值
    
    # 找到 AntiSpoofDetector 类初始化方法中的这些值
    self.texture_threshold = 60
    self.required_video_detections = 4
    # 修改为
    self.texture_threshold = 50  # 降低阈值
    self.required_video_detections = 3  # 降低检测次数要求

#### 降低灵敏度（减少误报）

python
    # 找到 check_liveness 方法中的这行
    is_live = combined_score > 0.4
    # 修改为
    is_live = combined_score > 0.3  # 减小阈值
    # 找到 AntiSpoofDetector 类初始化方法中的这些值
    self.texture_threshold = 60
    self.required_video_detections = 4
    # 修改为
    self.texture_threshold = 70  # 提高阈值
    self.required_video_detections = 5  # 增加检测次数要求

### 调整人脸大小要求

如果需要改变人脸大小要求，编辑 `face_recognition_gui.py` 文件顶部常量：

python
    # 找到以下常量
    MIN_FACE_PROPORTION = 0.05  # 识别时的最小人脸比例
    MIN_REGISTER_PROPORTION = 0.1  # 注册时的最小人脸比例
    # 调整为所需值
    MIN_FACE_PROPORTION = 0.03  # 降低比例，允许更远距离识别
    MIN_REGISTER_PROPORTION = 0.08  # 降低比例，允许更远距离注册

### 调整识别刷新率

如果需要提高刷新率或降低系统资源使用，编辑 `face_recognition_gui.py` 文件顶部常量：

python
    # 找到以下常量
    PROCESS_EVERY_N_FRAMES = 3  # 每 N 帧处理一次
    RECOGNITION_INTERVAL = 0.1  # 处理间隔秒数
    # 调整值
    PROCESS_EVERY_N_FRAMES = 2  # 减少帧数，提高处理频率
    RECOGNITION_INTERVAL = 0.05  # 减少间隔，提高处理频率
    # 或者降低处理频率以节省资源
    PROCESS_EVERY_N_FRAMES = 5  # 增加帧数，降低处理频率
    RECOGNITION_INTERVAL = 0.2  # 增加间隔，降低处理频率

### 调整相机分辨率

如果需要调整相机分辨率，编辑 `face_recognition_gui.py` 文件顶部常量：python
    # 找到以下常量
    CAPTURE_WIDTH = 320
    CAPTURE_HEIGHT = 240
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 360
    # 调整为所需分辨率
    CAPTURE_WIDTH = 640  # 提高捕获分辨率
    CAPTURE_HEIGHT = 480
    DISPLAY_WIDTH = 800  # 提高显示分辨率
    DISPLAY_HEIGHT = 600
    # 或者降低分辨率以提高性能
    CAPTURE_WIDTH = 240  # 降低捕获分辨率
    CAPTURE_HEIGHT = 180
    DISPLAY_WIDTH = 320  # 降低显示分辨率
    DISPLAY_HEIGHT = 240
常见问题及解决方案 (Troubleshooting)

### 系统无法正确显示中文

确保系统已安装 SimHei 或 SimSun 字体。如果仍有问题，尝试修改 `draw_chinese_text` 函数：

python
    font = ImageFont.truetype("simhei.ttf", font_size)

改为:

python
    # 将字体文件复制到项目目录并使用完整路径
    font = ImageFont.truetype("C:/FaceRecognition/fonts/simhei.ttf", font_size)

### 摄像头未打开

确保系统有可用的摄像头设备。如果有多个摄像头，可以在 `recognition_loop` 和 `register_user` 方法中修改：

python
    self.camera = cv2.VideoCapture(0)  # 0 是默认摄像头

改为:

python
    self.camera = cv2.VideoCapture(1)  # 使用设备索引 1 的摄像头

### 识别准确率低

1. 增加训练样本数量 - 在添加用户时提供更多样本（3-5个）
2. 确保光线充足
3. 注册用户时确保不同角度的人脸样本
4. 调整识别阈值 - 在 `face_recognizer.py` 中修改 `confidence_threshold` 参数

### 系统运行缓慢

1. 降低相机分辨率

2. 增加 `PROCESS_EVERY_N_FRAMES` 和 `RECOGNITION_INTERVAL` 值

3. 减少打开的程序数量

4. 对于低配置设备，可以移除 `anti_spoof.py` 中的某些检测方法
    And here's the Linux version:
   
   ```markdown
   
   # 人脸识别系统 (Face Recognition System with Anti-Spoofing)
   
   ## 系统概述 (System Overview)
   
   这是一个具有防伪功能的人脸识别系统，可以识别登记用户的身份和职位，并能检测照片或视频欺骗。
   本系统具有以下功能：
   
   - 用户注册和管理
   
   - 实时人脸识别
   
   - 职位显示（机电科长、机电副科长、机电队长、机电副队长）
   
   - 防照片/视频欺骗检测
   
   - 距离检测（人脸需足够接近摄像头）
   
   - 完全中文界面
   
    ## Linux 安装指南 (Linux Installation Guide)
   
    ### 系统要求 (System Requirements)
   
   - 任何主流 Linux 发行版 (Ubuntu, Debian, Fedora, CentOS 等)
   
   - Python 3.9+
   
   - 网络摄像头
   
   - 最小 2GB RAM
   
   - 50MB 磁盘空间
   
    ### 步骤 1: 安装系统依赖
   
    对于 Ubuntu/Debian 系统:
    ```bash
    sudo apt update
    sudo apt install -y python3-pip python3-venv python3-tk python3-dev python3-pil.imagetk libopencv-dev
   
   ```


