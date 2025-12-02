# 防造假人脸试别系统

系统要求

* Windows 7/8/10/11
* Python 3.9+
* 最小 500MB RAM
* 50MB 磁盘空间

### 运行

1. 项目存储位置：C:\FaceRecognition

2. 创建虚拟环境：
   
   1. cd C:\FaceRecognition
   
   2. python -m venv venv
   
   3. venv\Scripts\activate
   
   4. pip install opencv-python numpy pillow scikit-learn

3. 运行程序 python face_recognition_gui.py

### 参数微调

1. 降低灵敏度
   
   1. 找到 check_liveness 中 
      is_live = combined_score > 0.4
      改为
      is_live = combined_score > 0.3 
      减小阈值
   2. 找到 AntiSpoofDetector 中
      
      self.texture_threshold = 60
      
      self.required_video_detections = 4
      
      改为
      
      self.texture_threshold = 70
      
      self.required_video_detections = 5

2. 调整刷新率
   
   1. face_recognition_gui.py 
   
   2. 
