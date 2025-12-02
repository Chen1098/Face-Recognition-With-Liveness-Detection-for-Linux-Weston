import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognizer:
    def __init__(self, model_path=None):
        self.model = None
        self.face_encodings = []
        self.user_ids = []
        
        # 载入模型训练
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.face_encodings = data['encodings']
                    self.user_ids = data['user_ids']
            except Exception as e:
                print(f"Error loading recognition model: {e}")
    
    def add_face(self, user_id, face_image):
        """Add a face to the recognizer"""
        # 提取
        encoding = self._extract_features(face_image)
        
        # 加入数据库
        self.face_encodings.append(encoding)
        self.user_ids.append(user_id)
        
        # AI训练
        if len(self.face_encodings) > 0:
            self._train_model()
            
        return True
    
    def recognize(self, face_image, threshold=0.6):
        """Recognize a face"""
        if self.model is None or len(self.face_encodings) == 0:
            return "Unknown", 0.0
            
        # 提取
        encoding = self._extract_features(face_image)
        
        # 找邻居
        distances, indices = self.model.kneighbors([encoding], 
                                                n_neighbors=min(3, len(self.face_encodings)))
        
        # 距离转化成概率
        distances = distances[0]
        max_dist = np.sqrt(400)  # 最大可能值
        similarities = [1 - (dist / max_dist) for dist in distances]
        avg_similarity = np.mean(similarities)
        
        # 找到频率最大
        predictions = [self.user_ids[i] for i in indices[0]]
        unique_preds, counts = np.unique(predictions, return_counts=True)
        most_common_idx = np.argmax(counts)
        predicted_id = unique_preds[most_common_idx]
        
        # 对比错误
        if avg_similarity < threshold:
            return "Unknown", avg_similarity
            
        return predicted_id, avg_similarity
    
    def _extract_features(self, face_image):
        """Extract features from face image"""
        # 转化正常大小
        face_resized = cv2.resize(face_image, (64, 64))
        
        # 黑白
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
            
        face_norm = face_gray.astype('float32') / 255.0
        
        # 扁平
        features = face_norm.flatten()
        
        return features
    
    def _train_model(self):
        """Train a lightweight model optimized for STM32MP257"""
        # Use simpler KNN with fewer neighbors for faster performance
        self.model = KNeighborsClassifier(
            n_neighbors=1,  # Reduced from 5 for faster computation
            weights='distance',  # Weight by distance for better accuracy with fewer neighbors
            algorithm='kd_tree',  # Faster algorithm
            leaf_size=30,
            n_jobs=1  # Force single-threaded
        )
        self.model.fit(self.face_encodings, self.user_ids)
    
    def save_model(self, model_path):
        """Save the model to disk"""
        if self.model is not None:
            # 如果不存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存数据
            data = {
                'model': self.model,
                'encodings': self.face_encodings,
                'user_ids': self.user_ids
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
                
            return True
        return False