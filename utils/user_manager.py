import os
import cv2
import pickle
from datetime import datetime

class UserManager:
    def __init__(self, data_path="data/users"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
    def _get_user_path(self, user_id):
        return os.path.join(self.data_path, f"{user_id}")
    
    def _get_face_samples_path(self, user_id):
        user_path = self._get_user_path(user_id)
        samples_path = os.path.join(user_path, "faces")
        os.makedirs(samples_path, exist_ok=True)
        return samples_path
    
    def user_exists(self, user_id):
        return os.path.exists(self._get_user_path(user_id))
    
    def create_user(self, user_id, name, job=""):
        """Create a new user with name and job title"""
        if self.user_exists(user_id):
            return False
            
        user_path = self._get_user_path(user_id)
        os.makedirs(user_path, exist_ok=True)
        
        # Create user info file with job title
        user_info = {
            "id": user_id,
            "name": name,
            "job": job,  # Store job title
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "samples": 0
        }
        
        with open(os.path.join(user_path, "info.pkl"), "wb") as f:
            pickle.dump(user_info, f)
            
        # Create samples directory
        self._get_face_samples_path(user_id)
        
        return True
    
    def add_face_sample(self, user_id, face_image):
        """Add a face sample for a user"""
        if not self.user_exists(user_id):
            return False
            
        # Get user info
        user_dir = os.path.join(self.data_path, user_id)
        info_path = os.path.join(user_dir, "info.pkl")
        
        with open(info_path, "rb") as f:
            user_info = pickle.load(f)
            
        # Update sample count
        sample_idx = user_info["samples"]
        user_info["samples"] += 1
        
        # Save updated info
        with open(info_path, "wb") as f:
            pickle.dump(user_info, f)
            
        # Save face image
        samples_dir = os.path.join(user_dir, "faces")
        image_path = os.path.join(samples_dir, f"sample_{sample_idx}.jpg")
        
        cv2.imwrite(image_path, face_image)
        
        return True
    
    def get_all_users(self):
        """Get list of all registered users"""
        users = []
        
        for user_id in os.listdir(self.data_path):
            user_dir = os.path.join(self.data_path, user_id)
            info_path = os.path.join(user_dir, "info.pkl")
            
            if os.path.isdir(user_dir) and os.path.exists(info_path):
                with open(info_path, "rb") as f:
                    user_info = pickle.load(f)
                    users.append(user_info)
                    
        return users
    
    def get_user_face_samples(self, user_id):
        """Get all face samples for a user"""
        samples = []
        
        if not self.user_exists(user_id):
            return samples
            
        samples_dir = os.path.join(self.data_path, user_id, "faces")
        
        for filename in os.listdir(samples_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(samples_dir, filename)
                face_image = cv2.imread(image_path)
                if face_image is not None:
                    samples.append(face_image)
                    
        return samples
    
    def delete_user(self, user_id):
        """Delete a user and all their face samples"""
        if not self.user_exists(user_id):
            return False
        
        # Get user directory
        user_dir = os.path.join(self.data_path, user_id)
        
        # Remove all files in the user directory
        for root, dirs, files in os.walk(user_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        
        # Remove the user directory
        os.rmdir(user_dir)
        
        return True