import cv2
import numpy as np
import os
import time
import argparse
from utils.face_detector import FaceDetector
from utils.anti_spoof import AntiSpoofDetector
from utils.face_recognizer import FaceRecognizer
from utils.user_manager import UserManager


RECOGNITION_MODEL_PATH = "models/face_recognition_model.pkl"

class FaceRecognitionApp:
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        
        # 预处理
        print("Initializing face detector...")
        self.face_detector = FaceDetector()
        
        print("Initializing anti-spoofing detector...")
        self.anti_spoof = AntiSpoofDetector()
        
        print("Initializing face recognizer...")
        self.face_recognizer = FaceRecognizer(RECOGNITION_MODEL_PATH)
        
        print("Initializing user manager...")
        self.user_manager = UserManager()
        
        # 预处理相机
        self.camera = None
        
        print("System initialization complete.")
    
    def _open_camera(self):
        """Open the camera"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
    
    def _close_camera(self):
        """Close the camera"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
    
    def register_user(self, user_id, name, num_samples=5):
        """Register a new user"""
        print(f"Registering user: {name} (ID: {user_id})")
        
        # 创建用户
        if not self.user_manager.create_user(user_id, name):
            print(f"Error: User {user_id} already exists")
            return False
        
        self._open_camera()
        
        # 测试
        samples_captured = 0
        last_capture_time = 0
        
        while samples_captured < num_samples:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # 镜像
            frame = cv2.flip(frame, 1)
            
            # 人脸试别
            faces = self.face_detector.detect_face(frame)
            
            # 框
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # DEBUG
            cv2.putText(frame, f"Capturing sample {samples_captured+1}/{num_samples}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Move your face slightly between captures", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 框
            cv2.imshow("Register User", frame)
            
            # 捕捉
            current_time = time.time()
            if len(faces) == 1 and (current_time - last_capture_time) > 2:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # 保存！！！！
                self.user_manager.add_face_sample(user_id, face_img)
                
                # 记录
                self.face_recognizer.add_face(user_id, face_img)
                
                samples_captured += 1
                last_capture_time = current_time
                print(f"Captured sample {samples_captured}/{num_samples}")
            
            # 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 清空
        cv2.destroyAllWindows()
        
        # 保存试别
        self.face_recognizer.save_model(RECOGNITION_MODEL_PATH)
        
        print(f"Registration completed for {name}")
        return True
    def reset_system(self):
        """Reset the entire recognition system"""
        print("Resetting the face recognition system...")
        
        # 1. Delete the model file if it exists
        if os.path.exists(RECOGNITION_MODEL_PATH):
            try:
                os.remove(RECOGNITION_MODEL_PATH)
                print(f"Deleted existing model file: {RECOGNITION_MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Could not delete model file: {e}")
        
        # 2. Clear all user data
        user_data_dir = "data/users"
        if os.path.exists(user_data_dir):
            try:
                # Remove all files and subdirectories
                for root, dirs, files in os.walk(user_data_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                print("Cleared user data directory")
            except Exception as e:
                print(f"Warning: Could not fully clear user data: {e}")
        
        # 3. Recreate empty user directory
        os.makedirs(user_data_dir, exist_ok=True)
        
        # 4. Create a fresh face recognizer instance
        self.face_recognizer = FaceRecognizer()
        print("Created new face recognizer instance")
        
        # 5. Initialize with empty model
        empty_model_dir = os.path.dirname(RECOGNITION_MODEL_PATH)
        os.makedirs(empty_model_dir, exist_ok=True)
        
        # 6. Save empty model
        try:
            self.face_recognizer.save_model(RECOGNITION_MODEL_PATH)
            print("Saved empty recognition model")
        except Exception as e:
            print(f"Warning: Could not save empty model: {e}")
        
        print("System reset complete! All recognition data has been cleared.")
        return True
    def recognize(self):
        """Start face recognition with anti-spoofing"""
        print("Starting face recognition")
        
        # Check if we have any users before starting
        users = self.user_manager.get_all_users()
        if not users:
            print("No users registered in the system. Please register a user first.")
            print("Use: python face_recognition_app.py --register --user-id user1 --name \"Your Name\"")
            return False
    
        """Start face recognition with anti-spoofing"""
        print("Starting face recognition")
        
        # Open camera
        self._open_camera()
        
        # Reset anti-spoofing detector
        self.anti_spoof.reset()
        
        # For result smoothing - now track each face separately
        face_results = {}  # Dictionary to track results for each face
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            
            # Mirror image
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect_face(frame)
            
            # Initialize set of current faces
            current_faces = set()
            
            # Process each face
            for i, (x, y, w, h) in enumerate(faces):
                # Generate a face ID based on position (approximate tracking)
                face_id = f"face_{i}"
                current_faces.add(face_id)
                
                # Initialize tracking if it's a new face
                if face_id not in face_results:
                    face_results[face_id] = {
                        "user_id": "Unknown",
                        "is_real": False,
                        "confidence": 0.0,
                        "stability": 0,
                        "display_user_id": "Unknown",
                        "display_is_real": False,
                        "display_confidence": 0.0
                    }
                
                # Get current results for this face
                result = face_results[face_id]
                
                # Check if it's a real face
                is_real, spoof_score = self.anti_spoof.check_liveness(frame, (x, y, w, h))
                
                # If real, recognize
                if is_real:
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    user_id, confidence = self.face_recognizer.recognize(face_img)
                else:
                    user_id = "Unknown"
                    confidence = 0.0
                
                # Update stability for this specific face
                if user_id == result["user_id"] and is_real == result["is_real"]:
                    result["stability"] += 1
                else:
                    result["stability"] = 0
                
                # Update display when stable
                if result["stability"] >= 3:  # Required stability
                    result["display_user_id"] = user_id
                    result["display_is_real"] = is_real
                    result["display_confidence"] = confidence
                
                # Save current for next comparison
                result["user_id"] = user_id
                result["is_real"] = is_real
                result["confidence"] = confidence
                
                # Determine rectangle and text colors based on THIS face's status
                if not result["display_is_real"]:
                    # Spoof detection - Red
                    rect_color = (0, 0, 255)  # Red
                    text_color = (0, 0, 255)  # Red
                    result_text = "Spoof Detected"
                elif result["display_user_id"] != "Unknown":
                    # Registered user - Green
                    rect_color = (0, 255, 0)  # Green
                    text_color = (0, 255, 0)  # Green
                    result_text = f"{result['display_user_id']} ({result['display_confidence']:.2f})"
                else:
                    # Unknown but real person - Yellow
                    rect_color = (0, 255, 255)  # Yellow
                    text_color = (0, 255, 255)  # Yellow
                    result_text = "Unknown Person"
                
                # Draw face box with appropriate color
                cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)
                
                # Show result with matching color
                cv2.putText(frame, result_text, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # Clean up faces that are no longer visible
            face_ids_to_remove = []
            for face_id in face_results:
                if face_id not in current_faces:
                    face_ids_to_remove.append(face_id)
                    
            for face_id in face_ids_to_remove:
                del face_results[face_id]
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show anti-spoofing instruction only if there's at least one spoof
            if any(not result["display_is_real"] for result in face_results.values()):
                cv2.putText(frame, "Move slightly to verify liveness", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Face Recognition", frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Clean up
        cv2.destroyAllWindows()
        self._close_camera()
    
    def delete_user(self, user_id):
        """Delete a user from the system"""
        print(f"Deleting user: {user_id}")
        
        # Check if user exists
        if not self.user_manager.user_exists(user_id):
            print(f"Error: User {user_id} does not exist")
            return False
        
        # Delete user from user manager
        if not self.user_manager.delete_user(user_id):
            print("Failed to delete user data")
            return False
        
        # Completely reset the recognition model and rebuild from scratch
        print("Rebuilding recognition model without the deleted user...")
        
        # Create a new face recognizer instance
        self.face_recognizer = FaceRecognizer()
        
        # Get all remaining users
        remaining_users = self.user_manager.get_all_users()
        print(f"Found {len(remaining_users)} remaining users")
        
        # Add all face samples from remaining users
        for user_info in remaining_users:
            current_id = user_info["id"]
            print(f"Adding samples for user: {current_id}")
            
            # Get face samples for this user
            face_samples = self.user_manager.get_user_face_samples(current_id)
            print(f"Found {len(face_samples)} samples")
            
            # Add each face to the recognizer
            for face in face_samples:
                self.face_recognizer.add_face(current_id, face)
        
        # Save the updated model
        if os.path.exists(RECOGNITION_MODEL_PATH):
            # Delete existing model first
            try:
                os.remove(RECOGNITION_MODEL_PATH)
                print("Deleted old model file")
            except Exception as e:
                print(f"Warning: Could not delete old model file: {e}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(RECOGNITION_MODEL_PATH), exist_ok=True)
        
        # Save new model
        success = self.face_recognizer.save_model(RECOGNITION_MODEL_PATH)
        
        if success:
            print(f"Successfully rebuilt recognition model without user {user_id}")
        else:
            print("Failed to save updated recognition model")
            return False
        
        print(f"User {user_id} deleted successfully")
        return True
        
    def cleanup(self):
        """Clean up resources"""
        self._close_camera()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition with Anti-Spoofing")
    parser.add_argument("--register", action="store_true", help="Register a new user")
    parser.add_argument("--recognize", action="store_true", help="Start face recognition")
    parser.add_argument("--delete-user", type=str, help="Delete a user by ID")
    parser.add_argument("--list-users", action="store_true", help="List all registered users")
    parser.add_argument("--reset", action="store_true", help="Reset the entire system")
    parser.add_argument("--user-id", type=str, help="User ID for registration")
    parser.add_argument("--name", type=str, help="User name for registration")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to capture")
    
    args = parser.parse_args()
    
    app = FaceRecognitionApp()
    
    try:
        if args.register:
            if not args.user_id or not args.name:
                print("Error: --user-id and --name are required for registration")
                return
            app.register_user(args.user_id, args.name, args.samples)
        elif args.recognize:
            app.recognize()
        elif args.delete_user:
            app.delete_user(args.delete_user)
        elif args.list_users:
            users = app.user_manager.get_all_users()
            if not users:
                print("No users registered")
            else:
                print("Registered users:")
                for user in users:
                    print(f"ID: {user['id']}, Name: {user['name']}, Samples: {user['samples']}")
        elif args.reset:
            app.reset_system()
        else:
            print("Error: Please specify an action (--register, --recognize, --delete-user, --list-users, or --reset)")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        app.cleanup()
if __name__ == "__main__":
    main()