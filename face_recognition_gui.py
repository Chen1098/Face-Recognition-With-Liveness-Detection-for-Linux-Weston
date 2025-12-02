import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
# Add these imports at the top (around line 5)
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import numpy as np
from PIL import Image, ImageTk
import gc  # For garbage collection
from utils.face_detector import FaceDetector
from utils.anti_spoof import AntiSpoofDetector
from utils.face_recognizer import FaceRecognizer
from utils.user_manager import UserManager
from utils.ipc_manager import IPCManager
# Limit threads for better resource sharing on STM32MP257
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Constants
RECOGNITION_MODEL_PATH = "models/face_recognition_model.pkl"

# Reduced resolution for STM32MP257
CAPTURE_WIDTH = 160
CAPTURE_HEIGHT = 120
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

# Reduced processing frequency for STM32MP257
PROCESS_EVERY_N_FRAMES = 10
RECOGNITION_INTERVAL = 0.5

# Face size requirements
MIN_FACE_PROPORTION = 0.05
MIN_REGISTER_PROPORTION = 0.1

# Available job titles
JOB_TITLES = ["机电科长", "机电副科长", "机电队长", "机电副队长"]
# Processing parameters
PROCESS_EVERY_N_FRAMES = 3  # Only process every Nth frame for recognition
RECOGNITION_INTERVAL = 0.1  # Seconds between recognitions

# Face size requirements
MIN_FACE_PROPORTION = 0.05  # Face should take up at least 5% of the frame for recognition
MIN_REGISTER_PROPORTION = 0.1  # Face should take up at least 10% of the frame for registration

# Available job titles
JOB_TITLES = ["机电科长", "机电副科长", "机电队长", "机电副队长"]
# Add this function after imports but before class definition (around line 30)
def draw_chinese_text(img, text, position, color, font_size=20):
    """Draw Chinese text on image using PIL"""
    # Convert OpenCV BGR image to RGB for PIL
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_img)
    
    try:
        # Try to use SimHei font if available
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        try:
            # Fallback to SimSun if available
            font = ImageFont.truetype("simsun.ttc", font_size)
        except:
            # Last resort - use default
            font = ImageFont.load_default()
    
    # Draw text
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV BGR
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv2_img

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别系统")
        
        # Initialize system components
        self.initialize_system()
        
        # Setup GUI components with optimized layout
        self.setup_gui()
        self.ipc_manager = IPCManager()
        # Initialize camera variables
        self.camera = None
        self.is_camera_active = False
        self.recognition_thread = None
        self.stop_recognition = False
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Start periodic maintenance tasks
        self.periodic_cleanup()
        self.check_system_temperature()  # Start temperature monitoring
            
    def initialize_system(self):
        """Initialize the face recognition system components"""
        print("初始化系统...")
        os.makedirs("models", exist_ok=True)
        
        self.face_detector = FaceDetector()
        self.anti_spoof = AntiSpoofDetector()
        self.face_recognizer = FaceRecognizer(RECOGNITION_MODEL_PATH)
        self.user_manager = UserManager()
        print("系统初始化完成")
        
    def setup_gui(self):
        """Setup GUI components with optimized layout"""
        # Main frame with reduced padding
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create video canvas with optimized size
        self.video_canvas = tk.Canvas(main_frame, bg="black", width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        self.video_canvas.pack(padx=2, pady=2)
        
        # Status label with smaller font
        self.status_label = ttk.Label(main_frame, text="状态: 系统就绪", font=("SimHei", 9))
        self.status_label.pack(pady=2)
        
        # Create button frame with reduced padding
        button_frame = ttk.Frame(main_frame, padding="2")
        button_frame.pack(fill=tk.X)
        
        # Create buttons with optimized size
        button_width = 10
        
        self.start_button = ttk.Button(button_frame, text="开始识别", 
                                       width=button_width, command=self.toggle_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.add_user_button = ttk.Button(button_frame, text="添加用户", 
                                         width=button_width, command=self.add_user)
        self.add_user_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.manage_users_button = ttk.Button(button_frame, text="管理用户", 
                                             width=button_width, command=self.show_user_management)
        self.manage_users_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reset_button = ttk.Button(button_frame, text="重置系统", 
                                      width=button_width, command=self.reset_system)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Adjust window size to content
        self.root.update()
        self.root.geometry(f"{DISPLAY_WIDTH+20}x{DISPLAY_HEIGHT+100}")
        
        # Initialize message on canvas
        self.video_canvas.create_text(DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2, 
                                     text="点击 '开始识别' 按钮启动摄像头", 
                                     fill="white", font=("SimHei", 12), tags="message")
    
    def periodic_cleanup(self):
        """Perform periodic garbage collection to free memory"""
        gc.collect()
        # Schedule the next cleanup
        self.root.after(30000, self.periodic_cleanup)  # Every 30 seconds
        
    def check_system_temperature(self):
        """Monitor system temperature and reduce processing load if needed"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            
            # Check if temperature is too high
            if temp > 75.0:  # Critical temperature threshold
                # Increase frame skipping to reduce processing load
                global PROCESS_EVERY_N_FRAMES
                current_skip = PROCESS_EVERY_N_FRAMES
                PROCESS_EVERY_N_FRAMES = min(20, current_skip + 2)
                print(f"Temperature: {temp}°C - Reducing processing (frame skip: {PROCESS_EVERY_N_FRAMES})")
            elif temp > 65.0:  # Warning temperature threshold
                print(f"Temperature: {temp}°C - High")
            
        except Exception as e:
            # Temperature monitoring might not be available
            pass
        
        # Schedule next temperature check (every 30 seconds)
        self.root.after(30000, self.check_system_temperature)
        
    def toggle_recognition(self):
        """Start or stop the recognition process"""
        if self.is_camera_active:
            # Stop recognition
            self.stop_recognition = True
            self.start_button.configure(text="开始识别")
            self.status_label.configure(text="状态: 系统停止")
        else:
            # Start recognition
            self.start_button.configure(text="停止识别")
            self.status_label.configure(text="状态: 初始化中...")
            self.start_recognition()
            
    def start_recognition(self):
        """Start the recognition process in a separate thread"""
        self.stop_recognition = False
        self.recognition_thread = threading.Thread(target=self.recognition_loop)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
        
    def recognition_loop(self):
        """Optimized recognition loop with improved display stability"""
        # Open camera with reduced resolution
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("错误", "无法打开摄像头！")
            self.start_button.configure(text="开始识别")
            self.status_label.configure(text="状态: 摄像头错误")
            return
            
        # Set camera properties for lower resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 15)  # Lower framerate
        
        self.is_camera_active = True
        self.status_label.configure(text="状态: 正在识别...")
        
        # Reset detection components
        self.anti_spoof.reset()
        
        # For result tracking
        face_results = {}
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # For display stability
        last_display_frame = None
        display_tk_img = None
        
        # Use a separate thread for frame grabbing to prevent blocking
        frames_queue = []
        max_queue_size = 2
        
        def grab_frames():
            """Thread function to continuously grab frames"""
            while not self.stop_recognition:
                ret, frame = self.camera.read()
                if ret:
                    # Mirror image
                    frame = cv2.flip(frame, 1)
                    
                    # Add to queue, maintain max size
                    frames_queue.append(frame)
                    while len(frames_queue) > max_queue_size:
                        frames_queue.pop(0)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        # Start frame grabbing thread
        grab_thread = threading.Thread(target=grab_frames)
        grab_thread.daemon = True
        grab_thread.start()
        
        # Get all users and their information
        users_info = {}
        for user in self.user_manager.get_all_users():
            user_id = user['id']
            # Extract name and job from the user info if available, otherwise use defaults
            name = user.get('name', user_id)
            job = user.get('job', "未指定职位")  # Default job if not specified
            users_info[user_id] = {"name": name, "job": job}
        
        # Main loop for processing and display
        while not self.stop_recognition:
            # Get a frame from the queue if available
            if not frames_queue:
                # No frames available, use last display frame or wait
                if last_display_frame is not None:
                    # Just update the display with the last frame to prevent flickering
                    try:
                        self.video_canvas.img = display_tk_img  # Use existing image
                    except:
                        pass
                time.sleep(0.01)
                continue
            
            # Get the latest frame
            frame = frames_queue[-1].copy()
            frames_queue.clear()  # Clear queue after getting the latest
            
            # Always save a copy for backup display
            if last_display_frame is None:
                last_display_frame = frame.copy()
            
            # Increment frame counter
            self.frame_count += 1
            current_time = time.time()
            
            # Decide whether to process this frame for recognition
            should_process = (self.frame_count % PROCESS_EVERY_N_FRAMES == 0 and 
                             (current_time - self.last_process_time) >= RECOGNITION_INTERVAL)
            
            # Process face detection and recognition
            # Update status for C++ applications via IPC
            if should_process:
                # Collect recognition results for IPC
                recognition_results = []
                for face_id, result in face_results.items():
                    if not result.get("is_too_far", False) and result["display_is_real"]:
                        recognition_results.append({
                            "user_id": result["display_user_id"],
                            "name": result["display_name"],
                            "job": result["display_job"],
                            "confidence": float(result["display_confidence"]),
                            "position": result["rect"]
                        })
                
                # Update IPC status
                self.ipc_manager.update_status({
                    "timestamp": time.time(),
                    "faces_detected": len(face_results),
                    "recognized_users": recognition_results
                })
                
                # Check for commands from C++ applications
                command = self.ipc_manager.check_commands()
                if command:
                    # Process command
                    command_type = command.get("command")
                    if command_type == "reset":
                        # Schedule a reset
                        self.root.after(100, self.reset_system)
                    elif command_type == "add_user":
                        # Schedule user addition (after stopping recognition)
                        user_data = command.get("data", {})
                        if user_data:
                            def add_user_from_command():
                                if self.is_camera_active:
                                    self.toggle_recognition()  # Stop recognition
                                self.root.after(500, lambda: self.add_user_with_data(user_data))
                            self.root.after(100, add_user_from_command)
            # Always create a fresh display frame
            display_frame = frame.copy()
            
            # Draw stored results
            # Draw stored results
            for face_id, result in face_results.items():
                x, y, w, h = result["rect"]
                
                # Check if this face is too far
                if "is_too_far" in result and result["display_is_too_far"]:
                    # Yellow for "too far" faces
                    rect_color = (0, 255, 255)  # Yellow (BGR)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), rect_color, 2)
                    # Draw text
                    display_frame = draw_chinese_text(display_frame, "距离太远", (x, y-25), (0, 255, 255), 20)
                
                # If it's a spoof detection
                elif not result["display_is_real"]:
                    # Red for spoof faces
                    rect_color = (0, 0, 255)  # Red
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), rect_color, 2)
                    # Draw text
                    display_frame = draw_chinese_text(display_frame, "假体检测", (x, y-25), (0, 0, 255), 20)
                
                # If it's a recognized user
                elif result["display_user_id"] != "Unknown":
                    # Green for recognized users
                    rect_color = (0, 255, 0)  # Green
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), rect_color, 2)
                    
                    # Get name and job
                    name_text = result["display_name"] if result["display_name"] else result["display_user_id"]
                    job_text = result["display_job"] if result["display_job"] else ""
                    
                    # Draw name text with Chinese support
                    display_frame = draw_chinese_text(display_frame, name_text, (x, y-25), (0, 255, 0), 20)
                    
                    # Draw job title below if available
                    if job_text:
                        display_frame = draw_chinese_text(display_frame, job_text, (x, y+h+5), (0, 255, 0), 20)
                
                # If it's an unknown face
                else:
                    # Yellow for unknown faces
                    rect_color = (0, 255, 255)  # Yellow
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), rect_color, 2)
                    # Draw text
                    display_frame = draw_chinese_text(display_frame, "未知人员", (x, y-25), (0, 255, 255), 20)

            # Update the last display frame reference
            last_display_frame = display_frame.copy()
            # Update the last display frame reference
            last_display_frame = display_frame.copy()
            # Create display image with double-buffering (prepare new image before displaying)
            try:
                # Resize first for efficiency
                display_small = cv2.resize(display_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), 
                                         interpolation=cv2.INTER_AREA)
                rgb_small = cv2.cvtColor(display_small, cv2.COLOR_BGR2RGB)
                
                # Use PIL for efficient conversion
                pil_img = Image.fromarray(rgb_small)
                new_tk_img = ImageTk.PhotoImage(image=pil_img)
                
                # Store new image
                next_display_img = new_tk_img
                
                # Only update the display when we have a valid image ready
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=next_display_img)
                self.video_canvas.img = next_display_img  # Update reference after display is ready
                
                # Update our saved display image
                display_tk_img = next_display_img
                
            except Exception as e:
                print(f"显示错误: {e}")
            
            # Add a brief delay to maintain consistent frame pacing
            time.sleep(0.033)  # ~30fps display rate
        
        # Stop the frame grabbing thread
        self.stop_recognition = True
        if grab_thread.is_alive():
            grab_thread.join(timeout=1.0)
        
        # Clean up
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        
        self.is_camera_active = False
        
        # Display message
        self.video_canvas.delete("all")
        self.video_canvas.create_text(DISPLAY_WIDTH//2, DISPLAY_HEIGHT//2, 
                                    text="点击 '开始识别' 按钮启动摄像头", 
                                    fill="white", font=("SimHei", 12), tags="message")
        
        # Clear references to help garbage collection
        display_tk_img = None
        last_display_frame = None
        frames_queue.clear()
        face_results.clear()  # Clear face results dictionary
        
        # Force garbage collection
        gc.collect()
        print("Recognition stopped and resources released")
    
    def add_user(self):
        """User registration with job title selection and confirm button"""
        if self.is_camera_active:
            messagebox.showerror("错误", "请先停止识别！")
            return
        
        # Get user info
        user_id = simpledialog.askstring("输入", "用户ID:", parent=self.root)
        if not user_id:
            return
            
        if self.user_manager.user_exists(user_id):
            messagebox.showerror("错误", f"用户已存在: {user_id}")
            return
            
        name = simpledialog.askstring("输入", "姓名:", parent=self.root)
        if not name:
            return
        
        # Let user select job title
        job_window = tk.Toplevel(self.root)
        job_window.title("选择职位")
        job_window.geometry("300x250")  # Made taller for button
        job_window.resizable(False, False)
        job_window.transient(self.root)
        job_window.grab_set()
        
        job_var = tk.StringVar(value=JOB_TITLES[0])  # Default to first job
        
        # Create job selection frame
        job_frame = ttk.Frame(job_window, padding="10")
        job_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add label
        ttk.Label(job_frame, text="选择职位:", font=("SimHei", 10)).pack(pady=10)
        
        # Add radio buttons for job titles
        for job in JOB_TITLES:
            ttk.Radiobutton(job_frame, text=job, variable=job_var, value=job).pack(anchor=tk.W, pady=5)
        
        # Add buttons frame
        button_frame = ttk.Frame(job_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        # Add confirm and cancel buttons
        job_selected = [False]  # Use list to modify in inner function
        selected_job = [JOB_TITLES[0]]  # Default
        
        def on_job_confirm():
            selected_job[0] = job_var.get()
            job_selected[0] = True
            job_window.destroy()
        
        def on_job_cancel():
            job_window.destroy()
            
        # Add proper confirm and cancel buttons
        ttk.Button(button_frame, text="确认", command=on_job_confirm, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="取消", command=on_job_cancel, width=10).pack(side=tk.RIGHT, padx=10)
        
        # Center the window on parent
        job_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - job_window.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - job_window.winfo_height()) // 2
        job_window.geometry(f"+{x}+{y}")
        
        # Wait for window to close
        self.root.wait_window(job_window)
        
        if not job_selected[0]:
            return  # User canceled
        
        job = selected_job[0]
        
        # Ask for number of samples
        samples = simpledialog.askinteger("输入", "样本数量:", 
                                         minvalue=1, maxvalue=5, initialvalue=3)
        if not samples:
            return
        
        # Register with optimized parameters
        self.register_user(user_id, name, job, samples)
    
    def add_user_with_data(self, user_data):
        """Add a user with data from IPC command"""
        user_id = user_data.get("user_id")
        name = user_data.get("name")
        job = user_data.get("job", JOB_TITLES[0])
        samples = user_data.get("samples", 3)
        
        if not user_id or not name:
            print("Invalid user data from IPC command")
            return
        
        self.register_user(user_id, name, job, samples)

    def register_user(self, user_id, name, job, num_samples):
        """Optimized user registration with face size check"""
        # Create window
        reg_window = tk.Toplevel(self.root)
        reg_window.title(f"添加用户: {name}")
        reg_window.geometry(f"{DISPLAY_WIDTH+20}x{DISPLAY_HEIGHT+80}")
        
        # Create canvas
        canvas = tk.Canvas(reg_window, bg="black", width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        canvas.pack(padx=5, pady=5)
        
        # Status label
        status_label = ttk.Label(reg_window, text=f"准备采集 1/{num_samples}...",
                               font=("SimHei", 10))
        status_label.pack(pady=2)
        
        # Create user with job title
        self.user_manager.create_user(user_id, name, job)
        
        # Open camera with reduced resolution
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("错误", "无法打开摄像头！")
            reg_window.destroy()
            return
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        
        # Variables for capture
        samples_captured = 0
        last_capture_time = 0
        display_tk_img = None
        
        def update_frame():
            nonlocal samples_captured, last_capture_time, display_tk_img
            
            # Check if done
            if samples_captured >= num_samples:
                # Save model
                self.face_recognizer.save_model(RECOGNITION_MODEL_PATH)
                messagebox.showinfo("完成", f"用户添加成功: {name}")
                camera.release()
                reg_window.destroy()
                gc.collect()  # Force cleanup
                return
            
            # Read frame
            ret, frame = camera.read()
            if not ret:
                reg_window.after(10, update_frame)
                return
            
            # Mirror
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect_face(frame)
            
            # Draw faces
            for (x, y, w, h) in faces:
                # Check face size during registration
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_proportion = face_area / frame_area
                
                # Show different colors based on face size
                if face_proportion < MIN_REGISTER_PROPORTION:
                    # Yellow for faces that are too small
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                else:
                    # Green for good-sized faces
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, f"样本 {samples_captured+1}/{num_samples}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # Capture if conditions met
            current_time = time.time()
            if len(faces) == 1 and (current_time - last_capture_time) > 1.5:  # Reduced delay
                x, y, w, h = faces[0]
                
                # Check face size before capture
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_proportion = face_area / frame_area
                
                if face_proportion < MIN_REGISTER_PROPORTION:
                    # Face too small - add instruction to move closer
                    cv2.putText(frame, "请靠近摄像头", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Only capture if face is large enough
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Save and add to recognizer
                    self.user_manager.add_face_sample(user_id, face_img)
                    self.face_recognizer.add_face(user_id, face_img)
                    
                    samples_captured += 1
                    last_capture_time = current_time
                    
                    status_label.configure(text=f"已采集 {samples_captured}/{num_samples}" +
                                          (", 完成" if samples_captured >= num_samples else ""))
            
            # Display frame 
            try:
                display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                new_tk_img = ImageTk.PhotoImage(image=pil_img)
                
                display_tk_img = new_tk_img
                canvas.delete("all")
                canvas.create_image(0, 0, anchor=tk.NW, image=display_tk_img)
                canvas.img = display_tk_img
            except Exception as e:
                print(f"显示错误: {e}")
            
            # Continue updating
            reg_window.after(30, update_frame)
        
        # Start updating
        update_frame()
        
        # Handle window closing
        def on_closing():
            camera.release()
            reg_window.destroy()
            gc.collect()
            
        reg_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    def show_user_management(self):
        """Simplified user management"""
        if self.is_camera_active:
            messagebox.showerror("错误", "请先停止识别！")
            return
        
        # Get users
        users = self.user_manager.get_all_users()
        
        # Create window
        mgmt_window = tk.Toplevel(self.root)
        mgmt_window.title("用户管理")
        mgmt_window.geometry("400x300")
        
        # Create frame
        frame = ttk.Frame(mgmt_window, padding="5")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create listbox with scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, width=40, height=10, font=("SimHei", 10),
                           yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=listbox.yview)
        
        # Add users to list
        if not users:
            listbox.insert(tk.END, "无用户")
        else:
            for user in users:
                job = user.get('job', "")
                display = f"{user['id']} - {user['name']} - {job}" if job else f"{user['id']} - {user['name']}"
                listbox.insert(tk.END, display)
        
        # Button frame
        button_frame = ttk.Frame(mgmt_window, padding="5")
        button_frame.pack(fill=tk.X)
        
        # Buttons
        delete_button = ttk.Button(button_frame, text="删除", 
                                 command=lambda: self.delete_selected_user(listbox, mgmt_window))
        delete_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        reset_button = ttk.Button(button_frame, text="全部删除", 
                                command=lambda: self.delete_all_users(mgmt_window))
        reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        close_button = ttk.Button(button_frame, text="关闭", 
                                command=mgmt_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def delete_selected_user(self, listbox, parent_window):
        """Delete selected user"""
        selection = listbox.curselection()
        
        if not selection:
            messagebox.showerror("错误", "请选择用户", parent=parent_window)
            return
        
        item = listbox.get(selection[0])
        
        if item == "无用户":
            return
        
        user_id = item.split(" - ")[0]
        
        if not messagebox.askyesno("确认", f"确定删除用户 {user_id}?", parent=parent_window):
            return
        
        try:
            self.delete_user(user_id)
            messagebox.showinfo("成功", f"已删除用户: {user_id}", parent=parent_window)
            
            parent_window.destroy()
            self.show_user_management()
        except Exception as e:
            messagebox.showerror("错误", f"删除错误: {str(e)}", parent=parent_window)
    
    def delete_all_users(self, parent_window):
        """Delete all users"""
        if not messagebox.askyesno("确认", "确定删除所有用户?", parent=parent_window):
            return
        
        try:
            self.reset_system()
            messagebox.showinfo("成功", "已删除所有用户", parent=parent_window)
            
            parent_window.destroy()
            self.show_user_management()
        except Exception as e:
            messagebox.showerror("错误", f"错误: {str(e)}", parent=parent_window)
    
    def delete_user(self, user_id):
        """Delete a user"""
        # Check if exists
        if not self.user_manager.user_exists(user_id):
            raise Exception(f"用户不存在: {user_id}")
        
        # Delete from user manager
        if not self.user_manager.delete_user(user_id):
            raise Exception("删除失败")
        
        # Rebuild model
        self.face_recognizer = FaceRecognizer()
        
        # Get remaining users
        remaining_users = self.user_manager.get_all_users()
        
        # Add samples from remaining users
        for user_info in remaining_users:
            current_id = user_info["id"]
            face_samples = self.user_manager.get_user_face_samples(current_id)
            for face in face_samples:
                self.face_recognizer.add_face(current_id, face)
        
        # Save updated model
        if os.path.exists(RECOGNITION_MODEL_PATH):
            os.remove(RECOGNITION_MODEL_PATH)
        
        # Create directory
        os.makedirs(os.path.dirname(RECOGNITION_MODEL_PATH), exist_ok=True)
        
        # Save model
        if not self.face_recognizer.save_model(RECOGNITION_MODEL_PATH):
            raise Exception("保存模型失败")
    
    def reset_system(self):
        """Reset system"""
        # Delete model
        if os.path.exists(RECOGNITION_MODEL_PATH):
            os.remove(RECOGNITION_MODEL_PATH)
        
        # Clear user data
        user_data_dir = "data/users"
        if os.path.exists(user_data_dir):
            for root, dirs, files in os.walk(user_data_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
        
        # Recreate directory
        os.makedirs(user_data_dir, exist_ok=True)
        
        # Create fresh recognizer
        self.face_recognizer = FaceRecognizer()
        
        # Save empty model
        os.makedirs(os.path.dirname(RECOGNITION_MODEL_PATH), exist_ok=True)
        self.face_recognizer.save_model(RECOGNITION_MODEL_PATH)
        
        # Force cleanup
        gc.collect()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_camera_active:
            self.stop_recognition = True
            if self.recognition_thread:
                self.recognition_thread.join(timeout=1.0)
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()