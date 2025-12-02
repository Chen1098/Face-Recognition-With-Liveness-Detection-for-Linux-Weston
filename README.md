# Face-Recognition-With-Liveness-Detection-for-Linux-Weston
A Face Recognition With Liveness Detection for Linux Weston

# Face Recognition System with Anti-Spoofing

## System Overview

This is a face recognition system with anti-spoofing capabilities that can identify registered users' identities and positions, and detect photo or video spoofing attempts.

This system features the following functions:

- User registration and management
- Real-time face recognition
- Position display (Mechanical and Electrical Section Chief, Mechanical and Electrical Deputy Section Chief, Mechanical and Electrical Team Leader, Mechanical and Electrical Deputy Team Leader)
- Anti-photo/video spoofing detection
- Distance detection (face must be sufficiently close to the camera)
- Fully Chinese interface

## Windows Installation Guide

### System Requirements

- Windows 7/8/10/11
- Python 3.9+
- Webcam
- Minimum 2GB RAM
- 50MB disk space

### Step 1: Install Python

1. Download and install Python 3.9 or higher from the [Python official website](https://www.python.org/downloads/)
2. During installation, check the "Add Python to PATH" option

### Step 2: Download the Project

1. Create a folder for the project, e.g., `C:\FaceRecognition`
2. Copy the project files to this folder, ensuring the original directory structure is maintained

### Step 3: Create a Virtual Environment and Install Dependencies

Open the Command Prompt (CMD), then run the following commands:

```
cd C:\FaceRecognition
python -m venv venv
venv\Scripts\activate
pip install opencv-python numpy pillow scikit-learn
```

### Step 4: Install Chinese Fonts (If Needed)

If SimHei or SimSun fonts are not installed in the system, download them from the following locations:
- SimHei: [Download Link](https://www.fontpalace.com/font-details/SimHei/)
- SimSun: [Download Link](https://www.fontpalace.com/font-details/SimSun/)

After downloading, double-click the font file to install it.

### Step 5: Run the Program

```
python face_recognition_gui.py
```

## Usage Guide

### Adding a New User

1. Click the "Add User" button
2. Enter the user ID and name
3. Select the position
4. After confirmation, follow the on-screen instructions to capture face samples

### Recognizing a User

1. Click the "Start Recognition" button
2. Align your face with the camera
3. The system will display the recognition result

### Managing Users

1. Click the "Manage Users" button
2. View or delete users

### Resetting the System

1. Click the "Reset System" button
2. Confirm to delete all user data

## System Adjustments

### Adjusting Anti-Spoofing Detection Sensitivity

If you need to increase or decrease the anti-spoofing detection sensitivity, edit the `utils/anti_spoof.py` file:

#### Increase Sensitivity (Easier to Detect Spoofing)

```python
# Find this line in the check_liveness method
is_live = combined_score > 0.4
# Change to
is_live = combined_score > 0.5  # Increase threshold

# Find these values in the AntiSpoofDetector class initialization method
self.texture_threshold = 60
self.required_video_detections = 4
# Change to
self.texture_threshold = 50  # Decrease threshold
self.required_video_detections = 3  # Decrease detection count requirement
```

#### Decrease Sensitivity (Reduce False Positives)

```python
# Find this line in the check_liveness method
is_live = combined_score > 0.4
# Change to
is_live = combined_score > 0.3  # Decrease threshold

# Find these values in the AntiSpoofDetector class initialization method
self.texture_threshold = 60
self.required_video_detections = 4
# Change to
self.texture_threshold = 70  # Increase threshold
self.required_video_detections = 5  # Increase detection count requirement
```

### Adjusting Face Size Requirements

If you need to change the face size requirements, edit the constants at the top of the `face_recognition_gui.py` file:

```python
# Find the following constants
MIN_FACE_PROPORTION = 0.05  # Minimum face proportion during recognition
MIN_REGISTER_PROPORTION = 0.1  # Minimum face proportion during registration
# Adjust to desired values
MIN_FACE_PROPORTION = 0.03  # Decrease proportion to allow farther distance recognition
MIN_REGISTER_PROPORTION = 0.08  # Decrease proportion to allow farther distance registration
```

### Adjusting Recognition Refresh Rate

If you need to increase the refresh rate or reduce system resource usage, edit the constants at the top of the `face_recognition_gui.py` file:

```python
# Find the following constants
PROCESS_EVERY_N_FRAMES = 3  # Process every N frames
RECOGNITION_INTERVAL = 0.1  # Processing interval in seconds
# Adjust values
PROCESS_EVERY_N_FRAMES = 2  # Decrease frame count to increase processing frequency
RECOGNITION_INTERVAL = 0.05  # Decrease interval to increase processing frequency
# Or decrease processing frequency to save resources
PROCESS_EVERY_N_FRAMES = 5  # Increase frame count to decrease processing frequency
RECOGNITION_INTERVAL = 0.2  # Increase interval to decrease processing frequency
```

### Adjusting Camera Resolution

If you need to adjust the camera resolution, edit the constants at the top of the `face_recognition_gui.py` file:

```python
# Find the following constants
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240
DISPLAY_WIDTH = 480
DISPLAY_HEIGHT = 360
# Adjust to desired resolution
CAPTURE_WIDTH = 640  # Increase capture resolution
CAPTURE_HEIGHT = 480
DISPLAY_WIDTH = 800  # Increase display resolution
DISPLAY_HEIGHT = 600
# Or decrease resolution for better performance
CAPTURE_WIDTH = 240  # Decrease capture resolution
CAPTURE_HEIGHT = 180
DISPLAY_WIDTH = 320  # Decrease display resolution
DISPLAY_HEIGHT = 240
```

## Common Issues and Solutions (Troubleshooting)

### System Fails to Display Chinese Correctly

Ensure SimHei or SimSun fonts are installed. If issues persist, try modifying the `draw_chinese_text` function:

```python
font = ImageFont.truetype("simhei.ttf", font_size)
```

Change to:

```python
# Copy the font file to the project directory and use the full path
font = ImageFont.truetype("C:/FaceRecognition/fonts/simhei.ttf", font_size)
```

### Camera Not Opening

Ensure a camera device is available. If multiple cameras are present, modify the following in the `recognition_loop` and `register_user` methods:

```python
self.camera = cv2.VideoCapture(0)  # 0 is the default camera
```

Change to:

```python
self.camera = cv2.VideoCapture(1)  # Use camera with device index 1
```

### Low Recognition Accuracy

1. Increase the number of training samples - Provide more samples (3-5) when adding users
2. Ensure sufficient lighting
3. Capture face samples from different angles during registration
4. Adjust the recognition threshold - Modify the `confidence_threshold` parameter in `face_recognizer.py`

### System Running Slowly

1. Decrease camera resolution
2. Increase `PROCESS_EVERY_N_FRAMES` and `RECOGNITION_INTERVAL` values
3. Close other running programs
4. For low-spec devices, remove certain detection methods in `anti_spoof.py`

And here's the Linux version:

```markdown
# Face Recognition System with Anti-Spoofing

## System Overview

This is a face recognition system with anti-spoofing capabilities that can identify registered users' identities and positions, and detect photo or video spoofing attempts.

This system features the following functions:

- User registration and management
- Real-time face recognition
- Position display (Mechanical and Electrical Section Chief, Mechanical and Electrical Deputy Section Chief, Mechanical and Electrical Team Leader, Mechanical and Electrical Deputy Team Leader)
- Anti-photo/video spoofing detection
- Distance detection (face must be sufficiently close to the camera)
- Fully Chinese interface

## Linux Installation Guide

### System Requirements

- Any mainstream Linux distribution (Ubuntu, Debian, Fedora, CentOS, etc.)
- Python 3.9+
- Webcam
- Minimum 2GB RAM
- 50MB disk space

### Step 1: Install System Dependencies

For Ubuntu/Debian systems:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-tk python3-dev python3-pil.imagetk libopencv-dev
```
```
