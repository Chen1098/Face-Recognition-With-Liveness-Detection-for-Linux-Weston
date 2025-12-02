import json
import os
import time
import platform

class IPCManager:
    """Cross-platform IPC manager for communication with C++ applications"""
    
    def __init__(self):
        # Set different paths based on operating system
        if platform.system() == "Windows":
            # Use temp directory on Windows
            temp_dir = os.environ.get("TEMP", "C:/temp")
            self.status_file = os.path.join(temp_dir, "face_recognition_status.json")
            self.command_file = os.path.join(temp_dir, "face_recognition_command.json")
            
            # Create temp directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
        else:
            # Use /tmp on Linux
            self.status_file = "/tmp/face_recognition_status.json"
            self.command_file = "/tmp/face_recognition_command.json"
        
        self.last_status = {}
        self.last_update_time = 0
        print(f"IPC initialized. Status file: {self.status_file}")
    
    def update_status(self, status_dict):
        """Update status file for C++ applications to read"""
        current_time = time.time()
        
        # Only update at most once per second to reduce I/O
        if current_time - self.last_update_time >= 1.0:
            try:
                with open(self.status_file, 'w') as f:
                    json.dump(status_dict, f)
                self.last_status = status_dict.copy()
                self.last_update_time = current_time
            except Exception as e:
                print(f"Failed to update IPC status: {e}")
    
    def check_commands(self):
        """Check for commands from C++ applications"""
        try:
            if os.path.exists(self.command_file):
                # Get file modification time
                mod_time = os.path.getmtime(self.command_file)
                
                # Only process if file is recent (last 5 seconds)
                if time.time() - mod_time <= 5.0:
                    with open(self.command_file, 'r') as f:
                        command = json.load(f)
                    
                    # Delete the command file after reading
                    os.remove(self.command_file)
                    
                    return command
        except Exception as e:
            print(f"Error checking commands: {e}")
            
        return None