import cv2
import numpy as np
import time

class AntiSpoofDetector:
    def __init__(self):
        # Initialize parameters with values optimized for STM32MP257
        self.motion_threshold = 0.5
        self.texture_threshold = 50  # Reduced threshold for faster processing
        
        # For motion detection
        self.prev_frame = None
        self.motion_scores = []
        self.motion_direction_history = []
        self.last_motion_time = time.time()
        
        # For tracking frames
        self.frame_count = 0
        
        # Simplified state tracking for embedded system
        self.is_video_cache = False
        self.video_detection_timeout = 0
        self.last_reset_time = time.time()

    def check_liveness(self, frame, face_box):
        """Simplified liveness check optimized for STM32MP257"""
        x, y, w, h = face_box
        
        # Ensure face box is valid
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return True, 0.8  # Default to real if invalid box
        
        face = frame[y:y+h, x:x+w]
        
        # Skip tiny faces (prevents errors and saves resources)
        if face.size == 0 or w < 20 or h < 20:
            return True, 0.8
        
        # Only use texture check for efficiency on embedded system
        # This is the most lightweight yet effective check
        texture_score = self._check_texture(face)
        
        # Add motion check only occasionally (every 5th frame) to save resources
        if getattr(self, 'frame_count', 0) % 5 == 0:
            motion_score = self._check_motion(face)
            # Combine scores with more weight on texture (faster)
            combined_score = 0.8 * texture_score + 0.2 * motion_score
        else:
            # Use only texture score most of the time
            combined_score = texture_score
        
        # Increment frame counter
        self.frame_count = getattr(self, 'frame_count', 0) + 1
        if self.frame_count > 1000:
            self.frame_count = 0
        
        # Lower threshold for less strict detection (saves false positives)
        is_live = combined_score > 0.35
        return is_live, combined_score
        
    def _check_texture(self, face):
        """Optimized texture check for embedded systems"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (key indicator for screen detection)
            # This is computationally efficient yet effective
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            
            # Scale the result with optimized threshold
            texture_score = min(laplacian_var / self.texture_threshold, 1.0)
            
            # Boost score slightly to reduce false positives
            texture_score = min(1.0, texture_score * 1.2)
            
            return texture_score
        except Exception:
            # Return higher score on error to avoid false rejections
            return 0.8
    def _update_frame_history(self, face):
        """Update the frame history buffer for video detection"""
        try:
            # Resize to very small size for faster processing
            small_face = cv2.resize(face, (32, 32))
            gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            
            # Add to history
            self.frame_history.append(gray)
            
            # Keep limited history
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
        except Exception:
            # Clear history if error
            self.frame_history = []
    
    def _check_video_patterns(self, face):
        """Specialized detection for video playback patterns - more lenient now"""
        # Need multiple frames for analysis
        if len(self.frame_history) < 4:
            return 0.7, False  # Higher neutral score if not enough history
        
        try:
            # 1. Check for repeating patterns (looping videos) - less weight
            is_repeating = self._detect_repeating_patterns() * 0.7  # Reduced impact by 30%
            
            # 2. Check for screen refresh patterns - less weight
            has_refresh_artifacts = self._detect_refresh_artifacts() * 0.7  # Reduced impact
            
            # 3. Check for compression artifacts - less weight
            compression_level = self._detect_compression_artifacts(face) * 0.6  # Reduced impact
            
            # 4. Check for color banding - less weight
            has_color_banding = self._detect_color_banding(face) * 0.6  # Reduced impact
            
            # Combine video-specific indicators with reduced weights
            video_indicators = [
                is_repeating * 0.3,            # Reduced from 0.4
                has_refresh_artifacts * 0.2,   # Reduced from 0.3
                compression_level * 0.1,       # Reduced from 0.2
                has_color_banding * 0.05       # Reduced from 0.1
            ]
            
            # Calculate video detection score with an offset to be more lenient
            video_detection_score = 1.0 - sum(video_indicators)
            
            # Add leniency offset
            video_detection_score = min(1.0, video_detection_score + 0.2)
            
            # Require stronger evidence to classify as video
            is_likely_video = video_detection_score < 0.5  # Changed from 0.6
            
            return video_detection_score, is_likely_video
        except Exception:
            return 0.7, False  # Default to higher score (real) if error
    
    def _detect_repeating_patterns(self):
        """Detect repeating patterns in frame sequence (video loops) - more lenient"""
        # Need enough frames for this check
        if len(self.frame_history) < 6:
            return 0.0
        
        try:
            # Compare current frame with frames from a few steps back
            current = self.frame_history[-1].flatten()
            
            # Check correlation with frames at different distances back
            correlations = []
            for i in range(2, min(6, len(self.frame_history))):
                past = self.frame_history[-(i+1)].flatten()
                corr = np.corrcoef(current, past)[0, 1]
                correlations.append(corr)
            
            # High correlation with a past frame suggests repeating content
            # Using higher thresholds to be more lenient
            max_correlation = max(correlations) if correlations else 0
            
            # Scale the result with higher thresholds
            if max_correlation > 0.98:  # Was 0.95
                return 1.0
            elif max_correlation > 0.95:  # Was 0.85
                return 0.7
            elif max_correlation > 0.90:  # Was 0.75
                return 0.3
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _detect_refresh_artifacts(self):
        """Detect screen refresh artifacts - more lenient"""
        # Need multiple consecutive frames
        if len(self.frame_history) < 3:
            return 0.0
        
        try:
            # Look at differences between consecutive frames
            diffs = []
            for i in range(len(self.frame_history) - 1):
                diff = cv2.absdiff(self.frame_history[i], self.frame_history[i+1])
                row_means = np.mean(diff, axis=1)
                diffs.append(row_means)
            
            # Convert to numpy array for analysis
            diffs_array = np.array(diffs)
            
            # Calculate row-wise variance
            row_variances = np.var(diffs_array, axis=0)
            
            # Look for alternating high/low patterns
            pattern_strength = 0.0
            if len(row_variances) > 10:
                # Calculate differences between adjacent rows
                row_diff = np.abs(np.diff(row_variances))
                
                # Count significant alternating patterns with higher threshold
                alternating_count = 0
                for i in range(len(row_diff) - 1):
                    if (row_diff[i] > np.mean(row_diff) * 3 and  # Increased from 2
                        row_diff[i+1] > np.mean(row_diff) * 3):   # Increased from 2
                        alternating_count += 1
                
                # Calculate pattern strength with reduced sensitivity
                pattern_strength = min(1.0, alternating_count / 7.0)  # Was 5.0
            
            return pattern_strength
        except Exception:
            return 0.0
    
    def _detect_compression_artifacts(self, face):
        """Detect video compression artifacts - more lenient"""
        try:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
            y, _, _ = cv2.split(ycrcb)
            
            # Apply high-pass filter to find compression block boundaries
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered_y = cv2.filter2D(y, -1, kernel)
            
            # Threshold to find edges with higher threshold
            _, edges = cv2.threshold(filtered_y, 30, 255, cv2.THRESH_BINARY)  # Was 20
            
            # Look for grid-like patterns (JPEG/MPEG compression blocks)
            h, w = edges.shape
            
            # Projections
            horizontal_projection = np.sum(edges, axis=0)
            vertical_projection = np.sum(edges, axis=1)
            
            # Detect regular peaks in projections
            regularity_score = 0.0
            
            # Function to detect regular peaks
            def detect_regular_peaks(projection):
                if len(projection) < 16:
                    return 0.0
                    
                # Find peaks
                peaks = []
                for i in range(1, len(projection) - 1):
                    if (projection[i] > projection[i-1] and 
                        projection[i] > projection[i+1] and
                        projection[i] > np.mean(projection) * 1.5):  # Increased from 1.0
                        peaks.append(i)
                
                # Check distances between peaks
                if len(peaks) < 4:  # Was 3
                    return 0.0
                    
                distances = np.diff(peaks)
                
                # Calculate regularity
                if len(distances) == 0:
                    return 0.0
                    
                distance_var = np.var(distances) / np.mean(distances)
                
                # Lower variance = more regular
                regularity = max(0.0, 1.0 - min(1.0, distance_var / 0.3))  # Was 0.5
                
                return regularity
            
            h_reg = detect_regular_peaks(horizontal_projection)
            v_reg = detect_regular_peaks(vertical_projection)
            
            # Combine horizontal and vertical regularity
            regularity_score = max(h_reg, v_reg)
            
            # Scale result with reduced sensitivity
            compression_level = min(1.0, regularity_score * 1.2)  # Was 1.5
            
            return compression_level
        except Exception:
            return 0.0
    
    def _detect_color_banding(self, face):
        """Detect color banding - more lenient"""
        try:
            # Check for color banding in each channel
            b, g, r = cv2.split(face)
            
            # Calculate histograms
            hist_b = cv2.calcHist([b], [0], None, [32], [0, 256])  # Reduced bins
            hist_g = cv2.calcHist([g], [0], None, [32], [0, 256])
            hist_r = cv2.calcHist([r], [0], None, [32], [0, 256])
            
            # Normalize histograms
            hist_b = hist_b / np.sum(hist_b)
            hist_g = hist_g / np.sum(hist_g)
            hist_r = hist_r / np.sum(hist_r)
            
            # Check for "combing" in histogram with higher threshold
            banding_scores = []
            for hist in [hist_b, hist_g, hist_r]:
                # Calculate differences between adjacent bins
                diffs = np.abs(np.diff(hist.flatten()))
                
                # Count significant spikes with higher threshold
                spikes = np.sum(diffs > np.mean(diffs) * 3)  # Was 2
                
                # Calculate banding score with reduced sensitivity
                banding_score = min(1.0, spikes / 15.0)  # Was 10.0
                banding_scores.append(banding_score)
            
            # Take maximum banding score across channels
            max_banding = max(banding_scores)
            
            return max_banding
        except Exception:
            return 0.0
    
    def _check_motion(self, face):
        """Simplified motion check for embedded systems"""
        try:
            # Use smaller size for faster processing
            small_face = cv2.resize(face, (24, 24))  # Reduced from 32x32
            gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return 0.7  # Default to higher score (assume real)
                
            # Calculate frame difference (simple and fast)
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            motion_amount = np.mean(frame_diff)
            
            # Add to motion history (keep shorter history)
            self.motion_scores.append(motion_amount)
            if len(self.motion_scores) > 5:  # Reduced from 10
                self.motion_scores.pop(0)
                
            # Calculate motion score with simpler logic
            if len(self.motion_scores) >= 3:  # Need at least 3 samples
                motion_var = np.var(self.motion_scores)
                
                # Very simple heuristic based on motion variance
                if motion_var < 0.1:  # Almost no variance
                    motion_score = 0.5  # Neutral
                elif motion_var > 10.0:  # Too much variance
                    motion_score = 0.5  # Neutral
                else:
                    # Natural range - higher variance = more likely real
                    motion_score = min(1.0, 0.6 + motion_var * 0.05)
            else:
                motion_score = 0.7  # Default higher
                
            # Update previous frame
            self.prev_frame = gray
            
            return motion_score
        except Exception:
            # Handle errors gracefully
            return 0.7  # Default to higher score on error
    def reset(self):
        """Reset the detector state"""
        self.prev_frame = None
        self.motion_scores = []
        self.motion_direction_history = []
        self.frame_history = []
        self.is_video_cache = False
        self.consecutive_video_detections = 0