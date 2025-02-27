"""
Advanced Video Quality Analyzer with Metadata and Visual Assessment
"""

import os
import subprocess
import json
import csv
import logging
import concurrent.futures
from datetime import datetime
from time import time
from tqdm import tqdm
import cv2
import numpy as np
from skimage import metrics

# Configuration - Adjust these values as needed
CONFIG = {
    # Core functionality
    'OUTPUT_COLUMNS': [
        'file_name', 'folder_path', 'file_path', 'error_flags','creation_date', 'modified_date',
        'duration_sec', 'duration_hms', 'file_size (bytes)', 'file_size_mb (MB)',
        'container', 'video_codec_technical', 'video_codec_common',
        'width (px)', 'height (px)', 'video_bitrate (kBit/s)',
        'compression_ratio', 'compression_quality',
        'conversion_potential_h265', 'conversion_quality_h265',
        'conversion_potential_av1', 'conversion_quality_av1',
        'problem_flag', 'quality_estimate', 
        'vqa_score', 'vqa_algorithm_used',
    ],
    
    # Quality estimation parameters
    'QUALITY_THRESHOLDS': {
        'bitrate': {
            '1080p': 5000,  # kbit/s
            '720p': 2500,
            '480p': 1000
        },
        'size_duration': {
            '1080p': 1.5,  # MB per minute
            '720p': 0.8,
            '480p': 0.3
        }
    },

    # Fixed threshold keys
    'COMPRESSION_RATIO_THRESHOLDS': {
        'good': 0.03,
        'questionable': 0.07,
        'poor': 0.1
    },
    
    'CONVERSION_THRESHOLDS': {
        'High': 0.3,
        'Medium': 0.1,
        'Low': 0
    },
    
    'CODEC_EFFICIENCY': {
        'h264': 1.0, 'hevc': 0.65, 'av1': 0.5, 
        'vp9': 0.7, 'mpeg4': 1.3
    },
    
    # VQA configuration
    'ENABLE_VQA': False,
    'VQA_SAMPLE_RATE': 0.01,  # 1% of frames
    'VQA_FRAME_SAMPLING': 'fixed',  # 'fixed' or 'random'
    'VQA_METADATA_FILTER': 'poor',  # 'all', 'good', 'poor'
    
    # Error detection
    'ERROR_DETECTION': {
        'min_color_variation': 0.1,
        'max_consecutive_similar_frames': 10,
        'laplacian_noise_threshold': 15
    },

    'QUALITY_THRESHOLDS': {
        'bitrate': {
            '1080p': 5000,  # kbit/s
            '720p': 2500,
            '480p': 1000,
            'default': 1000
        },
        'size_duration': {
            '1080p': 1.5,  # MB per minute
            '720p': 0.8,
            '480p': 0.3,
            'default': 0.1
        }
    },
    'VQA': {
        'ENABLED': False,
        'METADATA_FILTER': 'poor'
    },
    
    # Performance
    'MAX_WORKERS': os.cpu_count() // 2,
    'USE_GPU': False
}

CODEC_MAP = {
    'hevc': 'H.265', 'h265': 'H.265', 'avc': 'H.264', 'h264': 'H.264',
    'avc1': 'H.264', 'vp9': 'VP9', 'av1': 'AV1', 'mpeg4': 'MPEG-4',
    'aac': 'AAC', 'mp3': 'MP3', 'ac3': 'Dolby Digital', 'dts': 'DTS'
}

# Configure logging
logging.basicConfig(
    filename='video_analysis_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VideoAnalyzer:
    def __init__(self):
        self.problem_flags = []
        self.error_flags = []
        self.vqa_results = {}
        
    def calculate_brisque(self, frame):
        """Calculate BRISQUE score for a frame (simplified example)"""
        if CONFIG['USE_GPU']:
            # GPU-accelerated implementation would go here
            pass
        return np.random.normal(40, 10)  # Placeholder
        
    def analyze_frame(self, frame):
        """Analyze in-memory frame"""
        try:
            if frame is None or frame.size == 0:
                return None
                
            # Convert to float32 for processing
            frame = frame.astype(np.float32)
            
            # Calculate quality metrics
            return {
                'color_variation': np.std(frame),
                'sharpness': cv2.Laplacian(frame, cv2.CV_32F).var()
            }
        except Exception as e:
            logging.error(f"Frame analysis error: {str(e)}")
            return None

    def sample_frames(self, video_path):
        """Sample frames from video based on config"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = max(1, int(total_frames * CONFIG['VQA_SAMPLE_RATE']))
        
        if CONFIG['VQA_FRAME_SAMPLING'] == 'fixed':
            sample_indices = np.linspace(0, total_frames-1, sample_count, dtype=int)
        else:
            sample_indices = np.random.choice(total_frames, sample_count, replace=False)
            
        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames

    def detect_errors(self, frames):
        """Detect video errors with comprehensive validation"""
        error_flags = set()
        prev_frame = None
        similar_count = 0
        max_retries = 3
        
        for frame in frames:
            # Skip invalid frames
            if frame is None or frame.size == 0:
                error_flags.add('invalid_frame')
                continue
                
            # Validate frame dimensions
            try:
                h, w = frame.shape[:2]
                if h < 7 or w < 7:
                    error_flags.add('small_frame_size')
                    continue
                    
                # Dynamic window size calculation
                win_size = min(7, min(h, w))
                win_size = win_size if win_size % 2 else win_size - 1
            except AttributeError as e:
                logging.error(f"Frame dimension error: {str(e)}")
                continue

            # Detect single-color frames
            try:
                color_std = np.std(frame)
                if color_std < self.config['ERROR_DETECTION']['min_color_variation']:
                    error_flags.add('low_color_variation')
            except Exception as e:
                logging.error(f"Color variation error: {str(e)}")

            # Detect consecutive similar frames
            if prev_frame is not None:
                for attempt in range(max_retries):
                    try:
                        ssim = metrics.structural_similarity(
                            prev_frame, frame,
                            win_size=win_size,
                            channel_axis=2 if frame.ndim == 3 else None
                        )
                        if ssim > 0.98:
                            similar_count += 1
                            if similar_count > self.config['ERROR_DETECTION']['max_consecutive_similar_frames']:
                                error_flags.add('consecutive_similar_frames')
                        else:
                            similar_count = 0
                        break
                    except ValueError as e:
                        if attempt < max_retries - 1:
                            win_size = max(3, win_size - 2)
                            continue
                        logging.error(f"SSIM error: {str(e)}")
                        
            prev_frame = frame
            
        return '|'.join(error_flags) if error_flags else 'False'

    def assess_quality(file_info):
        """Comprehensive quality estimation"""
        estimates = []
        
        # Bitrate check
        res = file_info.get('readable_res', 'default')
        try:
            if (file_info.get('video_bitrate (kBit/s)', 0) < 
                CONFIG['QUALITY_THRESHOLDS']['bitrate'].get(res, 1000)):
                estimates.append('low_bitrate')
        except TypeError:
            pass
        
        # Size/duration check
        try:
            mb_per_min = (file_info['file_size_mb (MB)'] / 
                        (file_info['duration_sec'] / 60))
            if mb_per_min < CONFIG['QUALITY_THRESHOLDS']['size_duration'].get(res, 0.1):
                estimates.append('small_size')
        except (KeyError, ZeroDivisionError):
            pass
        
        # Compression quality
        if file_info.get('compression_quality', 'N/A') == 'Poor':
            estimates.append('high_compression')
            
        return ', '.join(estimates) if estimates else 'acceptable'

def process_file(file_path):
    """Process file with proper defaults"""
    analyzer = VideoAnalyzer()
    result = {
        'file_path': file_path,
        'problem_flag': 'False',
        'error_flags': 'N/A',
        'quality_estimate': 'N/A',
        'vqa_score': 'N/A',
        'vqa_algorithm_used': 'False'
    }
    
    try:
        # Metadata extraction
        metadata = get_video_metadata(file_path)
        if not metadata:
            result['problem_flag'] = 'metadata_error'
            return result
            
        # Parse metadata
        file_info = parse_metadata(metadata, file_path)
        result.update(file_info)
        
        # Quality estimation
        result['quality_estimate'] = analyzer.assess_quality(file_info)
        
        # VQA processing
        if CONFIG['ENABLE_VQA']:
            # Get metadata quality first
            quality = file_info.get('compression_quality', 'unknown')
            
            # Apply VQA filter
            run_vqa = True
            if CONFIG['VQA_METADATA_FILTER'] == 'poor' and quality != 'poor':
                run_vqa = False
            elif CONFIG['VQA_METADATA_FILTER'] == 'good' and quality != 'good':
                run_vqa = False
                
            if run_vqa:
                frames = analyzer.sample_frames(file_path)
                if frames:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=CONFIG['MAX_WORKERS']
                    ) as executor:
                        futures = [executor.submit(analyzer.analyze_frame, f) 
                                 for f in frames]
                        analyzer.vqa_results = {i: f.result() for i, f in enumerate(futures)}
                    
                    result['error_flags'] = analyzer.detect_errors(frames)
                    result['vqa_score'] = np.mean(
                        [f['brisque'] for f in analyzer.vqa_results.values() if f]
                    )
            result['vqa_algorithm_used'] = 'True'  # If VQA was actually performed
        else:
            # Explicitly set VQA columns to N/A
            result.update({
                'vqa_score': 'N/A',
                'error_flags': 'N/A'
            })
            
    except Exception as e:
        result.update({
            'problem_flag': 'processing_error',
            'error_flags': str(e)[:100]  # Truncate long errors
        })
        logging.error(f"Processing failed {file_path}: {str(e)}")
    
    # Ensure all columns are present
    for col in CONFIG['OUTPUT_COLUMNS']:
        if col not in result:
            result[col] = 'N/A'
            
    return result

def format_duration(seconds: float) -> str:
    """Convert duration in seconds to HH:MM:SS format"""
    try:
        return f"{int(seconds//3600):02}:{int((seconds%3600)//60):02}:{int(seconds%60):02}"
    except Exception as e:
        logging.error(f"Duration error: {str(e)}")
        return "N/A"

def get_video_metadata(file_path: str) -> dict:
    """Extract metadata with proper encoding"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', file_path],
            capture_output=True, text=True, check=True,
            encoding='utf-8',
            errors='replace'  # Handle decoding errors gracefully
        )
        return json.loads(result.stdout)
    except UnicodeDecodeError:
        logging.error(f"Encoding error in {file_path}")
        return None

def calculate_conversion_potential(current_codec: str, target_codec: str) -> float:
    """Calculate size reduction potential"""
    try:
        current = CONFIG['CODEC_EFFICIENCY'].get(current_codec.lower(), 1.0)
        target = CONFIG['CODEC_EFFICIENCY'].get(target_codec.lower(), 1.0)
        return 1 - (target / current)
    except Exception as e:
        logging.error(f"Conversion calc error: {str(e)}")
        return 0.0

def get_quality_label(value: float, threshold_type: str) -> str:
    """Get human-readable quality label"""
    try:
        if not isinstance(value, (int, float)):
            return 'N/A'
            
        thresholds = CONFIG[
            'COMPRESSION_RATIO_THRESHOLDS' if threshold_type == 'compression' 
            else 'CONVERSION_THRESHOLDS'
        ]
        
        if threshold_type == 'compression':
            if value < thresholds['good']: return 'Good'
            if value < thresholds['poor']: return 'Questionable'
            return 'Poor'
            
        elif threshold_type == 'conversion':
            if value >= thresholds['High']: return 'High'
            if value >= thresholds['Medium']: return 'Medium'
            return 'Low'
            
    except Exception as e:
        logging.error(f"Label error: {str(e)}")
        return 'N/A'

def calculate_compression_ratio(file_info: dict) -> dict:
    """Calculate compression metrics with enhanced validation"""
    try:
        # Check for required config parameters
        if 'COMPRESSION_RATIO_THRESHOLDS' not in CONFIG:
            raise KeyError("COMPRESSION_RATIO_THRESHOLDS missing in config")
            
        thresholds = CONFIG['COMPRESSION_RATIO_THRESHOLDS']
        
        # Validate required fields
        required_keys = ['video_bitrate (kBit/s)', 'width (px)', 
                        'height (px)', 'duration_sec']
        for key in required_keys:
            if key not in file_info:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(file_info[key], (int, float)):
                raise ValueError(f"Invalid type for {key}")

        # Validate numeric values
        width = file_info['width (px)']
        height = file_info['height (px)']
        duration = file_info['duration_sec']
        br = file_info['video_bitrate (kBit/s)']
        
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions")
        if duration <= 0.1:  # Minimum 0.1 second duration
            raise ValueError("Invalid duration")
        if br < 10:  # Minimum 10 kbit/s
            raise ValueError("Unrealistic bitrate")

        # Calculate bits per pixel per second
        bpp = (br * 1000) / (width * height * duration)
        
        file_info['compression_ratio'] = round(bpp, 4)
        file_info['compression_quality'] = get_quality_label(bpp, 'compression')
        
    except Exception as e:
        logging.error(f"Compression ratio error: {str(e)}")
        file_info['compression_ratio'] = 'N/A'
        file_info['compression_quality'] = 'N/A'
    
    return file_info

def calculate_conversion_potentials(file_info: dict) -> dict:
    """Calculate conversion potentials with error handling"""
    try:
        current_codec = file_info.get('video_codec_technical', '').lower()
        
        for target in ['h265', 'av1']:
            if current_codec not in CONFIG['CODEC_EFFICIENCY']:
                file_info[f'conversion_potential_{target}'] = 'N/A'
                file_info[f'conversion_quality_{target}'] = 'N/A'
                continue
                
            potential = calculate_conversion_potential(current_codec, target)
            file_info[f'conversion_potential_{target}'] = round(potential, 2)
            file_info[f'conversion_quality_{target}'] = get_quality_label(potential, 'conversion')
            
    except Exception as e:
        logging.error(f"Conversion potential error: {str(e)}")
        file_info.update({
            'conversion_potential_h265': 'Error',
            'conversion_quality_h265': 'Error',
            'conversion_potential_av1': 'Error',
            'conversion_quality_av1': 'Error'
        })
    
    return file_info

def parse_metadata(metadata: dict, file_path: str) -> dict:
    """Parse metadata with enhanced validation"""
    file_info = {}
    try:
        # Basic file info
        stat = os.stat(file_path)
        file_info.update({
            'file_name': os.path.basename(file_path),
            'folder_path': os.path.dirname(file_path),
            'creation_date': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'container': os.path.splitext(file_path)[1][1:].upper(),
            'file_size (bytes)': stat.st_size,
            'file_size_mb (MB)': round(stat.st_size / (1024 ** 2), 2)
        })

        # Duration handling
        duration = float(metadata['format'].get('duration', 0))
        file_info.update({
            'duration_sec': round(duration, 2),
            'duration_hms': format_duration(duration)
        })

        # Video stream validation
        video_stream = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), {})
        tech_codec = video_stream.get('codec_name', 'N/A').lower()
        file_info.update({
            'video_codec_technical': tech_codec,
            'video_codec_common': CODEC_MAP.get(tech_codec, tech_codec).upper(),
            'width (px)': int(video_stream.get('width', 0)),
            'height (px)': int(video_stream.get('height', 0)),
            'video_bitrate (kBit/s)': int(metadata['format'].get('bit_rate', 0)) // 1000
        })

        # Quality calculations
        file_info = calculate_compression_ratio(file_info)
        file_info = calculate_conversion_potentials(file_info)

    except Exception as e:
        logging.error(f"Parsing error {file_path}: {str(e)}")
    
    return {k: v for k, v in file_info.items() if k in CONFIG['OUTPUT_COLUMNS']}

def main():
    """Main execution with parallel processing"""
    print("Advanced Video Quality Analyzer\n")
    
    try:

        required_config_keys = [
            'COMPRESSION_RATIO_THRESHOLDS',
            'CONVERSION_THRESHOLDS',
            'CODEC_EFFICIENCY',
            'OUTPUT_COLUMNS'
        ]

        for key in required_config_keys:
            if key not in CONFIG:
                raise SystemExit(f"Missing required config key: {key}")

        if 'VQA' not in CONFIG:
            CONFIG['VQA'] = {
                'ENABLED': False,
                'METADATA_FILTER': 'all'
            }
        
        target_folder = input("Enter folder path: ").strip()
        output_csv = os.path.join(os.getcwd(), 'video_analysis_report.csv')
        
        # Collect files
        video_files = [os.path.join(root, f) 
                      for root, _, files in os.walk(target_folder)
                      for f in files if f.lower().endswith(('.mp4', '.avi', '.mkv'))]
        
        # Process files in parallel
        results = []  # Initialize results list
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CONFIG['MAX_WORKERS']
        ) as executor:
            futures = {executor.submit(process_file, f): f for f in video_files}
            
            # Use tqdm for progress bar
            with tqdm(total=len(video_files), unit='file') as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Processing error: {str(e)}")
                        pbar.update(1)
            
        # Write results
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CONFIG['OUTPUT_COLUMNS'])
            writer.writeheader()
            
            for res in results:
                # Clean empty values
                cleaned = {k: v if v not in (None, '') else 'N/A' 
                        for k, v in res.items()}
                writer.writerow(cleaned)
            
        print(f"\nAnalysis complete. Results: {output_csv}")

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()