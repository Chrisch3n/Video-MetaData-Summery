"""
Video Metadata Analyzer with Quality Assessment and Progress Tracking
"""

import os
import subprocess
import json
import csv
import logging
import concurrent.futures
import numpy as np
from datetime import datetime
from time import time
from tqdm import tqdm
import cv2
from skimage import metrics

# Configuration - Adjust these values as needed
CONFIG = {
    'OUTPUT_COLUMNS': [
        'file_name', 'folder_path', 'file_path', 'error_flags', 'creation_date',
        'modified_date', 'duration_sec', 'duration_hms', 'file_size (bytes)',
        'file_size_mb (MB)', 'container', 'video_codec_technical',
        'video_codec_common', 'width (px)', 'height (px)', 'video_bitrate (kBit/s)',
        'compression_ratio', 'compression_quality', 'conversion_potential_h265',
        'conversion_quality_h265', 'conversion_potential_av1', 'conversion_quality_av1',
        'problem_flag', 'quality_estimate', 'vqa_score', 'vqa_algorithm_used'
    ],
    
    'QUALITY_THRESHOLDS': {
        'bitrate': {'1080p': 5000, '720p': 2500, '480p': 1000, 'default': 1000},
        'size_duration': {'1080p': 1.5, '720p': 0.8, '480p': 0.3, 'default': 0.1},
        'compression_ratio': {'good': 0.03, 'questionable': 0.07, 'poor': 0.1}
    },
    
    'CODEC_EFFICIENCY': {
        'h264': 1.0, 'hevc': 0.65, 'av1': 0.5, 'vp9': 0.7, 'mpeg4': 1.3
    },
    
    'VQA': {
        'ENABLED': False,  # Set to True to enable visual quality analysis
        'SAMPLE_RATE': 0.01,
        'FRAME_SAMPLING': 'fixed',
        'METADATA_FILTER': 'poor',
        'ERROR_DETECTION': {
            'min_color_variation': 0.1,
            'max_consecutive_similar_frames': 10,
            'laplacian_noise_threshold': 15
        }
    },
    
    'PERFORMANCE': {
        'MAX_WORKERS': os.cpu_count() // 2,
        'USE_GPU': False
    }
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
        
    def sample_frames(self, video_path):
        """Sample frames from video file"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = max(1, int(total_frames * CONFIG['VQA']['SAMPLE_RATE']))
        
        if CONFIG['VQA']['FRAME_SAMPLING'] == 'fixed':
            indices = np.linspace(0, total_frames-1, sample_count, dtype=int)
        else:
            indices = np.random.choice(total_frames, sample_count, replace=False)
            
        frames = []
        for idx in indices:
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
        
        for frame in frames:
            if frame is None or frame.size == 0:
                error_flags.add('invalid_frame')
                continue
                
            try:
                h, w = frame.shape[:2]
                if h < 7 or w < 7:
                    error_flags.add('small_frame_size')
                    continue
                    
                win_size = min(7, min(h, w))
                win_size = win_size if win_size % 2 else win_size - 1
            except Exception as e:
                logging.error(f"Frame validation error: {str(e)}")
                continue

            # Color variation check
            try:
                if np.std(frame) < CONFIG['VQA']['ERROR_DETECTION']['min_color_variation']:
                    error_flags.add('low_color_variation')
            except Exception as e:
                logging.error(f"Color analysis error: {str(e)}")

            # Similar frames check
            if prev_frame is not None:
                try:
                    ssim = metrics.structural_similarity(
                        prev_frame, frame,
                        win_size=win_size,
                        channel_axis=2 if frame.ndim == 3 else None
                    )
                    if ssim > 0.98:
                        similar_count += 1
                        if similar_count > CONFIG['VQA']['ERROR_DETECTION']['max_consecutive_similar_frames']:
                            error_flags.add('consecutive_similar_frames')
                    else:
                        similar_count = 0
                except Exception as e:
                    logging.error(f"SSIM calculation error: {str(e)}")
                    
            prev_frame = frame
            
        return '|'.join(error_flags) if error_flags else 'False'

def get_video_metadata(file_path: str) -> dict:
    """Robust metadata extraction with encoding handling"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', file_path],
            capture_output=True, text=True, check=True, 
            encoding='utf-8', errors='replace'
        )
        return json.loads(result.stdout)
    except Exception as e:
        logging.error(f"Metadata error {file_path}: {str(e)}")
        return None

def parse_metadata(metadata: dict, file_path: str) -> dict:
    """Parse metadata into structured format"""
    file_info = {}
    try:
        # Basic file info
        stat = os.stat(file_path)
        file_info = {
            'file_name': os.path.basename(file_path),
            'folder_path': os.path.dirname(file_path),
            'creation_date': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            'modified_date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'container': os.path.splitext(file_path)[1][1:].upper(),
            'file_size (bytes)': stat.st_size,
            'file_size_mb (MB)': round(stat.st_size / (1024 ** 2), 2)
        }

        # Duration handling
        duration = float(metadata['format'].get('duration', 0))
        file_info.update({
            'duration_sec': round(duration, 2),
            'duration_hms': f"{int(duration//3600):02}:{int((duration%3600)//60):02}:{int(duration%60):02}"
        })

        # Video stream
        video_stream = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), {})
        tech_codec = video_stream.get('codec_name', 'N/A').lower()
        file_info.update({
            'video_codec_technical': tech_codec,
            'video_codec_common': CODEC_MAP.get(tech_codec, tech_codec).upper(),
            'width (px)': int(video_stream.get('width', 0)),
            'height (px)': int(video_stream.get('height', 0)),
            'video_bitrate (kBit/s)': int(metadata['format'].get('bit_rate', 0)) // 1000
        })

        # Calculate compression metrics
        file_info = calculate_compression_ratio(file_info)
        file_info = calculate_conversion_potentials(file_info)

    except Exception as e:
        logging.error(f"Metadata parsing error {file_path}: {str(e)}")
    
    return file_info

def calculate_compression_ratio(file_info: dict) -> dict:
    """Calculate compression metrics with validation"""
    try:
        required = ['video_bitrate (kBit/s)', 'width (px)', 'height (px)', 'duration_sec']
        if any(not isinstance(file_info.get(k), (int, float)) for k in required):
            raise ValueError("Invalid metadata types")
            
        br = file_info['video_bitrate (kBit/s)'] * 1000
        area = file_info['width (px)'] * file_info['height (px)']
        duration = file_info['duration_sec']
        
        if area <= 0 or duration <= 0:
            raise ValueError("Invalid dimensions/duration")
            
        bpp = br / (area * duration)
        file_info['compression_ratio'] = round(bpp, 4)
        file_info['compression_quality'] = get_quality_label(bpp)
        
    except Exception as e:
        logging.error(f"Compression error: {str(e)}")
        file_info.update({'compression_ratio': 'N/A', 'compression_quality': 'N/A'})
    
    return file_info

def get_quality_label(bpp: float) -> str:
    """Get compression quality label"""
    try:
        if bpp < CONFIG['QUALITY_THRESHOLDS']['compression_ratio']['good']:
            return 'Good'
        if bpp < CONFIG['QUALITY_THRESHOLDS']['compression_ratio']['poor']:
            return 'Questionable'
        return 'Poor'
    except Exception:
        return 'N/A'

def calculate_conversion_potential(current_codec: str, target_codec: str) -> float:
    """Calculate size reduction potential"""
    try:
        current = CONFIG['CODEC_EFFICIENCY'].get(current_codec.lower(), 1.0)
        target = CONFIG['CODEC_EFFICIENCY'].get(target_codec.lower(), 1.0)
        return 1 - (target / current)
    except Exception as e:
        logging.error(f"Conversion calc error: {str(e)}")
        return 0.0

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
            file_info[f'conversion_quality_{target}'] = get_conversion_quality_label(potential)
            
    except Exception as e:
        logging.error(f"Conversion potential error: {str(e)}")
        file_info.update({
            'conversion_potential_h265': 'Error',
            'conversion_quality_h265': 'Error',
            'conversion_potential_av1': 'Error',
            'conversion_quality_av1': 'Error'
        })
    
    return file_info

def get_conversion_quality_label(potential: float) -> str:
    """Get conversion quality label"""
    try:
        if potential >= CONFIG['QUALITY_THRESHOLDS']['conversion']['High']:
            return 'High'
        if potential >= CONFIG['QUALITY_THRESHOLDS']['conversion']['Medium']:
            return 'Medium'
        return 'Low'
    except Exception:
        return 'N/A'

def assess_quality(file_info: dict) -> str:
    """Generate quality estimate based on metadata"""
    estimates = []
    try:
        # Bitrate check
        res = file_info.get('readable_res', 'default')
        br_threshold = CONFIG['QUALITY_THRESHOLDS']['bitrate'].get(res, 1000)
        if file_info.get('video_bitrate (kBit/s)', 0) < br_threshold:
            estimates.append('low_bitrate')
            
        # Size/duration check
        mb_per_min = (file_info['file_size_mb (MB)'] / 
                     (file_info['duration_sec'] / 60))
        size_threshold = CONFIG['QUALITY_THRESHOLDS']['size_duration'].get(res, 0.1)
        if mb_per_min < size_threshold:
            estimates.append('small_size')
            
        # Compression quality
        if file_info.get('compression_quality') == 'Poor':
            estimates.append('high_compression')
            
    except Exception as e:
        logging.error(f"Quality assessment error: {str(e)}")
    
    return ', '.join(estimates) if estimates else 'acceptable'

def process_file(file_path):
    """Process a single video file"""
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
        result['quality_estimate'] = assess_quality(file_info)
        
        # VQA processing
        if CONFIG['VQA']['ENABLED']:
            frames = analyzer.sample_frames(file_path)
            if frames:
                result['error_flags'] = analyzer.detect_errors(frames)
                result['vqa_algorithm_used'] = 'True'
                
    except Exception as e:
        result.update({
            'problem_flag': 'processing_error',
            'error_flags': str(e)[:100]
        })
        logging.error(f"Processing failed {file_path}: {str(e)}")
    
    # Ensure all columns are present
    for col in CONFIG['OUTPUT_COLUMNS']:
        result.setdefault(col, 'N/A')
        
    return result

def main():
    """Main execution with proper cleanup"""
    print("Video Metadata Analyzer\n")
    
    try:
        target_folder = input("Enter folder path: ").strip()
        output_csv = os.path.join(os.getcwd(), 'video_analysis_report.csv')
        
        # Collect files
        video_files = []
        for root, _, files in os.walk(target_folder):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mkv')):
                    video_files.append(os.path.join(root, f))
        
        # Process files with progress tracking
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CONFIG['PERFORMANCE']['MAX_WORKERS']
        ) as executor:
            futures = [executor.submit(process_file, f) for f in video_files]
            
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
                cleaned = {k: v if v not in (None, '') else 'N/A' for k, v in res.items()}
                writer.writerow(cleaned)
                
        print(f"\nAnalysis complete. Results: {output_csv}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()