#!/usr/bin/env python3
"""
Video Metadata Analyzer with Quality Assessment

This script analyzes video files in a directory (and subdirectories), extracting technical
metadata, quality estimations, and potential conversion benefits. The results are exported to CSV.

Features:
- Metadata extraction using FFprobe
- Compression quality analysis
- Conversion potential estimation (H.265/AV1)
- Visual quality assessment (VQA) with error detection (if enabled)
- Configurable output and analysis parameters
- Parallel processing with progress tracking
- Comprehensive error handling and logging

Note:
There is currently (02.2025) a bug in python (3.12) that causes an exception at shutdown
in "C:\Python312\Lib\concurrent\futures\process.py", line 310, in weakref_cb.
This is harmless and can safely be ignored and should be patched with a future python update. 

Usage Example:
    $ python video_analyzer.py
    Video Metadata Analyzer
    Enter folder path: /path/to/videos
    ...
    Analysis complete. Results saved to: /path/to/current/video_analysis_report.csv

Dependencies:
- FFmpeg/FFprobe (must be in system PATH)
- Python packages: tqdm, numpy, opencv-python, scikit-image
"""

import os
import subprocess
import json
import csv
import logging
import concurrent.futures
from datetime import datetime
from time import time, sleep
from typing import Any, Dict, List, Optional, Union

import numpy as np
import cv2
from skimage import metrics
from tqdm import tqdm

# Configuration dictionary - adjust these values to control script behavior
CONFIG: Dict[str, Any] = {
    'OUTPUT_COLUMNS': [
        'file_name', 'folder_path', 'file_path', 'error_flags', 'creation_date',
        'modified_date', 'duration_sec', 'duration_hms', 'file_size (bytes)',
        'file_size_mb (MB)', 'container', 'video_codec_technical',
        'video_codec_common', 'width (px)', 'height (px)', 'video_bitrate (kBit/s)',
        'compression_ratio', 'compression_quality', 'conversion_potential_hevc',
        'conversion_quality_hevc', 'conversion_potential_av1', 'conversion_quality_av1',
        'problem_flag', 'quality_estimate', 'vqa_score', 'vqa_algorithm_used'
    ],
    'QUALITY_THRESHOLDS': {
        'bitrate': {'1080p': 5000, '720p': 2500, '480p': 1000, 'default': 1000},
        'size_duration': {'1080p': 1.5, '720p': 0.8, '480p': 0.3, 'default': 0.1},
        'compression_ratio': {'good': 0.03, 'questionable': 0.07, 'poor': 0.1}
    },
    'CODEC_EFFICIENCY': {
        'h264': 1.0,
        'hevc': 0.65,
        'av1': 0.5,
        'vp9': 0.7,
        'mpeg4': 1.3
    },
    'CONVERSION_THRESHOLDS': {
        'High': 0.3,
        'Medium': 0.1,
        'Low': 0
    },
    'VQA': {
        'ENABLED': False,
        'SAMPLE_RATE': 0.01,
        'FRAME_SAMPLING': 'fixed',  # Options: 'fixed' or 'random'
        'METADATA_FILTER': 'poor',
        'ERROR_DETECTION': {
            'min_color_variation': 0.1,
            'max_consecutive_similar_frames': 10,
            'laplacian_noise_threshold': 15
        }
    },
    'PERFORMANCE': {
        'MAX_WORKERS': os.cpu_count() // 2 if os.cpu_count() else 1,
        'USE_GPU': False
    }
}

# Mapping between technical codec names and common display names
CODEC_MAP: Dict[str, str] = {
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
    """
    Handles visual quality analysis and error detection in video frames.
    """

    def __init__(self) -> None:
        """Initialize the analyzer with an empty state."""
        self.problem_flags: List[str] = []
        self.error_flags: List[str] = []

    def sample_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Sample frames from a video file based on configuration.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            List of sampled frames in BGR format. If no frames are available, returns an empty list.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logging.error(f"No frames found in video: {video_path}")
            cap.release()
            return []

        sample_count = max(1, int(total_frames * CONFIG['VQA']['SAMPLE_RATE']))

        # Generate frame indices based on sampling strategy
        if CONFIG['VQA']['FRAME_SAMPLING'] == 'fixed':
            indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        else:
            indices = np.random.choice(total_frames, sample_count, replace=False)

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def detect_errors(self, frames: List[np.ndarray]) -> str:
        """
        Detect errors and artifacts in sampled video frames.
        
        Args:
            frames: List of video frames in BGR format.
            
        Returns:
            A pipe-separated string of detected error flags, or 'False' if none are detected.
        """
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
            except Exception:
                logging.exception("Frame validation error")
                continue

            try:
                if np.std(frame) < CONFIG['VQA']['ERROR_DETECTION']['min_color_variation']:
                    error_flags.add('low_color_variation')
            except Exception:
                logging.exception("Color analysis error")

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
                except Exception:
                    logging.exception("SSIM calculation error")
            prev_frame = frame

        return '|'.join(error_flags) if error_flags else 'False'


def get_video_metadata(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a video file using FFprobe.
    
    Args:
        file_path: Path to the video file.
        
    Returns:
        A dictionary containing raw metadata, or None if extraction fails.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', file_path],
            capture_output=True, text=True, check=True,
            encoding='utf-8', errors='replace'
        )
        return json.loads(result.stdout)
    except Exception:
        logging.exception(f"Metadata extraction error for {file_path}")
        return None


def parse_metadata(metadata: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """
    Parse FFprobe metadata into a structured format.
    
    Args:
        metadata: Raw metadata dictionary from FFprobe.
        file_path: Source file path.
        
    Returns:
        A dictionary of parsed metadata values filtered to include only OUTPUT_COLUMNS.
    """
    file_info: Dict[str, Any] = {}
    try:
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

        duration = float(metadata.get('format', {}).get('duration', 0))
        file_info.update({
            'duration_sec': round(duration, 2),
            'duration_hms': f"{int(duration // 3600):02}:{int((duration % 3600) // 60):02}:{int(duration % 60):02}"
        })

        video_stream = next((s for s in metadata.get('streams', []) if s.get('codec_type') == 'video'), {})
        tech_codec = video_stream.get('codec_name', 'N/A').lower()
        file_info.update({
            'video_codec_technical': tech_codec,
            'video_codec_common': CODEC_MAP.get(tech_codec, tech_codec).upper(),
            'width (px)': int(video_stream.get('width', 0)),
            'height (px)': int(video_stream.get('height', 0)),
            'video_bitrate (kBit/s)': int(metadata.get('format', {}).get('bit_rate', 0)) // 1000
        })

        height = file_info.get('height (px)', 0)
        if height >= 2160:
            file_info['readable_res'] = '2160p'
        elif height >= 1080:
            file_info['readable_res'] = '1080p'
        elif height >= 720:
            file_info['readable_res'] = '720p'
        else:
            file_info['readable_res'] = 'SD'

        file_info = calculate_compression_ratio(file_info)
        file_info = calculate_conversion_potentials(file_info)

    except Exception:
        logging.exception(f"Parsing error for {file_path}")
    
    # Filter to include only output columns
    return {k: v for k, v in file_info.items() if k in CONFIG['OUTPUT_COLUMNS']}


def calculate_compression_ratio(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the compression ratio and quality based on bitrate, resolution, and duration.
    
    Args:
        file_info: Dictionary containing video metadata.
        
    Returns:
        Updated file_info with 'compression_ratio' and 'compression_quality'.
    """
    try:
        required_keys = ['video_bitrate (kBit/s)', 'width (px)', 'height (px)', 'duration_sec']
        if any(not isinstance(file_info.get(k), (int, float)) for k in required_keys):
            raise ValueError("Missing or invalid numeric metadata")
        
        br = file_info['video_bitrate (kBit/s)'] * 1000
        width = file_info['width (px)']
        height = file_info['height (px)']
        duration = file_info['duration_sec']
        
        if width <= 0 or height <= 0 or duration < 0.1:
            raise ValueError("Invalid video dimensions or duration")
        
        bpp = br / (width * height * duration)
        file_info['compression_ratio'] = round(bpp, 4)
        file_info['compression_quality'] = get_quality_label(bpp)
    except Exception:
        logging.exception("Compression calculation error")
        file_info.update({'compression_ratio': 'N/A', 'compression_quality': 'N/A'})
    return file_info


def get_quality_label(bpp: float) -> str:
    """
    Convert bits-per-pixel value to a human-readable quality label.
    
    Args:
        bpp: Bits per pixel per second.
        
    Returns:
        'Good', 'Questionable', 'Poor', or 'N/A' based on thresholds.
    """
    try:
        thresholds = CONFIG['QUALITY_THRESHOLDS']['compression_ratio']
        if bpp < thresholds['good']:
            return 'Good'
        if bpp < thresholds['poor']:
            return 'Questionable'
        return 'Poor'
    except Exception:
        return 'N/A'


def calculate_conversion_potential(current_codec: str, target_codec: str) -> float:
    """
    Calculate the conversion potential for size reduction when switching codecs.
    
    Args:
        current_codec: Source codec name.
        target_codec: Target codec name.
        
    Returns:
        A float indicating potential reduction (e.g., 0.3 means 30% reduction),
        or 0.0 if calculation fails.
    """
    try:
        current_eff = CONFIG['CODEC_EFFICIENCY'].get(current_codec.lower(), 1.0)
        target_eff = CONFIG['CODEC_EFFICIENCY'][target_codec.lower()]
        return 1 - (target_eff / current_eff)
    except KeyError:
        logging.error(f"Invalid target codec: {target_codec}")
        return 0.0
    except Exception:
        logging.exception("Conversion calculation error")
        return 0.0


def calculate_conversion_potentials(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate conversion potentials for codecs 'hevc' and 'av1'.
    
    Args:
        file_info: Dictionary containing video metadata.
        
    Returns:
        Updated file_info with conversion potential and quality labels.
    """
    try:
        current_codec = file_info.get('video_codec_technical', '').lower()
        for target in ['hevc', 'av1']:
            if current_codec not in CONFIG['CODEC_EFFICIENCY']:
                file_info[f'conversion_potential_{target}'] = 'N/A'
                file_info[f'conversion_quality_{target}'] = 'N/A'
                continue
            potential = calculate_conversion_potential(current_codec, target)
            file_info[f'conversion_potential_{target}'] = round(potential, 2)
            file_info[f'conversion_quality_{target}'] = get_conversion_quality_label(potential)
    except Exception:
        logging.exception("Conversion potential error")
        file_info.update({
            'conversion_potential_hevc': 'Error',
            'conversion_quality_hevc': 'Error',
            'conversion_potential_av1': 'Error',
            'conversion_quality_av1': 'Error'
        })
    return file_info


def get_conversion_quality_label(potential: float) -> str:
    """
    Convert conversion potential value into a quality label.
    
    Args:
        potential: Conversion potential value.
        
    Returns:
        'High', 'Medium', 'Low', or 'N/A'.
    """
    try:
        if potential >= CONFIG['CONVERSION_THRESHOLDS']['High']:
            return 'High'
        if potential >= CONFIG['CONVERSION_THRESHOLDS']['Medium']:
            return 'Medium'
        return 'Low'
    except Exception:
        return 'N/A'


def assess_quality(file_info: Dict[str, Any]) -> str:
    """
    Assess overall video quality based on bitrate, file size, and compression.
    
    Args:
        file_info: Dictionary containing video metadata.
        
    Returns:
        A comma-separated string of quality issues, or 'acceptable' if none.
    """
    estimates: List[str] = []
    try:
        res = file_info.get('readable_res', 'default')
        br_threshold = CONFIG['QUALITY_THRESHOLDS']['bitrate'].get(res, 1000)
        if file_info.get('video_bitrate (kBit/s)', 0) < br_threshold:
            estimates.append('low_bitrate')
        
        mb_per_min = file_info['file_size_mb (MB)'] / (file_info['duration_sec'] / 60)
        size_threshold = CONFIG['QUALITY_THRESHOLDS']['size_duration'].get(res, 0.1)
        if mb_per_min < size_threshold:
            estimates.append('small_size')
        
        if file_info.get('compression_quality') == 'Poor':
            estimates.append('high_compression')
    except Exception:
        logging.exception("Quality assessment error")
    return ', '.join(estimates) if estimates else 'acceptable'


def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a video file to extract metadata and assess quality.
    
    Pipeline includes:
    - Metadata extraction using FFprobe.
    - Parsing and computing quality metrics.
    - Optional visual quality analysis (VQA) if enabled.
    
    Args:
        file_path: Full path to the video file.
        
    Returns:
        A dictionary of results formatted according to CONFIG['OUTPUT_COLUMNS'].
    """
    analyzer = VideoAnalyzer()
    result: Dict[str, Any] = {
        'file_path': file_path,
        'problem_flag': 'False',
        'error_flags': 'N/A',
        'quality_estimate': 'N/A',
        'vqa_score': 'N/A',
        'vqa_algorithm_used': 'False'
    }
    
    try:
        metadata = get_video_metadata(file_path)
        if not metadata:
            result['problem_flag'] = 'metadata_error'
            return result
        
        file_info = parse_metadata(metadata, file_path)
        result.update(file_info)
        result['quality_estimate'] = assess_quality(file_info)
        
        if CONFIG['VQA']['ENABLED']:
            frames = analyzer.sample_frames(file_path)
            if frames:
                result['error_flags'] = analyzer.detect_errors(frames)
                result['vqa_algorithm_used'] = 'True'
    except Exception as e:
        logging.exception(f"Processing failed for {file_path}")
        result.update({
            'problem_flag': 'processing_error',
            'error_flags': str(e)[:100]
        })
    
    for col in CONFIG['OUTPUT_COLUMNS']:
        result.setdefault(col, 'N/A')
    return result


def main() -> None:
    """
    Main execution function for the video analysis pipeline.
    
    Handles:
    - User input for target directory.
    - File discovery and parallel processing.
    - Progress tracking and CSV export.
    
    Usage:
        Run the script and enter the target folder when prompted.
    """
    print("Video Metadata Analyzer\n")
    try:
        target_folder = input("Enter folder path: ").strip()
        if not target_folder or not os.path.isdir(target_folder):
            print("Invalid folder path provided.")
            return

        output_csv = os.path.join(os.getcwd(), 'video_analysis_report.csv')
        video_extensions = ('.mp4', '.avi', '.mkv')
        video_files: List[str] = [
            os.path.join(root, f)
            for root, _, files in os.walk(target_folder)
            for f in files if f.lower().endswith(video_extensions)
        ]

        if not video_files:
            print("No video files found in the specified directory.")
            return

        results: List[Dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CONFIG['PERFORMANCE']['MAX_WORKERS']
        ) as executor:
            futures = [executor.submit(process_file, f) for f in video_files]
            with tqdm(total=len(video_files), unit='file') as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception:
                        logging.exception("Error in processing a file.")
                    pbar.update(1)
            # Explicitly shutting down the executor
            executor.shutdown(wait=True)

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CONFIG['OUTPUT_COLUMNS'])
            writer.writeheader()
            for res in results:
                cleaned = {k: v if v not in (None, '') else 'N/A' for k, v in res.items()}
                writer.writerow(cleaned)

        print(f"\nAnalysis complete. Results saved to: {output_csv}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception:
        logging.exception("Fatal error in main execution.")
        print("Critical error occurred. Check the log file for details.")


if __name__ == "__main__":
    main()
