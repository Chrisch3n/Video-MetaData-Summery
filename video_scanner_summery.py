"""
Video Metadata Analyzer with Quality Assessment and Progress Tracking
"""

import os
import subprocess
import json
import csv
import logging
from datetime import datetime
from time import time
from tqdm import tqdm

# Configuration - Adjust these values as needed
CONFIG = {
    'OUTPUT_COLUMNS': [
        'file_name', 'folder_path', 'creation_date', 'modified_date',
        'duration_sec', 'duration_hms', 'file_size (bytes)', 'file_size_mb (MB)',
        'container', 'video_codec_technical', 'video_codec_common',
        'width (px)', 'height (px)', 'video_bitrate (kBit/s)',
        'compression_ratio', 'compression_quality',
        'conversion_potential_h265', 'conversion_quality_h265',
        'conversion_potential_av1', 'conversion_quality_av1'
    ],
    
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
    }
}

CODEC_MAP = {
    'hevc': 'H.265', 'h265': 'H.265', 'avc': 'H.264',
    'h264': 'H.264', 'av1': 'AV1', 'vp9': 'VP9'
}

# Configure logging
logging.basicConfig(
    filename='video_analysis_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def format_duration(seconds: float) -> str:
    """Convert duration in seconds to HH:MM:SS format"""
    try:
        return f"{int(seconds//3600):02}:{int((seconds%3600)//60):02}:{int(seconds%60):02}"
    except Exception as e:
        logging.error(f"Duration error: {str(e)}")
        return "N/A"

def get_video_metadata(file_path: str) -> dict:
    """Extract video metadata using FFprobe"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', file_path],
            capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except Exception as e:
        logging.error(f"Metadata error {file_path}: {str(e)}")
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
    """Calculate compression metrics with error handling"""
    try:
        required_keys = ['video_bitrate (kBit/s)', 'width (px)', 
                        'height (px)', 'duration_sec']
        if not all(k in file_info and isinstance(file_info[k], (int, float)) 
                   for k in required_keys):
            raise ValueError("Missing required numeric fields")
            
        br = file_info['video_bitrate (kBit/s)'] * 1000  # Convert to bits
        width = file_info['width (px)']
        height = file_info['height (px)']
        duration = file_info['duration_sec']
        
        if width <= 0 or height <= 0 or duration <= 0:
            raise ValueError("Invalid dimension/duration values")
        
        bpp = br / (width * height * duration)
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
    """Main execution with robust error handling"""
    print("Video Metadata Analyzer\n")
    
    try:
        target_folder = input("Enter folder path: ").strip()
        output_csv = os.path.join(os.getcwd(), 'video_analysis_report.csv')
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')

        # Verify FFprobe
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)

        # Count files
        total_files = sum(
            len([f for f in files if f.lower().endswith(video_extensions)])
            for _, _, files in os.walk(target_folder)
        )

        # Process files
        with tqdm(total=total_files, unit='file', dynamic_ncols=True,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CONFIG['OUTPUT_COLUMNS'])
                writer.writeheader()

                for root, _, files in os.walk(target_folder):
                    for file in files:
                        if file.lower().endswith(video_extensions):
                            file_path = os.path.join(root, file)
                            try:
                                metadata = get_video_metadata(file_path)
                                if metadata:
                                    parsed = parse_metadata(metadata, file_path)
                                    writer.writerow(parsed)
                            except Exception as e:
                                logging.error(f"Processing error {file_path}: {str(e)}")
                            finally:
                                pbar.update(1)

        print(f"\nAnalysis complete. Report: {output_csv}")
        print(f"Error log: {os.path.abspath('video_analysis_errors.log')}")

    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        print("Critical error occurred. Check log file.")

if __name__ == "__main__":
    main()