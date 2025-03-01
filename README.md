# Video Metadata Analyzer with Quality Assessment

This repository contains a Python script that analyzes video files in a directory (and its subdirectories) to extract technical metadata, assess compression quality, and estimate conversion potentials (e.g., for H.265 and AV1). The results are exported to a CSV file. Optionally, it can also perform visual quality analysis (VQA) if enabled in the configuration.

## Features

- **Metadata Extraction:** Uses FFprobe to extract video metadata.
- **Quality Assessment:** Calculates video compression ratios and assigns quality labels.
- **Conversion Potential:** Estimates potential size reduction for conversion to modern codecs.
- **Parallel Processing:** Uses Python’s `ProcessPoolExecutor` and `tqdm` for progress tracking.
- **Robust Error Handling:** Includes comprehensive logging and exception management.
- **Configurable:** Easy to adjust thresholds, codec efficiencies, and processing settings via a configuration dictionary.

## Requirements

- **FFmpeg/FFprobe:** Ensure these tools are installed and added to your system PATH.
- **Python Version:** Python 3.6 or later (tested with Python 3.12).
- **Python Packages:** Listed in `requirements.txt`:
  - numpy
  - opencv-python
  - scikit-image
  - tqdm

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Chrisch3n/Video-Metadata-Analyzer.git
   cd Video-Metadata-Analyzer
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script using:

```bash
python video_analyzer.py
```

When prompted, enter the folder path containing your video files. The script will process the videos, display progress, and export the results to a CSV file (e.g., `video_analysis_report.csv` in your current working directory).

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests. Please ensure that any new code adheres to PEP 8 guidelines and includes proper documentation.

## License

This project is licensed under the [MIT License](LICENSE).
