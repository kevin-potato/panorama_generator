# Panorama Generator

Created by Zhongkai YANG

### Project Overview

Panorama Generator is a tool that utilises computer vision techniques to extract key frames from videos and generate panoramic images. I employ my own ways to handle image and video files, implementing techniques ranging from basic frame extraction to complex image stitching, utilising the OpenCV library.

### Features

- **Key Frame Extraction**: Automatically detect and extract key frames from video files.
- **Panoramic Image Generation**: Stitch extracted key frames into panoramic images using image stitching techniques.

### Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

Ensure your Python environment is set up with the following dependencies. You can install them using pip:

```
pip install opencv-python-headless numpy
```

## Usage

### Extracting Key Frames

Run the `key_frames_capture.py` script to extract key frames from the specified video file. Ensure the video file path is correctly set in the script.

```
python key_frames_capture.py
```

### Generating Panoramic Images

Run the `panorama_generation.py` script to generate panoramic images. This script will automatically read the key frames extracted by `key_frames_capture.py` and process them for stitching.

```
python panorama_generation.py
```

## Examples

The `keyframes_1`, `keyframes_2`, `keyframes_3`, `keyframes_4`  folders included in the project contains some sample keyframes captured from videos that can be used to test and demonstrate the project's functionality. The original videos and the generated panoramas are also stored in the repository.