# Real-Time Object Detection with OpenCV

A robust real-time object detection system using OpenCV and MobileNetSSD deep neural network. This project detects and localizes objects in video streams from webcams, video files, or image sequences with high accuracy and low latency.

## Overview

This project leverages the MobileNetSSD (Single Shot MultiBox Detector) architecture combined with OpenCV's DNN module to achieve real-time object detection. The system:

- **Detects 21 object classes** including: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, and TV monitor
- **Processes video streams in real-time** from webcams or video files
- **Draws bounding boxes** around detected objects with class labels and confidence scores
- **Calculates FPS** for performance monitoring
- **Highly optimized** using MobileNet for efficient inference on CPU

## Technical Details

### Architecture
- **Model**: MobileNetSSD (pre-trained on COCO dataset)
- **Framework**: Caffe deep learning framework
- **Input**: 300x300 RGB images
- **Output**: Bounding boxes, class predictions, and confidence scores

### Key Techniques
- **Mean subtraction** for illumination normalization
- **Scaling** for input normalization
- **Blobification** via `cv2.dnn.blobFromImage()` for preprocessing
- **FPS counter** for real-time performance tracking

## Requirements

- **Python 3.7+**
- **OpenCV 3.3+** (with DNN module)
- Webcam or video file for input

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Isha062004/CV-Project.git
cd CV-Project
```

### Step 2: Install Dependencies

#### Windows
```bash
pip install opencv-python imutils numpy
```

#### macOS
```bash
brew install opencv
pip install opencv-python imutils numpy
```

#### Linux
```bash
sudo apt-get install python3-opencv
pip install imutils numpy
```

### Step 3: Verify Model Files

Ensure these files are present in the project directory:
- `MobileNetSSD_deploy.prototxt.txt` (architecture definition)
- `MobileNetSSD_deploy.caffemodel` (pre-trained weights)

## Usage

### Run with Webcam

```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

### Run with Video File

```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --input path/to/video.mp4
```

### Configure Detection Confidence

Adjust the confidence threshold (default: 0.2) to filter weak predictions:

```bash
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --confidence 0.5
```

### View Help

```bash
python real_time_object_detection.py --help
```

## Script Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--prototxt` | `-p` | str | Required | Path to Caffe deploy prototxt file |
| `--model` | `-m` | str | Required | Path to Caffe pre-trained model |
| `--confidence` | `-c` | float | 0.2 | Minimum probability to filter weak predictions |
| `--input` | `-i` | str | 0 (webcam) | Path to input video file or camera index |

## Controls

While the video stream is running:
- **Press 'q'** to quit the application
- **Press 's'** to save the current frame

## Project Structure

```
Real-Time-Object-Detection-With-OpenCV/
├── real_time_object_detection.py    # Main detection script
├── MobileNetSSD_deploy.prototxt.txt # Model architecture
├── MobileNetSSD_deploy.caffemodel   # Pre-trained weights (~26 MB)
├── README.md                         # This file
├── LICENSE                           # MIT License
└── real_time_output_gif/            # Sample output videos/gifs
```

## Detectable Classes

The model can detect objects from the following 20 classes:
```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, 
diningtable, dog, horse, motorbike, person, pottedplant, sheep, 
sofa, train, tvmonitor
```

## Performance

- **Speed**: ~50-100 FPS on modern CPU (Intel i5+/i7)
- **Memory**: ~100 MB
- **Accuracy**: ~70% mAP on COCO dataset

## Troubleshooting

### ModuleNotFoundError

Ensure all dependencies are installed:
```bash
pip install --upgrade opencv-python imutils numpy
```

### Camera Not Found

- Verify webcam is connected and not in use by another application
- Try specifying a different camera index: `--input 1` or `--input 2`

### Slow Performance

- Increase confidence threshold to reduce processing overhead
- Run on a machine with dedicated GPU support
- Process lower resolution input video

## Future Enhancements

- [ ] GPU acceleration with NVIDIA CUDA
- [ ] Additional model architectures (YOLOv4, Faster R-CNN)
- [ ] Multi-threaded processing
- [ ] Video export with detections
- [ ] REST API for remote inference
- [ ] Real-time performance metrics dashboard

## References

- [MobileNet-SSD GitHub](https://github.com/chuanqi305/MobileNet-SSD)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyImageSearch DNN Guide](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)
- [Imutils Library](https://github.com/jrosebr1/imutils)
- [Caffe Framework](https://caffe.berkeleyvision.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Developed as part of a Computer Vision project. Contributions and improvements are welcome!

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing issues on GitHub
3. Create a new issue with detailed description and error logs
