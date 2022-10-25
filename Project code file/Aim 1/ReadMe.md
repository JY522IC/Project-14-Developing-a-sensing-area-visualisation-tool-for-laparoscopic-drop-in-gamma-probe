# Pose estimation algorithm

## How to run the program
- Open `main.py` on your text editor.
- Uncomment desire camera stream and marker detection.
- run `python main.py` on the command line.

## Implemented camera stream input
- Intel RealSense
- Webcam

## Implemented marker
- ArUco Marker

## Directory
```
.
├── markers                 # Marker Interfaces
│   ├── __init__.py
│   └── aruco.py            # Aruco marker interface
├── camera.py               # Camera interface
├── main.py                 # Main file to run the detection algorithm
└── ReadMe.md               # Folder information
```