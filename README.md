# AI_WII_SPORTS

## Requirements

- Python 3+
- Tensorflow
- [CUDA](https://developer.nvidia.com/cuda-downloads) or equivalent is strongly recommended

## Installation

Clone repository an install following dependencies

On debian/ubuntu :

```
sudo apt-get install python3-tk
pip3 install Pillow cv2 argparse numpy
```

On Windows :

```
pip install Pillow cv2 argparse numpy
```

## Execution

- Put `.mp4` file in the `videos/` directory
- For each video file run

  - Linux
    ```
    cd src/
    python3 get_from_video.py --file [yourfilename]
    ```
  - Windows
    ```
    cd src/
    python get_from_video.py --file [yourfilename]
    ```

- To run
  - Linux
    ```
    python3 run.py
    ```
  - Windows
    ```
    python run.py
    ```
