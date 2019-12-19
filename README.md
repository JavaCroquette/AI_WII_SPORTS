# AI_WII_SPORTS

## Need Python 3+ and Tensorflow

## Packages to install

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

- Put `.mp4` file in the `video/` directory
- For each video file run in `src/`

  - Linux :
    ```
    python3 get_from_video.py --file [yourfilename]
    ```
  - Windows :
    ```
    python get_from_video.py --file [yourfilename]
    ```

- To run

```
cd src/
python3 run.py
```
