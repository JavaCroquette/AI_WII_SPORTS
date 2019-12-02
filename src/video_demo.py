import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true', default='false')
parser.add_argument('--image_dir', type=str, default='./in')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

pathIn= './output/'
pathOut = 'video.avi'

def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        start = time.time()
        cap = cv2.VideoCapture('video.mp4')
        cap.set(cv2.CAP_PROP_FPS, 10)
        fps = int(cap.get(5))
        print(fps)
        fps = fps/1
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        frame_array = []
        counter = 0;
        while(cap.isOpened()):
            counter += 1
            ret, frame = cap.read()
            if counter%1 == 0:
                print(counter)
                if ret == True:
                    input_image, draw_image, output_scale = posenet.read_video(frame,scale_factor=args.scale_factor, output_stride=output_stride)

                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image}
                        )

                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=1,
                        min_pose_score=0)

                    keypoint_coords *= output_scale
                    draw_image = posenet.draw_skel_and_kp(
                        draw_image, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0, min_part_score=0)

                    frame_array.append(draw_image)
                    height, width, layers = draw_image.shape
                    size = (width,height)
                else:
                    break
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
if __name__ == "__main__":
    main()
