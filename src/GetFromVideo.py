import posenet
import argparse
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None,
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

hotpoints = ['leftWrist', 'rightWrist', 'leftShoulder',
             'rightShoulder', 'leftKnee', 'rightKnee']


def main():
    if args.file is None:
        raise OSError("No file provided")
    savePoses = np.array([])
    with tf.Session() as sess:
        start = time.time()
        frame_count = 0
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        cap = cv2.VideoCapture(args.file)
        while True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.stopthread()
                break
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            newPose = []
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    if posenet.PART_NAMES[ki] in hotpoints:
                        newPose.append([posenet.PART_NAMES[ki], c])
                if pi == 0:
                    if savePoses.size == 0:
                        savePoses = np.array([newPose])
                    else:
                        savePoses = np.append(
                            savePoses, np.array([newPose]), axis=0)

            np.save(args.file, savePoses)
            keypoint_coords *= output_scale

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                print('Frame executed: %d / %d ...' % (frame_count,
                                                       int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                print("Done.")
                break
            print('Frame executed: %d / %d ...' % (frame_count,
                                                   int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end='\r')


if __name__ == "__main__":
    main()
