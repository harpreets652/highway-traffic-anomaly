import cv2
import argparse
from utils import flow_util as fu

"""
Visualize optical flow
"""


def main(args):
    video = cv2.VideoCapture(args.input_video)

    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    resize = tuple(args.resize) if args.resize else None

    ret, frame_1 = video.read()
    frame_1 = pre_process(frame_1, resize)

    print(f"Frame Size: {frame_1.shape}")
    print(f"Frame rate: {frame_rate}")

    while video.isOpened():
        ret, frame_2 = video.read()

        if frame_2 is None:
            break

        frame_2 = pre_process(frame_2, resize)

        # compute dense flow
        flow_vectors = dense_flow(frame_1, frame_2)

        dense_image = fu.flow_2_rgb(flow_vectors)
        dense_vectors_image = fu.dense_vector(frame_1, flow_vectors, args.dense_vec_patch)

        cv2.imshow("dense", dense_image)
        cv2.imshow("dense_vec", dense_vectors_image)
        cv2.waitKey(0)

        frame_1 = frame_2

    video.release()
    cv2.destroyAllWindows()

    return


def dense_flow(frame_a, frame_b):
    frame_a_gray = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    frame_b_gray = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    # flow = (delta y, delta x)
    # num-levels: similar to SIFT features, 'zooming' into image to account for large displacement
    # win size: pre-smoothing (see above)
    # num_iters: book/paper suggests 6 is good
    # polyN, polySigma: see book
    # flags: gaussian smoothing makes flow worse for my data set
    flow = cv2.calcOpticalFlowFarneback(frame_a_gray, frame_b_gray, None, 0.5, 3, 9, 6, 7, 1.5, 0)

    return flow


def pre_process(frame, frame_resize):
    if frame_resize:
        frame = cv2.resize(frame, frame_resize, interpolation=cv2.INTER_AREA)

    return frame


COLOR_WHEEL = fu.make_color_wheel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True, help="video to visualize")
    parser.add_argument("--resize", type=int, nargs='+', default=None, help="resize each frame")
    parser.add_argument("--dense_vec_patch", type=int,
                        nargs='+', default=(16, 16),
                        help="average flow ing patch and draw arrow")

    input_arguments = parser.parse_args()

    main(input_arguments)
