try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import argparse
import pickle


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the result data file or directory')
    return parser.parse_args()


def main():
    args = parse_opt()
    with open(args.data_path, 'rb') as f:
        data_dict = pickle.load(f)
        pred_dicts = pickle.load(f)
        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)


if __name__ == '__main__':
    main()
