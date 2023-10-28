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
import torch
import io


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the result data file or directory')
    return parser.parse_args()


class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def main():
    args = parse_opt()
    with open(args.data_path, 'rb') as f:
        data_dict = CpuUnpickler(f).load()
        pred_dicts = CpuUnpickler(f).load()
        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
        )

        if not OPEN3D_FLAG:
            mlab.show(stop=True)


if __name__ == '__main__':
    main()
