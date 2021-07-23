import torch
import torch.nn as nn
import torch.onnx as onnx
import os

from checkpoint_mgr.checkpoint_mgr import CheckpointMgr


def load_model():
    from model_design.hopenet_mbv2 import HopenetMBV2
    model = HopenetMBV2(num_bins=180)
    save_dir = '/data/output/head_pose_estimate_hopenet_mbv2_biwi_v2'

    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model,
                                  warm_load=False,
                                  map_location='cpu')
    model.eval()
    return model




def output_to_onnx(model, model_onnx_path):
    print('n_params: {}'.format(len([p for p in model.parameters()])))
    n_params = sum(p.nelement() for p in model.parameters())
    print('n_vals: {}'.format(n_params))

    # batch_size = 1
    dummy_input = torch.randn(1, 3, 224, 224)
    output = onnx.export(model=model,
                         args=dummy_input,
                         f=model_onnx_path,
                         verbose=True,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes={
                             'input': {0: 'b'},
                             'output': {0: 'b'}
                         },
                         opset_version=11
                         )

def main():
    model = load_model()
    output_to_onnx(model=model,
                   model_onnx_path='head_pose_estimation_hopenet_biwi_mbv2_v2.onnx',
                   )

    '''
    python3 -m onnxsim retinaface_mb_s1024_downfpn.onnx retinaface_mb_s1024_downfpn_sim.onnx 
    scp retinaface_mb_s1024_downfpn_sim.onnx ubuntu@10.42.20.153:/data/workspace/opensources/tengine/Tengine-Convert-Tools/build/install/bin/

    convert_model_to_tm -f onnx -m ./models/retinaface_mb_s1024_downfpn_sim.onnx -o ./models/retinaface_mb_s1024_downfpn_sim.tmfile

    '''


if __name__ == '__main__':
    main()
