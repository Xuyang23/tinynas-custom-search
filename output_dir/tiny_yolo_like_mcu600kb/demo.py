# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import argparse
import torch
from cnnnet import CnnNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure_json', type=str)
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()
    return args

def get_backbone(filename,
                 pretrained=True,
                 network_id=0,
                 classification=False  #False for detetion
                 ):
    # load best structures
    with open(filename, 'r') as fin:
        output_structures = json.load(fin)

    network_arch = output_structures['space_arch']
    best_structures = output_structures['best_structures']

    # If task type is classification, param num_classes is required
    # 自动根据 stride 决定 stage_idx
    def get_stage_indices(structure):
        stage_idx = []
        for i, block in enumerate(structure):
            if block.get("s", 1) > 1:
                stage_idx.append(i)
        return stage_idx

    structure = best_structures[network_id]
    stage_idx = get_stage_indices(structure)
    out_indices = tuple(range(len(stage_idx)))  # 比如 (0, 1, 2)

    backbone = CnnNet(
            structure_info=best_structures[network_id],
            out_indices=out_indices,
            num_classes=1000,
            classification=classification)
    backbone.init_weights(pretrained)

    return backbone, network_arch


if __name__ == '__main__':
    # make input

    args = parse_args()

    x = torch.randn(1, 3, 224, 224)

    # instantiation
    backbone, network_arch = get_backbone(args.structure_json, 
                                            args.pretrained)

    print(backbone)
    # forward
    input_data = [x]
    backbone.eval()
    pred = backbone(*input_data)

    #print output
    for o in pred:
        print(o.size())
