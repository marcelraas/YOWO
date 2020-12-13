import argparse
from pathlib import Path

import pytest
import torch

from yowo.model import YOWO


@pytest.fixture
def cfg_path():
    return Path(__file__).parent.parent.parent / "yowo" / "cfg"


@pytest.fixture
def yowo_options(cfg_path):
    yowo_options = argparse.Namespace(
        n_classes=24,
        backbone_2d='darknet',
        backbone_3d='resnext101',
        darknet_config=cfg_path / "yolo.cfg",
        backbone_2d_weights='',
        backbone_3d_weights='',
        freeze_backbone_2d=False,
        freeze_backbone_3d=False,
        batch_size=1,
        use_middle_frame=True
    )
    return yowo_options


def test_model_forward_default(yowo_options):

    model = YOWO(yowo_options, key_frame_for_ol=-1)
    model.cpu()
    output = model.forward(torch.zeros(size=(1, 3, 9, 224, 224)).cpu())

    assert output.shape == torch.Size([1, 145, 7, 7])


def test_model_forward_center(yowo_options):
    
    model = YOWO(yowo_options, key_frame_for_ol=4)
    model.cpu()

    output = model.forward(torch.zeros(size=(1, 3, 9, 224, 224)).cpu())
    assert output.shape == torch.Size([1, 145, 7, 7])
