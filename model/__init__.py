from torch.autograd.grad_mode import F
from torch.nn.modules.transformer import Transformer
from .hdmapnet import HDMapNet
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar


def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    elif method == 'HDMapNet_Transformer':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False, transformer=True)
        print("Model type: HDMapNet_Transformer")
    else:
        raise NotImplementedError

    return model
