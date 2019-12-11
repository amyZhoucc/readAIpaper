import mmcv
from torch import nn

from .registry import BACKBONES, NECKS, ROI_EXTRACTORS, HEADS, DETECTORS


def _build_module(cfg, registry, default_args):
    # 判断模型参数数据是否为dict，并且是否存在type键，即 type='CascadeRCNN', 一定要存在网络类型的标志，以调用相应的神经网络。
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    # 复制model数据
    args = cfg.copy()
    # 获取神经网络类型 CascadeRCNN，并且从arg中移除该键值
    obj_type = args.pop('type')
    # 判断type是否是字符串类型
    if mmcv.is_str(obj_type):
        # 输入的神经网络不在注册的类型中
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    # 如果有train和test的配置数据
    if default_args is not None:
        for name, value in default_args.items():
            #将其放入args中
            args.setdefault(name, value)

    # **args是将字典得到的各个元素unpack分别与形参匹配送入函数中，针对CascadeRCNN就是传入到cascade_rcnn的CascadeRCNN类中
    # 注意不包含type，因为已经pop出去了
    # 每个都正好有对应——num_stages，pretrained，backbone，neck，rpn_head，bbox_roi_extractor，bbox_head，train_cfg，test_cfg，其余CascadeRCNN类中有的但是配置文件中没有的就默认为None
    return obj_type(**args)


def build(cfg, registry, default_args=None):
    # isinstance判断变量是否是已知类型eg list，说明有多个模型那就一个一个构建模型 即[{},{},{}]
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    # 否则就说明是dict类型，说明只有一个模型，直接调用
    else:
        return _build_module(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
