import torch.nn as nn


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    # 读name属性
    def name(self):
        return self._name

    @property
    # 读_module_dict属性
    def module_dict(self):
        return self._module_dict

    # 注册一个module，输入是一个model类
    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError(
                'module must be a child of nn.Module, but got {}'.format(
                    module_class))
        module_name = module_class.__name__

        # 判断传入的module是否在_module_dict已经注册了，如果没有注册就去注册
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    # 针对models中的各个网络调用的 @DETECTORS.register_module
    #直接把定义好的类当作参数传入
    def register_module(self, cls):
        self._register_module(cls)
        return cls

# 类的实例化，Regisitry是一个类，传入一个字符串，该字符串为属性name的值，并且创建_module_dict这个字典来存放字典数据
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
HEADS = Registry('head')
DETECTORS = Registry('detector')
