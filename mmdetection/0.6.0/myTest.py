import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/cascade_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'checkpoints/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth')

# test a single image
#img = mmcv.imread('demo/000000000019.jpg')
#result = inference_detector(model, img, cfg)
#print(result)
#show_result(img, result)

# test a list of images
imgs = ['demo/19.jpg','demo/20.jpg','demo/21.jpg','demo/22.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
	print(i, imgs[i])
	show_result(imgs[i], result)
