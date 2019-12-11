# Cascade R-CNN阅读理解

文章核心：针对detection的classification和location做了细致的观察和实验分析发现了问题，并提出了简单易行但十分有效的办法。

### 发现问题

基础是2级的目标检测（two-stage detector），eg：RCNN。这些网络都会有1个header结构（在RCNN中就是SVM分类器；在Faster-RCNN中就是RPN网络）来对每个proposal（候选区域）做两件事情：classification和boundingbox regression (bbox reg，边界框回归)。

在classification中，每个proposal都会根据固定的IOU被分为正样本和负样本（由于候选区域可能只包含物体的一部分，所以需要界定正负样本）。

在bbox reg中，对每个标记为正样本的bbox会向标定好的框（ground-truth）进行回归。

### 1. 当前的问题

通过IOU来界定样本是正or负，所以IOU对模型的训练和测试都有较大的影响。

左图，横轴是**proposal自身的IOU**（就是在classification中界定正负样本的），纵轴是经过**bbox reg得到的原proposal的IOU**，不同线条代表的是利用**不同的阈值训练出来的header**(detector，单一阈值检测器，就是在训练过程中也需要进行正负样本的界定，它也会有设定IOU的阈值，**为u**)。

整体看：回归得到的IOU普遍比输入的IOU要高（根据灰色的线界定，在灰线上的，就是输出 > 输入）

仔细看：界定正负样本的IOU值设定在0.55-0.6范围内，用阈值0.5训练出来的detector测试效果是最好的；

​				0.6-0.75范围内，用阈值0.6训练出来的detector测试效果最好（在proposal相同的IOU下看效果）；

​				0.75-，用阈值0.7训练出来的detector测试效果最好；

说明：

1. bbox reg有效果，能提高location的准确度
2. **proposal的阈值和训练器训练时设定的阈值相近的时候，效果最好**（阈值相距太远，就会存在mismatch问题）

所以能联想到，**单一阈值训练出来的检测器效果有限**。eg，在测试的时候，我们以IOU = 0.5（常见的界定正负样本的阈值）作为阈值，那么所有IOU > 0.5的proposal都会被选中，那些在0.5-0.6左右的proposal经过bbox reg得到的边界框效果很好；但是 > 0.6的那些proposal，经过detector得到的效果就很差。

但是why不直接用高阈值训练出来的detector来检测呢？

可以从右图看到，在相同的测试阈值的设定下，经过不同阈值训练出来的detector效果是不一样的。eg，当输入的proposal划定正负样本的阈值设定为0.5时，u=0.7的detector效果是最差的（AP值越小，误差就越大），原因是用高阈值训练时，由于阈值较高，正样本数量就会大大减少，**过拟合（overfitting）情况十分严重**。所以作者提出了一个巧妙但是也十分直观的方法。

<img src="markdown_pic\image-20191208104047021.png" alt="image-20191208104047021" style="zoom:80%;" >

### 2. 解决方法

如何能保证proposal的高质量又不减少训练样本？

用**一个stage的输出去训练下一个stage**。根据上左图，**某个proposal经过detector后输出的IOU都是比原来好的，所以它的输出再经过下一个更高阈值的detector后效果一定更好**。

eg：现有3个训练好的detector串联，它们分别是用阈值0.5/0.6/0.7训练得到的，现在丢入一个IOU=0.55的proposal，经过阈值为0.5的detector后，IOU变成0.7；再经过0.6的detector，IOU变成0.8；再经过0.7的detector，IOU变成0.89，可以看到经过一系列的detector，效果比单一的detector要好。

而且，在训练过程中，也用相同的策略，那么经过低阈值的训练并且bbox reg得到的样本的IOU也会提高，**样本质量不断的提高**，那么下一个detector的阈值即使高点，但是样本数量不会减少太多，从而也**避免了样本数目太少而引起的过拟合问题**。

所以，作者提出在**训练和测试的过程中，均有相同的操作**（而不是单独训练3个阈值逐步提高的detector）

经过一个卷积层提取图片整体的特征图，除了**第一个bbox ( B0 )是原始的神经网络（eg，fast-rcnn的RPN）提供的proposal**（候选区域）。之后的都是根据**前一低阈值回归得到的bbox作为输入——B1->pool、B2->pool**，每个**header（用来bbox reg）的阈值不一样——H1、H2、H3（逐渐提高）**，并且由于训练的样本池（pool）是前一回归的结果，所以使已经经过优化的，所以**pool内的样本质量优于之前的（即保证有足够多的正样本来进行训练）**。

<img src="markdown_pic\image-20191208104012089.png" alt="image-20191208104012089">

与其他已经存在的方法相比较，可以发现它的优势：

<img src = "markdown_pic\image-20191208133851433.png" alt="image-20191208133851433">

(a) 为faster-rcnn——二级训练的神经网络

(b) 迭代式的bbox回归：和cascade-rcnn很像，但是它的**每个header都是相同**的——根据1中存在的问题：**单一阈值的detector无法对所有IOU范围的proposal进行良好的回归**；即B1回归得到的样本的IOU一般都从0.5->0.75以上了，再用相同阈值的detector就没有很好的效果。

并且detector会改变样本的分布，如果依旧使用原来的阈值的detector效果不好，如下图第一行：横纵轴分别表示回归目标中的box的x方向和y方向的偏移量。可以发现，从1st->2nd，proposal的分布发生了很大的变化。样本逐渐靠近Ground Truth——质量变高了；离群点也变多了，需要提高IOU的阈值来剔除离群点。

<img src="markdown_pic\image-20191208143704295.png" alt="image-20191208143704295">

(c) 迭代损失：用相同样本根据不同阈值训练出不同的detector，然后根据输入的proposal分别丢入不同阈值的detector中，然后综合看三个结果进行输出。**在训练过程中，由于样本都是一样的，还是存在过拟合的问题。**

虽然阈值不断提高，但是对应的阈值内的样本没有变少。入下图可以看出。初始从RPN网络出来的proposal的IOU都较小——质量较差；经过第一次bbox reg之后，大部分的proposal的IOU均明显提高，所以高阈值下的样本数目没有变少；第二次也是类似。

<img src= "markdown_pic\image-20191208145531489.png" alt="image-20191208145531489">



### 3. cascade-rcnn的实现和结果

确定的是一个4级结构：1个RPN+3个检测器（阈值分别为0.5/0/6/0.7），实现思路类似于Faster-rcnn第二阶段检测器的第二阶段。

# CascadeRCNN的实现

## 					——基于mmdetection的实现

### 1. init()

是module的构造函数

init()函数将config配置文件中的字典映射成module，将数据进行保存到module的属性中。这些module类都是torch.nn.module的子类。

```python
# 根据如下的语句，将类——CascadeRCNN作为形参传入了register_module
@DETECTORS.register_module
# 参数来自cascade_rcnn_r50_fpn_1x.py
class CascadeRCNN(BaseDetector, RPNTestMixin):    
	# module的构造函数    
    #这些module类都是torch.nn.module的子类
	# num_stages = 3 backbone=ResNet neck=FPN rpn_head = RPNHead bbox_roi_extractor= 			SingleRoIExtractor bbox_head= SharedFCBBoxHead * 3   
    #其余的如果没有都默认赋值为None
	def __init__(self,                 
				num_stages,                 
				backbone,                 
				neck=None,                 
				rpn_head=None,                 
				bbox_roi_extractor=None,                 
				bbox_head=None,                 
				mask_roi_extractor=None,                 
				mask_head=None,                 
				train_cfg=None,                 
				test_cfg=None,                 
				pretrained=None):   
        #判断bbox_roi_extractor，bbox_head是否为None，这2个数据必须要传入
		assert bbox_roi_extractor is not None        
		assert bbox_head is not None  
        #继承之前的对象初始化方法
		super(CascadeRCNN, self).__init__()  
      	# 赋值级数 
       	self.num_stages = num_stages        
		# 创建backbone组件模型——ResNet实例，在register注册，具体位置在backbones/resnet.py中   
        #将传入的backbone的数据来初始化一个resnet类实例
		self.backbone = builder.build_backbone(backbone)        
		# 创建neck组件模型——FPN实例，在registers注册，具体位置是necks/fpn.py        		
        # 如果找不到就会报错说：未实现错误        
		if neck is not None:            
			self.neck = builder.build_neck(neck)        
		else:            
			raise NotImplementedError        
			# 创建build_head——RPNHead实例，在register注册，具体位置在anchor_heads/rpn_head 
			if rpn_head is not None:            
				self.rpn_head = builder.build_head(rpn_head)        
     		# 创建bbox_roi_extractor——SingleRoIExtractor实例，具体位置是roi_extractors/single_level        
       		# 创建bbox_heads组件——类型是SharedFCBBoxHead，具体位置是bbox_heads/convfc_bbox_head        
		if bbox_head is not None:            
			# bbox_roi_extractor= SingleRoIExtractor            
            self.bbox_roi_extractor = nn.ModuleList()            
          	self.bbox_head = nn.ModuleList()            
         	# 若bbox_roi_extractor不是list，意味着就一个网络，复制num_stages遍构成一个list   
          	if not isinstance(bbox_roi_extractor, list):                
            	bbox_roi_extractor = [                    
                	bbox_roi_extractor for _ in range(num_stages)                
             	]            
             # bbox_head是一个list，跳过，否则就复制num_stages次构建成list类型
          	if not isinstance(bbox_head, list):               
                bbox_head = [bbox_head for _ in range(num_stages)]   
         	# 判断三者是否是一致大小
           	assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages   
        	for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                # 创建3级的 bbox_head 和 bbox_roi_extractor，都是nn.ModuleList
              	self.bbox_roi_extractor.append(                    											builder.build_roi_extractor(roi_extractor))                							self.bbox_head.append(builder.build_head(head))        
         if mask_head is not None:            
        	self.mask_roi_extractor = nn.ModuleList()            
           	self.mask_head = nn.ModuleList()            
            if not isinstance(mask_roi_extractor, list):                								mask_roi_extractor = [                    
                	mask_roi_extractor for _ in range(num_stages) 
            	]            
        	if not isinstance(mask_head, list):                
            	mask_head = [mask_head for _ in range(num_stages)]            
         	assert len(mask_roi_extractor) == len(mask_head) == self.num_stages
            for roi_extractor, head in zip(mask_roi_extractor, mask_head):
                self.mask_roi_extractor.append(                    											builder.build_roi_extractor(roi_extractor))                							self.mask_head.append(builder.build_head(head))        
     	# 赋值train的配置/test的配置        
     	self.train_cfg = train_cfg        
     	self.test_cfg = test_cfg        
        # 初始化这些权值        
        self.init_weights(pretrained=pretrained)    
```

### 2. init_weight

```python
	@property    
	def with_rpn(self):        
        return hasattr(self, 'rpn_head') and self.rpn_head is not None    
    # 初始化权重    
    def init_weights(self, pretrained=None):        
        super(CascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)        
        if self.with_neck:            
            if isinstance(self.neck, nn.Sequential):                
                for m in self.neck:                    
                    m.init_weights()            
         	else:                
                self.neck.init_weights()        
   		if self.with_rpn:            
            self.rpn_head.init_weights()        
            for i in range(self.num_stages):            
                if self.with_bbox:                												self.bbox_roi_extractor[i].init_weights()                								self.bbox_head[i].init_weights()            
   		if self.with_mask:                
            self.mask_roi_extractor[i].init_weights()   
            self.mask_head[i].init_weights()    
```

### 3. extract_feat()

提取特征

```python
	#提取img特征    
	def extract_feat(self, img):        
        # 经过backbone的前向计算        
        x = self.backbone(img)        
        # 如果有neck的特征处理，就将提取的特征值传递到neck进行处理        
        if self.with_neck:            
            x = self.neck(x)        
     	return x    
```

### 4. forward_train()

前向传播训练，也就是实现了层之间的连接。



```python
	def forward_train(self,                     															  img,
                  	  img_meta,
                      gt_bboxes,
                  	  gt_labels,
                  	  gt_bboxes_ignore=None,
                  	  gt_masks=None,
                  	  proposals=None): 
    	#提取特征——通过backbone和neck网络
    	x = self.extract_feat(img)  
        #计算loss，包括rpn、bbox、mask
    	losses = dict()    
        #如果有rpn网络的，就需要执行rpn网络的训练
    	if self.with_rpn:            
            #x为提取出来的特征向量，将特征输入到rpn_head，提取出boundingbox
            rpn_outs = self.rpn_head(x) 
            #计算rpn的loss
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,                                          self.train_cfg.rpn)            
            rpn_losses = self.rpn_head.loss(                
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)            
            losses.update(rpn_losses)     
            #将RPN输出的bbox和相关参数输入到proposal
            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)            
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)        
      	else:            
            proposal_list = proposals 
       #三次循环     
     	for i in range(self.num_stages):  
            #当前循环次数
      		self.current_stage = i        
            #cascade-rcnn的rcnn每次迭代的参数都不一样，0.5-0.6-0.7
    		rcnn_train_cfg = self.train_cfg.rcnn[i]     
            #loss值每次循环权重不一样 stage_loss_weights=[1, 0.5, 0.25])
     		lw = self.train_cfg.stage_loss_weights[i]            
			
            # assign gts and sample proposals  
            #分正负样本分开采集
        	sampling_results = []            
        	if self.with_bbox or self.with_mask:                
            	bbox_assigner = build_assigner(rcnn_train_cfg.assigner)                
            	bbox_sampler = build_sampler(                    
                		rcnn_train_cfg.sampler, context=self)                
            	num_imgs = img.size(0)                
     			if gt_bboxes_ignore is None:                    
            		gt_bboxes_ignore = [None for _ in range(num_imgs)]                
            	#遍历
            	for j in range(num_imgs):    
                    #分离正负样本
                	assign_result = bbox_assigner.assign( 
                        proposal_list[j], gt_bboxes[j], 
                        gt_bboxes_ignore[j], gt_labels[j])  
                    #样本采样
                    sampling_result = bbox_sampler.sample(assign_result,
                                                          proposal_list[j],               
                    									  gt_bboxes[j],                        														gt_labels[j],                   
                    				feats=[lvl_feat[j][None] for lvl_feat in x])                    		#将采样结果
                    sampling_results.append(sampling_result) 
                    
            # roi pooling 池化过程        
        	# bbox head forward and loss            
         	bbox_roi_extractor = self.bbox_roi_extractor[i]            
            bbox_head = self.bbox_head[i]    
            
            rois = bbox2roi([res.bboxes for res in sampling_results])            					bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],                                            rois)            
            cls_score, bbox_pred = bbox_head(bbox_feats)            
            
            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,                                                gt_labels, rcnn_train_cfg)            
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)  
            
            #获得loss_bbox值
            for name, value in loss_bbox.items():                
                losses['s{}.{}'.format(i, name)] = (value * lw if                                                    'loss' in name else value)            
         	
            # mask head forward and loss            
         	if self.with_mask:                
            	mask_roi_extractor = self.mask_roi_extractor[i]                							mask_head = self.mask_head[i]                
              	pos_rois = bbox2roi(                    
                        [res.pos_bboxes for res in sampling_results])                					mask_feats = mask_roi_extractor(                    										x[:mask_roi_extractor.num_inputs], pos_rois)                						mask_pred = mask_head(mask_feats)                
                mask_targets = mask_head.get_target(sampling_results, gt_masks,                                                    rcnn_train_cfg)                
                pos_labels = torch.cat(                    
                        [res.pos_gt_labels for res in sampling_results])                				loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)                			  for name, value in loss_mask.items():                    
                    losses['s{}.{}'.format(i, name)] = (value * lw                                                        if 'loss' in name else value) 
                    
             # refine bboxes，再更新了proposal_list，此时的proposal和gt的iou更好了          
            if i < self.num_stages - 1:                
                pos_is_gts = [res.pos_is_gt for res in sampling_results]                				roi_labels = bbox_targets[0]   # bbox_targets is a tuple                
                with torch.no_grad():                    
                    proposal_list = bbox_head.refine_bboxes(                        
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)        
       return losses    
```
大体上思路：input -> backbone -> neck -> head -> cls and ref

forward()的整体实现过程：

- 将输入的图片**提取特征**：backbone ( ResNet ) + neck ( FPN )，是调用函数extract_feat()得到的
- 根据前一个输出的特征图，去**提取proposal**：rpn_head ( RPNHead )，用rpn_head(x)实现，在调用这个的时候还要用到anchor_head.py中的另一个函数get_bboxs()
- 根据输入的proposal，先**区分正负样本**，assigners完成样本正负判定；sampler对这些**样本采样**，得到sampling_result，这个是可以送入去进行训练的——**Cascade-RCNN就是这边开始有不同的，经过3次循环，每次循环的IOU阈值是逐渐提高的**
- 已经获得每个图片的采样之后的正负样本，进行一次RoI Pooling ( SingleRoIExtractor )，将不同大小的框映射成固定大小。
- 池化之后的结果送到**bbox head——classification+detection**，针对每个框进行classification和bbox的修正。之前rpn为单纯的二分类——前景、背景，这里分为N+1类(类别+背景)。调用的是bbox_head——并且**将优化后的bbox应用到proposal中，并且更新proposal中，则第二次循环就是用的优化过的proposal这是Cascade的另一个优势**

