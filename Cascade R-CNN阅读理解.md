# Cascade R-CNN阅读理解

文章核心：针对detection的classification和location做了细致的观察和实验分析发现了问题，并提出了简单易行但十分有效的办法。

## 发现问题

基础是2级的目标检测（two-stage detector），eg：RCNN。这些网络都会有1个header结构（在RCNN中就是SVM分类器；在Faster-RCNN中就是RPN网络）来对每个proposal（候选区域）做两件事情：classification和boundingbox regression (bbox reg，边界框回归)。

在classification中，每个proposal都会根据固定的IOU被分为正样本和负样本（由于候选区域可能只包含物体的一部分，所以需要界定正负样本）。

在bbox reg中，对每个标记为正样本的bbox会向标定好的框（ground-truth）进行回归。

### 1. 问题1

在classification中，指定不同的IOU划分正负样本会导致bbox reg效果不一样。