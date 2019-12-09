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

![image-20191208104047021](markdown_pic\image-20191208104047021.png)

### 2. 解决方法

如何能保证proposal的高质量又不减少训练样本？

用**一个stage的输出去训练下一个stage**。根据上左图，**某个proposal经过detector后输出的IOU都是比原来好的，所以它的输出再经过下一个更高阈值的detector后效果一定更好**。

eg：现有3个训练好的detector串联，它们分别是用阈值0.5/0.6/0.7训练得到的，现在丢入一个IOU=0.55的proposal，经过阈值为0.5的detector后，IOU变成0.7；再经过0.6的detector，IOU变成0.8；再经过0.7的detector，IOU变成0.89，可以看到经过一系列的detector，效果比单一的detector要好。

而且，在训练过程中，也用相同的策略，那么经过低阈值的训练并且bbox reg得到的样本的IOU也会提高，**样本质量不断的提高**，那么下一个detector的阈值即使高点，但是样本数量不会减少太多，从而也**避免了样本数目太少而引起的过拟合问题**。

所以，作者提出在**训练和测试的过程中，均有相同的操作**（而不是单独训练3个阈值逐步提高的detector）

经过一个卷积层提取图片整体的特征图，除了**第一个bbox ( B0 )是原始的神经网络（eg，fast-rcnn的RPN）提供的proposal**（候选区域）。之后的都是根据**前一低阈值回归得到的bbox作为输入——B1->pool、B2->pool**，每个**header（用来bbox reg）的阈值不一样——H1、H2、H3（逐渐提高）**，并且由于训练的样本池（pool）是前一回归的结果，所以使已经经过优化的，所以**pool内的样本质量优于之前的（即保证有足够多的正样本来进行训练）**。

![image-20191208104012089](markdown_pic\image-20191208104012089.png)

与其他已经存在的方法相比较，可以发现它的优势：

![image-20191208133851433](markdown_pic\image-20191208133851433.png)

(a) 为faster-rcnn——二级训练的神经网络

(b) 迭代式的bbox回归：和cascade-rcnn很像，但是它的**每个header都是相同**的——根据1中存在的问题：**单一阈值的detector无法对所有IOU范围的proposal进行良好的回归**；即B1回归得到的样本的IOU一般都从0.5->0.75以上了，再用相同阈值的detector就没有很好的效果。

并且detector会改变样本的分布，如果依旧使用原来的阈值的detector效果不好，如下图第一行：横纵轴分别表示回归目标中的box的x方向和y方向的偏移量。可以发现，从1st->2nd，proposal的分布发生了很大的变化。样本逐渐靠近Ground Truth——质量变高了；离群点也变多了，需要提高IOU的阈值来剔除离群点。

![image-20191208143704295](markdown_pic\image-20191208143704295.png)

(c) 迭代损失：用相同样本根据不同阈值训练出不同的detector，然后根据输入的proposal分别丢入不同阈值的detector中，然后综合看三个结果进行输出。**在训练过程中，由于样本都是一样的，还是存在过拟合的问题。**

虽然阈值不断提高，但是对应的阈值内的样本没有变少。入下图可以看出。初始从RPN网络出来的proposal的IOU都较小——质量较差；经过第一次bbox reg之后，大部分的proposal的IOU均明显提高，所以高阈值下的样本数目没有变少；第二次也是类似。

![image-20191208145531489](markdown_pic\image-20191208145531489.png)



### 3. cascade-rcnn的实现和结果

确定的是一个4级结构：1个RPN+3个检测器（阈值分别为0.5/0/6/0.7），实现思路类似于Faster-rcnn第二阶段检测器的第二阶段。



