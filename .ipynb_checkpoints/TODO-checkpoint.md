# 基本想法

114.212.22.207

114.212.21.221

# 任务

0. 方法本身超参数的调优。主要是Cp调整，使得前后两项的量级接近，其他超参数可以顺便当做消融实验来做
2. 和lamcts对比。结论是Lamcts不适合高维问题。lamcts效果好的原因是迭代的将搜索空间划分成了非常小的部分，在非常小的搜索空间采样，所以新样本质量高，所以性能很好。扩展到高维空间，1是空间指数增大，需要更多(指数多)次的划分才能取得效果，2是划分越多，意味着树越深，即需要训练的分类器越多，加上样本维度高，训练的时间代价巨大。

其他不太急的事情
1. 用平均排名的变化图展示算法性能。之前大组会上的hyperband，[hyperband](https://arxiv.org/pdf/2105.09821.pdf)。或者像edo cs中，用平均排名，然后画一个优势最大的图
4. bestk好像还是容易陷入局部最优，相比于lamcts达不到更高的值，所以应该还能改进

# 具体实验细节

claim：
1. 低维情况以及少量适合turbo的问题中，lvs-turbo能够达到至少和turbo一样甚至更高的性能，并且消耗的时间更短
2. 大部分中高维情况下，turbo不再适合(说明一下turbo的缺点)，结合bo时能够超过高维BO的sota？？？

第一组：BO，Dropout-BO，LA-MCTS-BO，LVS-BO，REMBO，说明我们无论是低维还是高维情况下算法的变量选择的有效性。
1. 低维：levy10_50，levy20_50
2. 高维：hartmann6_100/300/500, levy20_100, levy10_300
(低维和高维都要同时有levy和hartmann)并且保证和下面的一致性

最好的情况是
1. hartmann6_50, levy10_50, levy20_50
2. hartmann6_100/300/500, levy20_100, levy10_300

第二组：TuRBO，HeSBO，ALEBO，LVS-BO，LVS-TuRBO，CMA-ES，说明低维和高维情况下的性能可以达到最优。
1. 低维以及少量适合turbo的情况下问题为ackley20_100和ackley20_300，
2. 高维的情况下问题为hartmann6_300和hartmann6_500。待选rosenbrock10_300
确定的四个问题：levy低维，ackley20_300说明适合turbo的情况下表现更好，hartmann6_300、hartmann6_500。

真实问题：TuRBO，HeSBO，ALEBO，LVS-BO，LVS-TuRBO，CMA-ES。
先跑LA-MCTS-TuRBO、TuRBO、LVS-TuRBO和LVS-BO
1. nasbench
1. rover：lvs-turbo收敛效果差不多，时间更好
2. hopper作为低维的真实问题，lvs-turbo远远超过turbo
3. walker高维问题，lvs-bo表现大于等于turbo


实验超参数：
alebo会使用2倍的有效维度，rembo使用有效维度，hesbo使用2倍和有效维度

目前的函数：
ackley20_100 ackley20_300
hartmann6_50 hartmann6_100 hartmann6_300 hartmann6_500
levy10_50 levy10_100 levy10_300
lvs-bo，lvs-turbo，bo，dropout-bo，lamcts-bo，rembo
turbo，hesbo，alebo，



## 消融实验

1. split时的依据：mean、median、kmeans。目前：mean
2. 重要的值确定方式：bo、turbo。目前：bo
3. 不重要值确定方式：random，bestk。目前：bestk，也许还可以对比dropout的所以值都从同一个变量中选择的方法
4. uct计算时，max、mean。目前：max
5. 选择右侧的值的次数计算，是用单次选择的次数还是累计选择的次数

alebo为什么运行到400轮就停止了。。。

## 需要思考的

计算get_axis_score时使用feature采样的样本的最大值还是均值，目前是max

dynamic_treeify时是否需要建立一颗完整的树

backpropogate时，需要更新哪些东西，是否要向上传递所有的样本点

变量数量过小时，则把节点的估值变得特别低，不选择他，因为会引起gpr的bug，这个可不可以通过聚类选择代表的方式解决

会不会因为选择不重要变量后，每次重要变量都是选择的best，反而使得f比较高

为什么在优化有更多冗余变量的函数时表现反而更好

# 目前的结论

dropout中的维度d是一个很难调的超参数，不是在d=实际维度或略大于略小于实际维度时效果最好，而是在比较奇怪的情况下效果很好。这个可以作为额外发现，放在附录或正文中

Effectiveness

# 真实世界实验

人造函数，beanin，ackley，rosenbrock

alebo：
1. Nasbench101，D=36，限制训练时间小于30m。包含输入输出共7个节点的CNN网络，最多不能超过9条边。7个节点邻接矩阵共21个参数，除了输入输出层外，5个中间层各可以选择3种操作，所以共36个参数。优化时，对于每个层种类的3个参数，选择最大的，对于边连接方式参数，找最大数量非零(也许可以设一个阈值)的添加到图中，直到保证连通输入输入的子图(因为有可能存在不连通的点)的边数不超过9条边(可以通过捕捉不同的异常实现)。
2. Policy search for robot locomotion：六足，每足3电机控制问题，控制器共72个参数，HDBO表现效果整体不好，文中分析是因为函数存在突变点(某些区域经过小的调整后，机器人会直接摔倒)，如果我们尝试每次只优化一部分变量，会不会好一些(更稳定，不容易摔倒)。

vs：
1. rover：60d，vs在这个上面效果很好，不知道靠不靠谱
2. MOPTA08：124d，

turbo（实验采集的样本量都很大）：
robot pushing，16d，可以通过turbo中的处理将问题变为无噪声问题
rover：60d，优化2维空间中的30个点（一条轨迹），
lunar：12d

ebo：rover

hesbo中有一个神经网络参数优化的问题，100d

暂定：
1. nasbench101
2. 神经网络参数优化，在hesbo中只评估了100次
3. rover、robot pushing，但在turbo中，评估次数是2w和1w
4. lunar，不一定比得过
5. rl问题

应该选一个采样数量稍微多一点的