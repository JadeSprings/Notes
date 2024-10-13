# Multi-instance Learning



## Introduction and Basic Concept

考虑这样一个实际情景：

> 在医学影像领域中的癌症检测任务中，病理切片的分辨率高达200000 x 100000。很明显，这样巨大的图像数据很难一次性在内存中进行分类。所以我们需要将一张完整的病理切片进行裁剪，然后分别进行图像分类。但是这样又会造成一个问题：我们只知道这张完整的病理切片是否包含癌症，但是我们并不知道癌症信息被裁剪到了哪个子图像中，所以无法使用全监督的方法进行学习。 
>
> 这样的问题，其实就是 Multiple Instance Learning 所要解决的问题。



***MIL的概念***：

> 假设训练数据集中的每个数据是一个包(Bag)，每个包都是一个实例(instance)的集合，每个包都有一个训练标记，而包中的实例是没有标记的；如果包中至少存在一个正标记的实例，则包被赋予正标记；而对于一个有负标记的包，其中所有的实例均为负标记。
> （这里说包中的实例没有标记，而后面又说包中至少存在一个正标记的实例时包为正标记包，是相对训练而言的，也就是说训练的时候是没有给实例标记的，只是给了包的标记，但是实例的标记是客观存在的，存在正负实例来判断正负类别）



***Problem formulation[^1]***:

There is a bag of instances, e.g., $X = {x_1,...,x_K}$, where the instances exhibit neither dependency nor ordering among each other. We assume that $K$ could vary for different bags. The bag's label is $Y$ and the instances' labels are respectively $y_1,...y_K$. Therefore, the MIL problem can be written as the following form:
$$
Y = \begin{cases}
0, & if \sum_{K}y_K = 0 \\
1, & otherwise
\end{cases}
$$

> These assumptions imply that a MIL model must be **permutation-invariant**(排列/置换不变的).

The formulation can be re-write in a compact form using the maximum operator:
$$
Y = \underset{K}{max} (y_K)
$$

> Learning a model that tries to optimize an objective based on the maximum over instance labels would be problematic at least for two reasons. 
>
> + First, all gradient-based learning methods would encounter issues with **vanishing gradients**（梯度消失）. 
> + Second, this formulation is suitable **only when an instance-level classifier is used**.



In the MIL setting the bag probability $θ(X)$ must be permutation-invariant since we assume neither ordering nor dependency of instances within a bag. Therefore, the MIL problem can be considered in terms of  a specific form of the **Fundamental Theorem of Symmetric Functions(对称函数基本定理)** with monomials(单项式) given by the following theorem。



**Theorem 1.**  A scoring function for a set of instances $X$, $S(X) ∈ R$, is a symmetric function (i.e., permutation-invariant to the elements in $X$), if and only if it can be decomposed in the following form: 
$$
S(X) = g( \underset{x∈X}{\sum}f(x))
$$
where $f$ and $g$ are  suitable transformations

> This theorem provides a general strategy for modeling the bag probability using the decomposition given. A similar decomposition with max instead of sum is given by the following theorem.

**Theorem 2.**  For any $ε > 0$, a Hausdorff continuous symmetric function $S(X) ∈ R$ can be arbitrarily approximated by a function in the form $g( max_{x∈X} f (x))$, where max is the element-wise vector maximum operator and $f$ and $g$ are continuous functions, that is:
$$
|S(X) - g(\underset{x∈X}{max}f(x))|< ε
$$
The two theorems both formulate a general three-step approach for classifying a bag of instances:
(i) a transformation of instances using the function $f$
(ii) a combination of transformed instances using a symmetric (permutation-invariant) function $σ$ **(这个函数或这一步骤也被称为MIL池化)**
(iii) a transformation of combined instances transformed by $f$ using a function $g$

> The choice of functions $f , g$ and $σ$ determines a specific approach to modeling the label probability.

For a given MIL operator there are two main MIL approaches:
(i) **The instance-level approach**: The transformation $f$ is an instance-level classifier that returns scores for each instance. Then individual scores are aggregated by MIL pooling to obtain $θ(X)$. The function $g$ is the identity function.
(ii) **The embedding-level approach**: The function $f$ maps instances to a low-dimensional embedding. MIL pooling is used to obtain a bag representation that is independent of the number of instances in the bag. The bag representation is further processed by a bag-level classifier to provide $θ(X)$.

> It is advocated that the latter approach is preferable in terms of the bag level classification performance. Since the individual labels are unknown, there is a threat that the instance-level classifier might be trained insufficiently and it introduces additional error to the final prediction. The embedding-level approach determines a joint representation of a bag and therefore it does not introduce additional bias to the bag-level classifier.



***The main goal of MIL is[^1]***:

+ to learn a model that predicts a bag label, e.g., a medical diagnosis
+ to discover key instances, i.e., the instances that trigger the bag label

> In the medical domain the latter task is of great interest because of legal issues and its usefulness in clinical practice[^1]



***Methods that has been proposed[^1]***:

+ utilizing similarities among bags
+ embedding instances to a compact low-dimensional representation that is further fed to a bag-level classifier
+ combining responses of an instance-level classifier

> Drawbacks:
>
> + Only the last approach is capable of providing interpretable results.
>
> + the instance level accuracy of such methods is low and in general there is a disagreement among MIL methods at the instance level . These issues call into question the usability of current MIL models for interpreting the final decision.



*Some common concepts*:

> ROI: Region Of Interest
>
> weakly annotated data: 弱标签/注释数据是指，对单张图片来说，只标记这张图片里有哪些物体，而不标记这些物体对应的bounding box或者是像素级的标记







## Attention-based MIL



### Attention-based Deep Multiple Instance Learning[^1]-Maximilian Ilse/Jakub M. Tomczak/Max Welling

Contribution:



***

针对尝试使用max函数来优化目标的困难，作者提出：

In order to make the learning problem easier, we propose to train a MIL model by optimizing the **log-likelihood function** where the bag label is distributed according to the Bernoulli distribution with the parameter $θ(X) ∈ [0, 1]$, i.e., the probability of $Y = 1$ given the bag of instances $X$.



***MIL with Neural Networks[^1]***：

定理1和2表明，对于足够灵活的函数类，我们可以对任何具有排列不变性质的分数函数进行建模。

因此，我们可以考虑这样一类由神经网络$f_ψ(·)$参数化的变换，其参数$ψ$将第$k$个instance转化为low-dimensional emmbedding:
$$
h_k = f_ψ(x_k)
$$
其中，$h_k∈H$。对于instance-level approach, 有$H = [0,1]$；对于embedding-level approach，有$H = R^M$。

最终，参数$\theta(X)$(即得分函数)由以下变换决定：
$$
g_φ:H^k→[0,1]
$$
对于instance-level approach，$g_φ$是简单的恒等式；对于embedding-level approach，$g_φ$也可以由带有参数$φ$的神经网络进行参数化。

使用神经网络参数化所有变换的思想是非常吸引人的，因为整个方法可以任意地灵活，并且可以通过反向传播进行端到端的训练。唯一的限制是 MIL 池化（即函数$\sigma$）必须是可微分的。



***MIL Pooling[^1]***：

理论上，任何具有排列不变性质的池化算子都可以替换Theorem2中的max算子，且证明方法是相似的（证明参考引文）。当算子可微时，它也可以被用作由深度神经网络参数化变换的构架中的。



***Attention-based MIL Pooling[^1]***：

预先定义好的和不可训练的算子（我的理解是不能进行参数化的算子）不能通过调整任务和数据来获得更好的分类结果，而灵活且适应性强的池化算子可以做到这一点。



[^1]: [Attention-based Deep Multiple Instance Learning-All Databases (clarivate.cn)](https://webofscience.clarivate.cn/wos/alldb/full-record/WOS:000683379202024)
