The architecture of neural network:

+ Feedforward Neural Network
+ Recurrent Neural Network

***

NN可以进行学习的关键是：通过微调weights和bias可以对output进行微调，这意味着我们可以根据标签与预测值之间的差异反过来对weights和bias进行微调
$$
\Delta output= \frac{∂output}{∂w}\Delta w + \frac{∂output}{∂b}\Delta b \\
= \underset{j}{\sum} \frac{∂output}{∂w_j}\Delta w_j + \underset{j}{\sum} \frac{∂output}{∂b_j}\Delta b_j
$$

***



Dot product:
$$
y=wx+b = \underset{j}{\sum}w_jx_j+b
$$


Perceptron:
$$
y = \begin{cases}
1, & wx+b>0 \\
0, & wx+b<0
\end{cases}
$$
Sigmoid function:
$$
\sigma (z) = \frac{1}{1+e^{-z}}=\frac{1}{1+e^{-wx-b}}
$$

> 感知机和Sigmoid Neuron的区别就在于感知机的output是锐截止的，而Sigmoid Neuron的output是平滑过渡的。实际上，我们并不关心Sigmoid的具体代数表达式（我们关心的是其平滑过渡的特性），其他平滑过渡的Activation function也是可以使用的。Sigmoid被广泛应用的原因是其求导很方便
>
> -为什么Sigmoid平滑过渡的特性是重要的？
> 我们的目的是要让网络能够通过对weight和bias微调来微调output。对于锐截止的perceptron，在分界点的微小改变可能会使output变化很大，不能满足我们的目的。



***



## Cost Function and Gradient Descent

> 当我们能够通过微调weight和bias来微调output时，如何让网络自己通过标签与output间的差异自动调整weight和bias（即学习）是另一个需要解决的问题。要解决这个问题，就要用到cost function。Cost function用来表征标签与预测值之间的差异。

Cost function, also called Loss function or Objective function:
$$
C(w,b) ≡ \frac{1}{2n}\underset{x}{\sum}||y(x)-a||^2
$$
C is also called *quadratic cost function*; it's also sometimes known as *the mean squared error* or just *MSE*.



To minimize the cost function $C(w,b)$, we'll use an algorithm known as ***gradient descent***(梯度下降).



> 现在我们能够量化标签与预测值之间的差异了，但我们还要能够最小化这个差异，只有这样我们才能不断调整weight和bias使网络的预测值最接近真实值。

下面我们来讲讲梯度下降：

**Note $C(w,b)$ as $C(v)$ to simplify the question: How to minimize the $C(v)$**

> where $v$ consists of $v_1,v_2,...,v_n$

We have some definitions as follows:
$$
\Delta C ≡ \frac{∂C}{∂v_1} \Delta v_1 + ... + \frac{∂C}{∂v_n} \Delta v_n \\
\Delta v = (\Delta v_1,\Delta v_2,...,\Delta v_n)^T \\
∇C = (\frac{∂C}{∂v_1},\frac{∂C}{∂v_2},...,\frac{∂C}{∂v_n})^T[i.e.Gradient]
$$
With the definitions, the expressionn for $\Delta C$ can be rewritten as
$$
\Delta C ≡ ∇C · \Delta v
$$
这个公式的神奇之处在于：为我们提供了一种通过使C(cost function)不断变小来调整权重v(即,w和b)的策略：

Let **$\Delta v = - \eta ∇C $**, then we get:
$$
\Delta C ≡ ∇C · \Delta v = \Delta C ≡ -\eta ||∇C||^2 < 0 
$$
Because $\Delta C$ is always negative, the cost function $C$ decreases continuously until it reaches to the global minimum.

We can also express the "new $v$" as:
$$
v' = v+\Delta v = v-\eta ∇C
$$
 **where $\eta$ is a small, positive parameter, also known as the *Learning Rate*.**

<img src="C:\My\0ScientificReasearch\Notes\AI\NN\img\example-grad_des.png" alt="example-grandient_descent" style="zoom:70%;" />



我们该如何把梯度下降算法应用到神经网络中呢？下面我们用$w$和$b$来替换$v$.
$$
w' = w+\Delta w = w-\eta \frac{∂C}{∂w} \\
b' = b+\Delta b = b-\eta \frac{∂C}{∂b}
$$
在应用这个公式时，我们不妨这样考虑：

> Consider $C(w,b) = \frac{1}{2n} \underset{x}{\sum} ||y(x)-a||^2 = \frac{1}{n} \underset{x}{\sum} \frac{||y(x)-a||^2}{2} =  \frac{1}{n} \underset{x}{\sum} C_x$
>
> where $C_x = \frac{||y(x)-a||^2}{2}$. 
>
> To compute the $∇C$ , we need to compute all gradient $∇C_x$, then $∇C = \frac{1}{n} \underset{x}{\sum} ∇ C_x$. However, the procedure takes a long time, and learning thus occurs slowly.

To attack the problem above, an idea called ***Stochastic gradient descent*** was proposed to speed up learning.（随机梯度下降）

The main idea of stochastic gradient descent is *using the average gradient of a small sample that randomly chosen from the training inputs to approximate the average gradient of all training dataset*, i.e.:
$$
\frac{\underset{m}{\sum}∇C_x}{m} ≈ \frac{\underset{x}{\sum}∇C_x}{n} = ∇C
$$
where m is the subset of the n.

Thus we can get that:
$$
w' = w+\Delta w = w-\eta \frac{∂C}{∂w} ≈ w - \frac{\eta}{m} \underset{m}{\sum} C_x =  w - \frac{\eta}{m} \underset{m}{\sum} \frac{∂C_x}{∂w}\\
b' = b+\Delta b = b-\eta \frac{∂C}{∂b} ≈  b - \frac{\eta}{m} \underset{m}{\sum} C_x =  b - \frac{\eta}{m} \underset{m}{\sum} \frac{∂C_x}{∂b}
$$
where the sums $m$ are over all the training examples $C_x$ in the current mini-batch. Then we pick out another randomly chosen mini-batch and train with those. And so on, until we've exhausted the training inputs, which is said to complete an *epoch* of training. At that point we start over with a new training epoch.

> 有一点要注意：人们对cost function的定义并不总是相同的。例如在训练集样本数未知的情况下，我们的cost function仅仅是将模值相关的项累加（即没有$\frac{1}{n}$）。相关的计算也要对应修改。







***



## Backpropagation

**Backpropagation**: is about understanding how changing weights and biases changes the cost function in the nn

Some introduction to the notation:

> We'll use $w^{l}_{jk}$ to denote the weight for the connection from the $k^{th}$ neuron in the $(l-1)^{th}$ layer to the $j^{th}$ neuron in the $l^{th}$ layer.
>
> Similarily, we use $b_j^l$ for the bias of the $j^{th}$ neuron in the $l^{th}$ layer.
>
> And we use $a^l_j$ for the activation of the $j^{th}$ neuron in the $l^{th}$ layer.
>
> With the notation, we get the equation(element-wise):
> $$
> a^l_j = \sigma(\underset{k}{\sum}w^l_{jk}a^{l-1}_k + b^l_j)
> $$

![notation in Backpropagation section](C:\My\0ScientificReasearch\Notes\AI\NN\img\backpropa_notation.png)

![](C:\My\0ScientificReasearch\Notes\AI\NN\img\backpropa_notation_2.png)

To vectorize the equation, we rewrite the equation as:
$$
a^l = \sigma (w^l a^{l-1}+ b^l)
$$

> The vector $w^l$ is a 2D array. The row corresponds to the weights from the $(l-1)^{th}$ layer to the $l^{th}$ layer. The column corresponds to the nuerons in the $(l-1)^{th}$ layer. 
>
> Therefore, the size of the $w^l$ equals to (the neuron numbers of the $l^{th}$ layer) × (the neuron numbers of the $(l-1)^{th}$ layer). That's why we use the notation we defined as above.



Before we use backpropagation to compute the partial derivatives, we have 2 assumptions:

> **Assumption 1**
>
> The cost function can be rewrite as:
> $$
> C(w,x) = \frac{1}{n}\underset{x}{\sum}C_x
> $$
> **Assumption 2**
>
> The output can be rewrite as the function of the output layer activations.



**The four fundamental equations in backpropagation**

> $\delta^l_j$: the *error* of the $j^{th}$ neuron in the $l^{th}$ layer



//待补充

























****



## Improving the Way Neuron Networks Learn

**The cross-entropy cost function**

 our neuron learns by changing the weight and bias at a rate determined by the partial derivatives of the cost function, $∂C/∂w$ and $∂C/∂b$.



We define the cross-entropy cost function by:
$$
C=−\frac{1}{n}\underset{x}{∑}[ylna+(1−y)ln(1−a)]
$$

> Two properties in particular make it reasonable to interpret the cross-entropy as a cost function.
>
> + it's non-negative, that is, $C>0$
> + if the neuron's actual output is close to the desired output for all training inputs, $x$, then the cross-entropy will be close to zero

This expression fixes the learning slowdown problem. We substitute $a=σ(z)$ and apply the chain rule twice, obtaining:
$$
\frac{∂C}{∂w_j}=−\frac{1}{n}\underset{x}{∑}(\frac{y}{σ(z)}−\frac{(1−y)}{1−σ(z)})\frac{∂σ}{∂w_j} \\
= −\frac{1}{n}\underset{x}{∑}(\frac{y}{σ(z)}−\frac{(1−y)}{1−σ(z)})σ′(z)x_j.
$$
Putting everything over a common denominator and simplifying this becomes:
$$
\frac{∂C}{∂w_j}=\frac{1}{n}\underset{x}{∑}\frac{σ′(z)x_j}{σ(z)(1−σ(z))}(σ(z)−y) \\
 = \frac{1}{n}\underset{x}{∑}x_j(σ(z)−y)
$$

> Tip: $σ′(z)=σ(z)(1−σ(z))$

This is a beautiful expression. It tells us that the rate at which the weight learns is controlled by $σ(z)−y$, i.e., by the error in the output. The larger the error, the faster the neuron will learn. This is just what we'd intuitively expect.

In a similar way, we can compute the partial derivative for the bias:
$$
\frac{∂C}{∂b}=\frac{1}{n}\underset{x}{∑}(σ(z)−y).
$$


We've been studying the cross-entropy for a single neuron. In particular, suppose $y=y1,y2,…$ are the desired values at the output neurons, i.e., the neurons in the final layer, while $a^L_1,a^L_2,…$ are the actual output values. Then we define the cross-entropy by:
$$
C=−\frac{1}{n}\underset{x}{∑}\underset{j}{∑}[y_jlna^L_j+(1−y_j)ln(1−a^L_j)]
$$


**Softmax**

$$
a^L_j = \frac{e^{z^L_j}}{\underset{k}{\sum}e^{z^L_k}}
$$

> Softmax的输出总和是1，所以可以把它的输出看作是一种概率分布







Sigmoid output layer with cross-entropy cost function

Softmax output layer with log-likelihood cost function







**Overfitting and Regularization**

The phenomenon that the model performs well on the train dataset but badly on the test dataset is called ***overfitting***.



The sign of overfitting is that the accuracy on the test data and training data both stop imporiving at the same time, and that's the time we stop training the network.

> 过拟合发生在模型对训练数据学习得太好，以至于它捕获了数据中的噪声和异常值，而不是潜在的模式。这会导致模型在新的、未见过的数据上表现不佳。



One strategy to avoid overfitting is to use ***validation dataset***. If the classification accuracy on the validation data has saturated, we stop training. The strategy is called ***early stopping***.

> Besides avoiding overfitting, this method is also used to  find the best hyper-parameters, which is called *hold out*. Because the validation data is kept apart or 'held out' from the train data.



 In general, one of the best ways of reducing overfitting is to increase the size of the training data. With enough training data it is difficult for even a very large network to overfit.



***Regularization*** is a very useful technique to address the overfitting problem, even when the architecture or the data is fixed. Next, we will introduce a mostly used regularization techniques, which is known as ***weight decay*** or ***L2 regularization***. The idea of this technique is to add an extra term to the cost function, which is called ***regularization term***. Here's the regularized cross-entropy:
$$
C=−\frac{1}{n}\underset{x}{∑}[ylna+(1−y)ln(1−a)] + \frac{\lambda}{2n}\underset{w}{\sum}w^2
$$
where $\lambda>0$ is known as ***regularization parameter***.

> Intuitively, the effect of regularization is *to **make it so the network prefers to learn small weights***, all other things being equal. Large weights will only be allowed if they considerably improve the first part of the cost function. ***Put another way, regularization can be viewed as a way of compromising between finding small weights and minimizing the original cost function.*** The relative importance of the two elements of the compromise depends on the value of $λ$: ***when $λ$ is small we prefer to minimize the original cost function, but when $λ$ is large we prefer small weights***.



Now we consider the problem: how the gradient descent work on the regularized cost function?
$$
\frac{∂C}{∂w} = \frac{∂C_0}{∂w} + \frac{\lambda}{n}w \\
\frac{∂C}{∂b} = \frac{∂C_0}{∂b}
$$
Thus we get that:
$$
w' = w-\eta \frac{∂C_0}{∂w}-\frac{\eta \lambda}{n}w = (1-\frac{\eta \lambda}{n})w - \eta \frac{∂C_0}{∂w}\\
b' = b-\eta \frac{∂C_0}{∂b}
$$

> The weight $w$ is rescaled by a factor $1-\frac{\eta \lambda}{n}$. This rescaling is sometimes referred to as ***weight decay***

> From above, we know that gradient descent still works. So does SGD.



> 除了减弱过拟合外，regularization还可以减小不同的初始化权重带来的影响，避免损失函数落入局部最小值。



//Why regu help reduce ovf?





**Weight initialization**









**How to choose a neural network's hyper-parameters**











**Other techniques**









**Other models of atrificial neuron**
