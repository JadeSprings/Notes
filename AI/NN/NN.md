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



***

Cost function, also called Loss function or Objective function:
$$
C(w,b) ≡ \frac{1}{2n}\underset{x}{\sum}||y(x)-a||^2
$$
C is also called *quadratic cost function*;it's also sometimes known as *the mean squared error* or just *MSE*.



To minimize the cost function $C(w,b)$, we'll use an algorithm known as ***gradient descent***(梯度下降).



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
这个公式的神奇之处在于：为我们提供了一种通过使C(cost function)不断变小来调整权重v(即,w和b)的策略。

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
w' = w+\Delta w = w-\eta ∇w \\
b' = b+\Delta b = b-\eta ∇b
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
\frac{\underset{m}{\sum}∇C_m}{m} ≈ \frac{\underset{x}{\sum}∇C_x}{n} = ∇C
$$
where m is the subset of the x(or n).

Thus we can get that:
$$
w' = w+\Delta w = w-\eta ∇w ≈ w - \frac{\eta}{m} \underset{m}{\sum} C_m =  w - \frac{\eta}{m} \underset{m}{\sum} \frac{∂C_m}{∂w}\\
b' = b+\Delta b = b-\eta ∇b ≈  b - \frac{\eta}{m} \underset{m}{\sum} C_m =  b - \frac{\eta}{m} \underset{m}{\sum} \frac{∂C_m}{∂b}
$$
where the sums are over all the training examples $C_m$ in the current mini-batch. Then we pick out another randomly chosen mini-batch and train with those. And so on, until we've exhausted the training inputs, which is said to complete an *epoch* of training. At that point we start over with a new training epoch.

> 有一点要注意：人们对cost function的定义并不总是相同的。例如在训练集样本数未知的情况下，我们的cost function仅仅是将模值相关的项累加（即没有$\frac{1}{n}$）。相关的计算也要对应修改。







***

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







