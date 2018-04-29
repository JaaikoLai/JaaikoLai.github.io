title: bacth normlization
author: John Doe
tags: []
categories: []
mathjax: true
date: 2018-04-28 16:38:00
---
## bacth normlization中的前向传播
input:Xi(本层所有的样本矩阵为X维度为mxD，m为样本数，D为神经元的个数，其中Xi为X中的某一个样本)
$$ output格式为: Yi = BN(Xi, \gamma, \beta) $$
前向传播过程如下
$$\mu = E(X) = \frac{1}{m}\sum_{i=1}^{m}Xi\\\\
\sigma^{2} = Var(X) = \frac{1}{m}\sum_{i=1}^{m}(Xi-\mu)^{2}\\\\
\hat{Xi} = \frac{Xi - \mu}{\sqrt{\sigma^{2} + \varepsilon }}\\\\
Yi = \gamma  \cdot \hat{Xi} + \beta$$
## bacth normlization中的反向传播
假设上层梯度为dY，本层输入为X，过程如下：
$$d\beta = \sum_{i=1}^{m}\frac{dLi}{dYi} \cdot \frac{dYi}{d\beta}\\\\
由Yi = \gamma  \cdot \hat{Xi} + \beta可知，\frac{dYi}{d\beta}=1\\\\所以d\beta = \sum_{i=1}^{m}\frac{dLi}{dYi}$$
$$d\gamma = \sum_{i=1}^{m}\frac{dLi}{dYi} \cdot \frac{dYi}{d\gamma}\\\\
由Yi = \gamma  \cdot \hat{Xi} + \beta可知，\frac{dYi}{d\gamma}=\hat{Xi}\\\\所以d\gamma = \sum_{i=1}^{m}\frac{dLi}{dYi}\cdot \hat{Xi}$$
最后我们还要对X进行求导，首先我们先看下面的链式路径：
![image](/images/chain_batchnorm.png)
$$对单个样本Xi我们要求\frac{dYi}{dXi} = \frac{dYi}{dXi} = \frac{dYi}{d\hat{Xi}} \cdot \frac{d\hat{Xi}}{dXi}\\\\
由Yi = \gamma  \cdot \hat{Xi} + \beta可知\frac{dYi}{d\hat{Xi}}=\gamma\\\\
由\hat{Xi} = \frac{Xi - \mu}{\sqrt{\sigma^{2} + \varepsilon }}，\mu = \frac{1}{m}\sum_{i=1}^{m}Xi，
\sigma^{2} = \frac{1}{m}\sum_{i=1}^{m}(Xi-\mu)^{2}和上图可知我们求\frac{d\hat{Xi}}{dXi}的话有三条路径：\\\\
第一条路径为：\frac{d\hat{Xi}}{dXi} = \frac{1}{\sqrt{\sigma^{2} + \varepsilon }}\\\\
第二条路径为：\frac{d\hat{Xi}}{dXi} = \frac{d\hat{Xi}}{d\mu} \cdot \frac{d\mu}{dXi}，\frac{d\hat{Xi}}{d\mu} = -\frac{1}{\sqrt{\sigma^{2} + \varepsilon }},\frac{d\mu}{dXi} = \frac{1}{m}Xi\\\\
则\frac{d\hat{Xi}}{dXi} = -\frac{Xi}{m\sqrt{\sigma^{2} + \varepsilon }}\\\\
第三条路径为：\frac{d\hat{Xi}}{dXi} = \frac{d\hat{Xi}}{d\sigma^{2}} \cdot \frac{d\sigma^{2}}{dXi}，\\\\ \frac{d\hat{Xi}}{d\sigma^{2}} = -\frac{Xi - \mu}{2(\sigma^{2} + \varepsilon)^{\frac{3}{2}}}，\\\\
求解\frac{d\sigma^{2}}{dXi}有两个路径：
路径1：\frac{d\sigma^{2}}{dXi} = \frac{2}{m}(Xi - \mu)，路径2：\frac{d\sigma^{2}}{dXi} = \frac{d\sigma^{2}}{d\mu} \cdot \frac{d\mu}{dXi} = - \frac{2}{m}(Xi - \mu) \cdot \frac{1}{m}Xi\\\\
则\frac{d\sigma^{2}}{dXi} = \frac{d\sigma^{2}}{dXi} + \frac{d\sigma^{2}}{d\mu} \cdot \frac{d\mu}{dXi} = \frac{2}{m}(Xi - \mu) + (- \frac{2}{m}(Xi - \mu) \cdot \frac{1}{m}Xi) = \frac{2}{m^{2}}(Xi - \mu)(m - Xi)\\\\
则\frac{d\hat{Xi}}{dXi} = \frac{d\hat{Xi}}{d\sigma^{2}} \cdot \frac{d\sigma^{2}}{dXi} = -\frac{Xi - \mu}{2(\sigma^{2} + \varepsilon)^{\frac{3}{2}}} \cdot \frac{2}{m^{2}}(Xi - \mu)(m - Xi) = \frac{(Xi - m)}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}}\\\\
综合上述三条路径可求得\frac{d\hat{Xi}}{dXi} = \frac{d\hat{Xi}}{dXi} + \frac{d\hat{Xi}}{d\mu} \cdot \frac{d\mu}{dXi} + \frac{d\hat{Xi}}{d\sigma^{2}} \cdot \frac{d\sigma^{2}}{dXi} = \frac{1}{\sqrt{\sigma^{2} + \varepsilon }} - \frac{Xi}{m\sqrt{\sigma^{2} + \varepsilon }} + \frac{(Xi - m)}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}} \\\\= \frac{m^{2}(\sigma^{2} + \varepsilon) - m(\sigma^{2} + \varepsilon)Xi +(Xi - m)}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}} = \frac{(Xi - m)\cdot (1 - m(\sigma^{2} + \varepsilon))}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}}\\\\
则\frac{dYi}{dXi} = \frac{dYi}{d\hat{Xi}} \cdot \frac{d\hat{Xi}}{dXi} =  \frac{\gamma \cdot (1 - m(\sigma^{2} + \varepsilon))}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}}\cdot(Xi - m) 且 i = 1...m\\\\
总结一下：\\\\
d\beta = \sum_{i=1}^{m}\frac{dLi}{dYi}\\\\
d\gamma = \sum_{i=1}^{m}\frac{dLi}{dYi}\cdot \hat{Xi}\\\\
\frac{dYi}{dXi} = \frac{\gamma \cdot (1 - m(\sigma^{2} + \varepsilon))}{m^{2}\cdot (\sigma^{2} + \varepsilon)^{\frac{3}{2}}}\cdot(Xi - m) 且 i = 1...m\\\\
\frac{dY}{dX} = \begin{bmatrix}
\frac{dY1}{dX1}\\\\
\frac{dY2}{dX2}\\\\
.\\\\
.\\\\
\frac{dYm}{dXm}\\\\
\end{bmatrix}
$$