---
tags: Advanced_Data_Analytics
Week: 1
Module: 1
typora-copy-images-to: ./attachments
---

# Probability theory

[toc]

## Overview

 [11 Conditional probability, Bayes rule and independence.md](../../R_Projects/My R-note/11 Conditional probability, Bayes rule and independence.md) 

- Understand the basics of probability theory, with a particular focus on **conditional probability**
- Understand **Bayes’ theorem** and some of its implications

 [14 Continuous random variables and limit raws.md](../../R_Projects/My R-note/14 Continuous random variables and limit raws.md) 

- Able to work with **probability density functions**
- Understand **expectations**, **means** and **covariances**

 [Unsupervised learning.md](../../Introduction to Artificial intelligence/Week3/Unsupervised learning.md) 

- Understand the basic features of **univariate** and **multivariate Gaussian distributions**

## Probability Theory

 [Supervised Learning.md](../../Introduction to Artificial intelligence/Week2/Supervised Learning.md) 

There are two boxes, suppose the probability of choosing **the red box is 0.4**, and the probability of choosing **the blue box is 0.6**.

At the same time, the orange sphere is orange, and the green sphere is apple.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677762015.png" alt="image-20230302130013255" style="zoom:50%;" />



**There are two random variables:**

The column is $x_i$ , the row is $y_i$, and the value of $x_i$ is $c_i$ , the value of $y_i$ is $r_j$.

The total value is $N$.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/02/upgit_20230206_1675681684.png" alt="image-20230206102536223" style="zoom:50%;" />



The schematic shows that: 
$$
c_i=\sum_jn_{ij} \quad \sum_{ij}n_{ij}=N
$$

- **Marginal Probability:**
  $$
  p(X=x_i)=\dfrac{c_i}{N} \quad p(Y=y_j)=\dfrac{r_j}{N}
  $$
  
- **Joint Probability:**
  $$
  p(X=x_i,Y=y_j)=\dfrac{n_{ij}}{N}
  $$
  
- **Conditional Probability:**
  $$
  p(Y=y_j|X=x_i)=\dfrac{n_{ij}}{c_i}
  $$
  
- **Sum Rule:**

  Here $p(X, Y )$ is a joint probability and is verbalized as “the probability of X and Y ”.
  $$
  p(X=x_i)=\dfrac{c_i}{N}=\dfrac{1}{N}\sum_{j=1}^{L}n_{ij}\\ 
  =\sum_{j=1}^{L}p(X=x_i,Y=y_j)
  $$

  $$
  p(X)=\sum_{Y}p(X,Y)
  $$

  

- **Product Rule:**

  Similarly, the quantity $p(Y |X)$ is a conditional probability and is verbalized as “the probability of Y given X”

$$
p(X=x_i,Y=y_j)=\dfrac{n_{ij}}{N}=\dfrac{n_{ij}}{c_i}\cdot \dfrac{c_i}{N} \\
=p(Y=y_j|X=x_i)p(X=x_i)
$$



****

## Bayes' Theorem

$$
p(Y|X)=\dfrac{p(X|Y)p(Y)}{p(X)}\\
p(X)=\sum_{Y}p(X|Y)p(Y)
$$



## Probability Densities

$$
p(x\in(a,b))=\int_{a}^{b}p(x)\mathrm{d}x
$$

$$
P(z)=\int_{-\infty}^zp(x)\text{d}x
$$



- Integrate the density between $a$ and $b$ that gives the total mass.

- The density function can be larger than 1, but:
  $$
  \int_{-\infty}^{\infty}p(x)\text{d}x=1 \quad p(x) \geq 0
  $$
  <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677763238.png" alt="image-20230302132036238" style="zoom:50%;" />

## Expectations

- Computing expected outcomes for a random variable

- The **expectation for discrete random**:
  $$
  \mathbb{E}[f]=\sum_x p(x)f(x)
  $$

- The **expectation for continuous random**:
  $$
  \mathbb{E}[f]=\int p(x)f(x) \text{d}x
  $$

- **Conditional Expectation**:
  $$
  \mathbb{E}_{x}[f|y]=\sum_xp(x|y)f(x)
  $$

- **Approximate Expectation**(discrete and continuous) based on a sample:
  $$
  \mathbb{E}[f] \simeq\dfrac{1}{N} \sum_{n=1}^N f(x_n)
  $$
  

## Variances and Covariances

#### Variance

$$
\text{Var}[f]=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^2]=\mathbb{E}[f(x)^2]-\mathbb{E}[f(x)]^2
$$



#### Covariance

$$
\begin{align}
\mathrm{cov}[x,y] &=\mathbb{E}_{x,y}[(x-\mathbb{E}[x])(y-\mathbb{E}[y])] \\
&=\mathbb{E}_{x,y}[xy]-\mathbb{E}[x][y]
\end{align}
$$

- This is a matrix
- We have to bring out the transpose here.

$$
\begin{align}
\mathrm{cov}[\vec x,\vec y] &=\mathbb{E}_{\vec x, \vec y}[(\vec x-\mathbb{E}[\vec x])(\vec y^T-\mathbb{E}[\vec y ^T])] \\
&=\mathbb{E}_{\vec x,\vec y}[\vec x \vec y^T]-\mathbb{E}[\vec x]\mathbb{E}[\vec y ^T]
\end{align}
$$



## The Gaussian Distribution

$$
\mathcal{N}(x|\mu,\sigma^2)=\dfrac{1}{(2\pi \sigma^2)^{1/2}}\text{exp}\left\{-\dfrac{1}{2\sigma^2}(x-\mu)^2\right\}
$$

$$
\mathcal{N}(x|\mu,\sigma^2)>0 \\
\int_{-\infty}^{\infty} \mathcal{N}(x|\mu,\sigma^2)\text{d}x=1
$$

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677764080.png" alt="image-20230302133438875" style="zoom:50%;" />



### Gaussian Mean and Variance

$$
\mathbb{E}[x]=\int_{-\infty}^{\infty} \mathcal{N}(x|\mu,\sigma^2)x\text{d}x=\mu
$$

?????
$$
\mathbb{E}[x^2]=\int_{-\infty}^{\infty} \mathcal{N}(x|\mu,\sigma^2)x^2 \text{d}x=\mu^2+\sigma^2
$$

$$
\text{var}[x]=\mathbb{E}[x^2]-\mathbb{E}[x]^2=\sigma^2
$$



## Central Limit Theorem

The distribution of **the sum of N i.i.d. random variables** becomes increasingly Gaussian as **N** grows.

**Example**: **N** uniform $[0,1]$ random variables.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677764596.png" alt="image-20230302134314763" style="zoom:50%;" />

## The Multivariate Gaussian

$$
\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu},\Sigma)=\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\text{exp}\left\{-\dfrac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right\}
$$

- $\Sigma$ is the matrix $\sigma$, get entries for each pair of $x_i$ and $x_j$.
- The Multivariate Gaussian measures lots of variables
- $-\dfrac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$ is the **quadratic form**

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677767015.png" alt="image-20230302142332162" style="zoom:50%;" />

(1) $\Delta^2=(\boldsymbol{x}-\boldsymbol{\mu})^{T} \Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$

- We begin by considering the geometrical form of the Gaussian distribution. The functional dependence of the Gaussian on $x$ is through the **quadratic form**，which appears in the exponent.

- The quantity $\Delta$ is called the **Mahalanobis distance** from $\mu$ to $x$ and reduces to the Euclidean distance when $\Sigma$ is the **identity matrix**.

- The **Gaussian distribution** will be **constant** on surfaces in **x-space** for which this quadratic form is **constant**.

- Now consider the **eigenvector**[^1] equation for the **covariance matrix**:
  
  **[1]**
  $$
  \Sigma \boldsymbol{u}_i=\lambda_i \boldsymbol{u}_i
  $$
  where $i=1,\cdots, D$. Because $\Sigma$ is a real, **symmetric matrix**[^2] its **eigenvalues**[^1] will be real, and its **eigenvectors** can be chosen to form an **orthonormal set**, so that
  
  **[2]** 
  $$
  \boldsymbol{u}_i^T\boldsymbol{u}_j=I_{ij}
  $$
  Where $I_{ij}$ is the $i,j$ element of the **identity matrix**[^2] and satisfies:
  $$
  I_{ij}=\begin{cases}
  1, & \quad \text{if} \quad i=j \\
  0, & \quad \text{otherwise}.
  \end{cases}
  $$
  The covariance matrix $\Sigma$ can be expressed as an expansion in terms of its eigenvectors in the form
  
  **[3]**
  $$
  \Sigma=\sum_{i=1}^D \lambda_i u_i u_i^T
  $$
  

(2) Similarly the inverse covariance matrix $\Sigma^{-1}$ can be expressed as:
$$
\Sigma^{-1}=\sum_{i=1}^{D}\dfrac{1}{\lambda_i}\boldsymbol{u}_i \boldsymbol{u}_i^T
$$
(3) Substituting ((1)) and ((2)), the quadratic form becomes:
$$
\Delta^2=\sum_{i=1}^{D}\dfrac{y_i^2}{\lambda_i}
$$
(4)  where we have defined
$$
y_i=\boldsymbol{u_i}^T(\boldsymbol{x}-\boldsymbol{\mu})
$$
We can interpret $\{y_i\}$ as a new **coordinate system** defined by the **orthonormal vectors** $u_i$ that are shifted and rotated with respect to the original $x_i$ coordinates. Forming the vector $\boldsymbol{y} = (y_1, \cdots , y_D)^T$, we have
$$
\boldsymbol{y}=\boldsymbol{\text{U}}(\boldsymbol{x}-\boldsymbol{\mu})
$$
where $\boldsymbol{\text{U}}$ is a matrix whose rows are given by $\boldsymbol{u}_i^T$.  From **[2]** it follows that $\boldsymbol{\text{U}}$ is an *orthogonal* matrix, i.e., it satisfies $\boldsymbol{\text{U}\text{U}^T}=\boldsymbol{\text{I}}$, and hence also $\boldsymbol{\text{U}^T}\boldsymbol{\text{U}}=\boldsymbol{\text{{I}}}$ , where $\boldsymbol{\text{I}}$ is the identity matrix.

The quadratic form, and hence the Gaussian density, will be constant on surfaces for which (4) is constant. If all of the eigenvalues $\lambda_i$ are positive, then these surfaces represent **ellipsoids**, with their centres at $\mu$ and their axes oriented along $u_i$, and with scaling factors in the directions of the axes given by $\lambda_{i}^{1/2}$.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677768022.png" alt="image-20230302144019340" style="zoom:50%;" />

#### Eigenvalues and Eigenvectos

> ***Reference:*** https://towardsdatascience.com/eigenvectors-and-eigenvalues-all-you-need-to-know-df92780c591f

![image-20230303135404969](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677851646.png)

### Moments of the Multivariate Gaussian

We now note that the **exponent** is an **even function** of the components of $z$ and, because the integrals over these are taken over the range $(-\infty, +\infty)$, the term in $z$ in the factor $(z+\mu)$ will vanish by **symmetry**.
$$
\mathbb{E}[x]=\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\int \text{exp}\left\{-\dfrac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\} \boldsymbol{x} \text{d}x \\
=\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\int \text{exp}\left\{-\dfrac{1}{2}\boldsymbol{z}^T\Sigma^{-1}\boldsymbol{z} \right\}(\boldsymbol{z}+\boldsymbol{\mu})\text{d}\boldsymbol{z}
$$

Thanks to **anti-symmetry** of $\boldsymbol{z}$
$$
\mathbb{E}[x]=\boldsymbol{\mu}
$$

We now consider second order moments of the Gaussian. For the **multivariate Gaussian**, there are $D^2$ second order moments given by $\mathbb{E}[x_i x_j]$, which we can group together to form the matrix $\mathbb{E}[\boldsymbol{xx}^T]$. This matrix can be written as
$$
\mathbb{E}[\boldsymbol{xx}^T]=\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\int \text{exp}\left\{-\dfrac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\} \boldsymbol{xx}^T \text{d}x \\
=\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\int \text{exp}\left\{-\dfrac{1}{2}\boldsymbol{z}^T\Sigma^{-1}\boldsymbol{z} \right\}(\boldsymbol{z}+\boldsymbol{\mu})(z+\boldsymbol{\mu})^T\text{d}\boldsymbol{z}
$$
where again we have changed variables using $\boldsymbol{z}=\boldsymbol{x-\mu}$. Note that the cross-terms involving $\boldsymbol{\mu z}^T$ and $\boldsymbol{\mu}^T\boldsymbol{z}$ will again vanish by symmetry. The term $\boldsymbol{\mu\mu^T}$ is constant and can be taken outside the integral, which itself is unity because the Gaussian distribution is normalized. Consider the term involving $\boldsymbol{zz^T}$ . Again, we can make use of the eigenvector expansion of the covariance matrix given by **[1]** , together with the completeness of the set of eigenvectors, to write
$$
\boldsymbol{z}=\sum_{j=1}^D y_j \boldsymbol{u_j}
$$
Where $y_j=\boldsymbol{u_j}^T\boldsymbol{z}$, which gives
$$
\dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\int \text{exp}\left\{-\dfrac{1}{2}\boldsymbol{z}^T\Sigma^{-1}\boldsymbol{z} \right\} \boldsymbol{z}\boldsymbol{z^T}\text{d}\boldsymbol{z} \\
= \dfrac{1}{(2\pi)^{D/2}}\dfrac{1}{|\Sigma|^{1/2}}\sum_{i=1}^D \sum_{j=1}^D \boldsymbol{u_i u_j^T} \int \text{exp} \left\{ -\sum_{k=1}^D \dfrac{y_k^2}{2\lambda_k}\right\}y_iy_j \text{d} \boldsymbol{y} \\
=\sum_{i=1}^D \boldsymbol{u_i u_i^T} \lambda_i =\Sigma
$$
where we have made use of the eigenvector equation **[1]**, together with the fact that the integral on the right-hand side of the middle line vanishes by symmetry unless $i=j$, and in the final line we have made use of the eigendecomposition, together with **[3]** . Thus we have
$$
\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^T]=\boldsymbol{\mu}\boldsymbol{\mu}^T+\Sigma
$$

$$
\text{cov}[\boldsymbol{x}]=\mathbb{E}[(\boldsymbol{x}-\mathbb{E}[x])(\boldsymbol{x}-\mathbb{E}[x])^T]=\Sigma
$$

| $\Sigma= \begin{bmatrix}\lambda_1 \quad \neq 0 \\ \neq 0 \quad \lambda_2 \end{bmatrix}$ | $\Sigma= \begin{bmatrix}\lambda_1 \quad 0 \\0 \quad \lambda_2 \end{bmatrix}$ | $\Sigma=\lambda I$                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677768967.png" alt="image-20230302145604456" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677768984.png" alt="image-20230302145622061" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230302_1677768997.png" alt="image-20230302145635587" style="zoom:50%;" /> |





[^2]: https://www.mathsisfun.com/algebra/matrix-types.html



[^1]: https://www.mathsisfun.com/algebra/eigenvalue.html
