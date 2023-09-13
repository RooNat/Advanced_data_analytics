---
tags: Advanced_Data_Analytics
Week: 1
Module: 2
typora-copy-images-to: ./attachments
---

# Statistical Pattern Analysis

[toc]

## Overview

- **Reading List**: https://www.zotero.org/users/10221833/items/THFLDXIM Chapter 1

- Understand key concepts of **fitting models to data**
- Able to use simple methods for **model selection**
- Understand the relevance of the ‘**curse of dimensionality**’ to **model fitting and overfitting**
- Understand the **basic principles of information theory** as they apply to **pattern analysis**

### Overview of pattern analysis part of unit

- We will cover principles, algorithms, and application of linear and nonlinear methods for statistical pattern recognition.
- The primary focus will be on **dimensionality reduction**, but this will allow us to cover a range of other models in passing (e.g. Gaussian mixture models, Gaussian Processes, kernel methods, etc.).
- It is aimed to complement Introduction to AI. Some of material may overlap, but the principles are so fundamental, this is not a bad thing
- It cannot be comprehensive: the [Wikipedia page](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction) on non-linear dimensionality reduction methods lists 12 ‘Important Concepts’ and 14 ‘Other Algorithms’ and that does not include some of the methods that will be discussed in this unit!
- Theory in lectures will be followed by practical exercises in the labs.



## Polynomial Curve Fitting

Polynomial curve fitting is a **simple regression problem** which we can use to motivate and illustrate a number of key concepts in pattern analysis. 

We observe a **real-valued input variable** $x$ and wish to use this to **predict** the value of a **real-valued target variable** $t$.

In this **synthetic example**, we have generated data from the function $sin (2\pi x)$ with some **random noise** added to the target values.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677855410.png" alt="image-20230303145645996" style="zoom:50%;" />
$$
y(x,\boldsymbol{w})=w_0+w_1x+w_2x^2+\cdots+w_Mx^M=\sum_{j=0}^M w_jx^j
$$

- The **green curve** represents the ‘**true**’ function that the data was generated from.

- **Our goal** is to use the training set to make predictions $\hat t$ from previously unseen input values $\hat x$.

- This means that we try to discover the **underlying function**.

- This inherently difficult because we only have a finite data set (10 points in this example).

- The addition of noise means that there is an inherent uncertainty of the value of $\hat t$.  Our starting point is a simple curve-fitting approach using a polynomial in $x$.

- $M$ is the order of the polynomial, and the coefficients $w_i$ are gathered into a vector $\boldsymbol{w}$.

  ![Curve fitting-2](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678028819.jpg)
  
  ![Curve fitting-3](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678028860.jpg)

### Sum-of-Squares Error Function

How do we fit a polynomial function to the data? We create an error function that measures the misfit between the true function $y(x, \boldsymbol{w})$ and the **training set data points**. 

A common choice is the **sum-of-squares error function** shown below, where the factor **1/2** is included for later convenience.
$$
E(\boldsymbol{w})=\dfrac{1}{2}\sum_{n=1}^N\{y(x_n,\boldsymbol{w})-t_n\}^2
$$
It is a non-negative quantity that is **zero** if and only if the function $y(x, \boldsymbol{w})$ were to pass exactly through each training data point.

Because the error function is a **quadratic function** of the weights $\boldsymbol{w}$, its derivatives with respect to the coefficients are linear in the elements of $\boldsymbol{w}$ and so the **minimisation of the error function** has a unique solution $w^*$ which can be found in closed form.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677856949.png" alt="image-20230303152226563" style="zoom:50%;" />

**Draw the optimal solution for different degrees of polynomial:**

This group of schematics illustrate the problem of choosing the order $M$ of the polynomial. This is an example of the important concept called *model comparison* or *model selection*.

| 0th Order Polynomial                                         | 1st Order Polynomial                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677857639.png" alt="image-20230303153357550" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677857695.png" alt="image-20230303153453724" style="zoom:50%;" /> |
| **3rd Order Polynomial**                                     | **9th Order Polynomial**                                     |
| <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677857756.png" alt="image-20230303153553699" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677857795.png" alt="image-20230303153634168" style="zoom:50%;" /> |

*M* is the degree of the polynomial

We notice that the constant and first order polynomials fit the data poorly. **The third order polynomial seems to give the best fit.** 

The higher order polynomial (*M* = 9) gives a perfect fit to the data: it passes exactly through every data point and $E(w^*) = 0$. However, the fitted curve oscillates wildly and gives a very poor representation of the function $sin(2\pi x)$. This phenomenon is known as ***over-fitting***.

#### Over-fitting

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677858211.png" alt="image-20230303154328947" style="zoom:50%;" />

**Root-Mean-Square(RMS) Error:**
$$
E_{RMS}=\sqrt{2E(w^*)/N}
$$

- **The RMS error** is more convenient to use for comparison since the division by *N* , the number of data points, allows us to compare different sizes of data sets on an equal footing, while the square root ensures that $E_{RMS}$ is measured on the same scale and in the same units as the **target variable** *t*.
- The results shown in the graph may appear paradoxical because a polynomial of a given order contains all lower order polynomials as special cases. The *M* = 9 polynomial is capable of generating just as good results as the *M* = 3 polynomial.
- This shows that the limited nature of the training set and the decision to fit it as closely as possible are at the root of the problem.



**Polynomial Coefficients:**

Further insight can be obtained by examining the values of the coefficients $w^*$ obtained from polynomials of various order.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677858796.png" alt="image-20230303155314871" style="zoom:50%;" />

As *M* increases, the magnitude of the coefficients gets larger.

It is interesting to see how the behaviour of the model changes with the size of the dataset. For a given model complexity, the over-fitting problem becomes less severe as **the size of the dataset increases**.

Another view of this is that as the dataset gets larger, the more complex (flexible) the model that we can afford to fit to the data.

| Data Set Size: $N$=15                                        | **Data Set Size:** $N$=100                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677859172.png" alt="image-20230303155929543" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677859191.png" alt="image-20230303155949325" style="zoom:50%;" /> |



**Managing model complexity**

- One rough heuristic that is sometimes advocated is that the number of data points should be at least **5 or 10 times** the number of adaptive parameters in the model.
- But **the number of parameters** is not necessarily the best measure of model complexity.
- There is something unsatisfying about having to limit the number of parameters in this way. It would be better to choose the complexity of the model according to the complexity of the problem.
- This approach is an example of **maximum likelihood**, and over-fitting is a general property of this approach.
- By adopting a **Bayesian approach**, this problem can be avoided, at the cost of additional mathematical complexity.
- Indeed, **from the Bayesian perspective**, there is no difficulty in employing models for which the number of parameters greatly exceeds the number of data points. The effective number of parameters adapts automatically to the size of the dataset.



#### Regularization

**Penalize large coefficient values**:
$$
\tilde E(\boldsymbol{w})=\dfrac{1}{2} \sum_{n=1}^N \{y(x_n,\boldsymbol{w})-t_n\}^2 +\color{red}\dfrac{\lambda}{2}\|\boldsymbol{w}\|^2
$$
Before we go into the complexities of Bayesian inference, we will consider the simpler (and perhaps more general) technique for **control over-fitting** which is **regularisation**.

This involves adding an penalty term to the error function in order to discourage the coefficients from reaching large values. In this equation
$$
\|\boldsymbol{w}\|^2=\boldsymbol{w}^T\boldsymbol{w}=w_0^2+w_1^2+\cdots+w_M^2
$$
is the squared length/magnitude of the vector $\boldsymbol{w}$ and the coefficient $\lambda$ governs the relative importance of the regularisation term compared with the sum-of-squares error term. (control the amount of regularisation)

- The coefficient $w_0$ is often left out since its inclusion causes the result to depend on the choice of origin for the target variable (it is the mean of the target variable).
- An alternative approach is to give it its own regularisation parameter. You will also note that this discussion leaves out the question of how to determine the best value of $\lambda$. Often this is done by **cross-validation**[^1]: a third dataset (distinct from both training and test sets) used to optimise the model complexity.

| $\text{ln}\lambda=-18$                                       | $\text{ln}\lambda=0$                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677859919.png" alt="image-20230303161157888" style="zoom:50%;" /> | <img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677859939.png" alt="image-20230303161218084" style="zoom:50%;" /> |



**Regularization** $E_{RMS}$ **vs** $\text{ln} \lambda$

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677860041.png" alt="image-20230303161400655" style="zoom:50%;" />

**Polynomial Coefficients**

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230303_1677860103.png" alt="image-20230303161501999" style="zoom:50%;" />

## Likelihood and Model Fitting

> ***Relevant videos:*** https://youtu.be/VQ1dxoopfEI

 [Proability theory.md](Proability theory.md) 

<u>**Gaussian function**</u>:  [14 Continuous random variables and limit raws.md](../../R_Projects/My R-note/14 Continuous random variables and limit raws.md) 

**<u>Likelihood function</u>**:  [16 An introduction to maximum likelihood estimation.md](../../R_Projects/My R-note/16 An introduction to maximum likelihood estimation.md) 

Recall that the **Gaussian distribution** and **Central limit theorem**:

An obvious question to ask is “**Given a set of data, what is the best fitting Gaussian?**”:

Suppose that we have a dataset of observations $\boldsymbol{x} = (x_1,\cdots, x_N )^T$ representing *N* observations of the scalar variable $x$.

We suppose that these observations are drawn from a **Gaussian distribution** whose mean $\mu$ and variance $\sigma^2$ are unknown.

Because the dataset is i.i.d., we can write the probability of the dataset given $\mu$ and $\sigma^2$ as shown below.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677926819.png" alt="image-20230304104656951" style="zoom:50%;" />
$$
p(\boldsymbol{x}|\mu,\sigma^2)=\prod_{n=1}^N\mathcal{N}(x_n|\mu,\sigma^2)
$$
N.B. This is a function of $\mu$ and $\sigma$.

What we are going to do now is to find the parameters that maximise the likelihood function. It might seem more natural to maximise the probability of the parameters given the data, not the probability of the data given the parameters. It turns out that these two concepts are related.
$$
\text{ln}p(\boldsymbol{x}|\mu,\sigma^2)=-\dfrac{1}{2\sigma^2}\sum_{n=1}^N(x_n-\mu)^2-\dfrac{N}{2}\text{ln}\sigma^2-\dfrac{N}{2}\text{ln}(2\pi)
$$

$$
\mu_{\text{ML}}=\dfrac{1}{N}\sum_{n=1}^N x_n \quad \sigma^2_{\text{ML}}=\dfrac{1}{N}\sum_{n=1}^{N}(x_n-\mu_{\text{ML}})^2
$$

It turns out that the calculations are easier by considering the log of the maximum likelihood. Differentiating this with respect to $\mu$ and $\sigma$ in turn, and setting the values to zero, leads easily to the closed form solutions.

There are significant limitations to the maximum likelihood approach, as we shall see later. For now, we can look at some issues that arise for the Gaussian distribution. In particular, the maximum likelihood approach systematically underestimates the variance of the distribution. This is an example of the phenomenon called *bias* which is related to **over-fitting**.
$$
\mathbb{E}[\mu_{\text{ML}}]=\mu \\
\mathbb{E}[\sigma^2_{\text{ML}}]=\left(\dfrac{N-1}{N}\right)\sigma^2
$$

$$
\tilde \sigma^2=\dfrac{N}{N-1}\sigma^2_{\text{ML}} \\
=\dfrac{1}{N-1} \sum_{n=1}^N (x_n-\mu_{\text{ML}})^2
$$

We note that $\mu_{ML}$ and $\sigma^2_{ML}$ are both functions of the dataset values. Consider the expectations of these quantities with respect to the dataset values, and we get the equations on the slide. On average, the true variance is underestimated by $(N − 1)/N$ .

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677929763.png" alt="image-20230304113601834" style="zoom:50%;" />

In the graph, the green curve shows the true Gaussian distribution from which data is generated while the red curves show the Gaussian distribution obtained by fitting to three datasets, each consisting of two data points shown in blue, using maximum likelihood.

**Averaged across the three datasets, the mean is correct, but the variance is systematically under-estimated because it is measured relative to the sample mean and not relative to the true mean.**

In this case, the bias is relatively small, and becomes less significant as $N \to \infty$. But in this case, we are only fitting two parameters to the data: later we shall be using models with many more parameters!



### Curve Fitting Re-visited

> ***Relevant videos:*** https://www.youtube.com/watch?v=NyH9K3stvP8

![Curve fitting-4](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678028946.jpg)

Essentially, the curve that you see represents the $\mu$ or the mean values, or we can say that the polynomial will give us the mean value of that target variable.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678015373.png" alt="image-20230305112252066" style="zoom: 25%;" />

In the curve fitting task, we can express our **uncertainty** over the value of the target variable using a **probability distribution**. For this purpose, we will **<u>use a conditional Gaussian distribution</u>**. Here $\beta^{-1}$ is the *precision* of the distribution which is the **inverse variance** (for consistency with later discussions).

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677929912.png" alt="image-20230304113830444" style="zoom:50%;" />



#### Maximum likelihood

![Curve fitting-5](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678028997.jpg)

![Curve fitting-6](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678029015.jpg)

Likelihood and probability distributions differ from each other, as probability distribution has an additional requirement of being normalized.
$$
p(\boldsymbol{t}|\boldsymbol{x},\boldsymbol{w},\beta)=\prod_{n=1}^N\mathcal{N}(t_n|y(x_n,\boldsymbol{w}),\beta^{-1})
$$

$$
\text{ln}p(\boldsymbol{t}|\boldsymbol{x},\boldsymbol{w},\beta)=-\underbrace{\dfrac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\boldsymbol{w})-t_n\}^2}_{\beta E(\boldsymbol{w})}+\dfrac{N}{2}\text{ln} \beta-\dfrac{N}{2}\text{ln}(2\pi)
$$

Determine $\boldsymbol{w}_{ML}$ by minimize [sum-of-squares](###Sum-of-Squares Error Function), $E(\boldsymbol{w})$.
$$
\dfrac{1}{\beta_{ML}}=\dfrac{1}{N}\sum_{n=1}^N\{y(x_n,\boldsymbol{w}_{ML})-t_n\}^2
$$
We now use the training data to determine the parameters $\boldsymbol{w}$ and $\beta$ by **maximum likelihood**.

Note that we are doing more than in the original example, because we are also trying to estimate the variance/precision of the noise model. We assume that the data is drawn independently from the distribution.

When maximising with respect to $\boldsymbol{w}$ we can omit the **last two terms** (since they don’t depend on $w$) and the scaling of the function does not change where the maximum is found, 

so we can replace $\beta/2$ by $1/2$. This is then directly equivalent to the sum-of-squares error function. Another way of saying this is that sum-of-squares is equivalent to maximum likelihood with a Gaussian noise model.

- We convert the **Log Likelihood** to the **Negative Log Likelihood** because we want to use the **minimum** procedure to find the stationary point, instead of **maximizing the log likelihood** we do the **minimization of negative log likelihood**.

#### Predictive distribution

Having determined the parameters $\boldsymbol{w}$ and $\beta$, we can now make predictions for new values of $x$. Because we have a probabilistic model, we obtain a *predictive distribution* that gives the probability distribution for $t$ by substituting in the maximum likelihood estimates.
$$
p(t|x,\boldsymbol{w}_\text{ML},\beta_{\text{ML}})=\mathcal{N}(t|y(x,\boldsymbol{w}_{\text{ML}}),\beta_{\text{ML}}^{-1})
$$
<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677932579.png" alt="image-20230304122257402" style="zoom:50%;" />

**What did we learn by this then?**

1. It's important to know the **uncertainty** in **your prediction**
2. ~~You learned how to think in terms of probabilistic modeling~~
3. You started to think in terms of probabilistic modeling
4. You saw a formal justification of **sum squared error** function which arises when we assume **gaussian noise**.

 

Most important takeaway this part is to think of your target variables as **random variables**, it helps us move away from **point estimate** to **predictive distribution**.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678021632.png" alt="image-20230305130707705" style="zoom:50%;" />

---

### Bayesian Curve Fitting

![Curve fitting-7](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678029060.jpg)

![Curve fitting-8](https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230305_1678029087.jpg)

---

One way to have a confidence about these **unknowns** is to treat them as **random variables**. This way we can have a probability distribution associated with various values. Our unknown parameters can take.

---



We can take a step towards a more **Bayesian approach** by placing a prior distribution over $\boldsymbol{w}$. For simplicity, we choose a Gaussian distribution with **zero mean** and **precision** $\alpha$. Recall that $M+1$ is the number of elements in the vector $\boldsymbol{w}$ for a polynomial of order $M$. 

Variables such as $\alpha$ which control the distribution of model parameters are called *hyperparameters*. Using Bayes’ theorem, the **posterior distribution** for $\boldsymbol{w}$ is proportional to the product of the **prior distribution** and the likelihood function.
$$
p(\boldsymbol{w}|\alpha)=\mathcal{N}(\boldsymbol{w}|\boldsymbol{0},\alpha^{-1}\boldsymbol{\text{I}})=\left(\dfrac{\alpha}{2 \pi}\right)^{(M+1)/2}\text{exp}\left\{-\dfrac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w}\right\} \\
p(\boldsymbol{w}|\boldsymbol{x,t},\alpha,\beta)\propto p(\boldsymbol{t}|\boldsymbol{x,w},\beta)p(\boldsymbol{w}|\alpha) \\
\beta \tilde{E}(\boldsymbol{w})=\dfrac{\beta}{2}\sum_{n=1}^{N}\{y(x_n,\boldsymbol{w})-t_n\}^2+\dfrac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w}
$$
Determine $\boldsymbol{w}_{MAP}$ by minimising regularized sum-of-squares error, $\tilde E(\boldsymbol{w})$

We can now determine $\boldsymbol{w}$ by finding the most probable value of $\boldsymbol{w}$ given the data, in other words by maximising the **posterior distribution**. This is called *maximum a posterior* or MAP. Taking the negative logarithm of the equation, we find that the MAP solution is found by the minimum of the third equation: this is equivalent to the regularised sum-of-squares error function with regularisation parameter $\lambda=\alpha/\beta$
$$
p(t|x,\boldsymbol{x},\boldsymbol{t})=\int p(t|x,\boldsymbol{w})p(\boldsymbol{w}|\boldsymbol{x,t})\text{d}\boldsymbol{w}=\mathcal{N}(t|m(x),s^2(x)) \\
m(x)=\beta \phi(x)^T \boldsymbol{\text{S}}\sum_{n=1}^N \phi(x_n)t_n \quad s^2(x) =\beta^{-1}+\phi(x)^T \boldsymbol{\text{S}}\phi(x) \\
\boldsymbol{\text{S}}^{-1}=\alpha\boldsymbol{\text{I}}+\beta\sum_{n=1}^{N} \phi(x_n)\phi(x_n)^T \quad \phi(x_n)=(x_n^0,\cdots, x_n^M)^T
$$
A fully Bayesian treatment would integrate over all possible values of $\boldsymbol{w}$ weighted by the corresponding probability. This marginalisation is at the heart of the Bayesian approach to pattern analysis.

This is represented in the first equation above (omitting the dependence on $\alpha$ and $\beta$ for simplicity). It turns out that this can be evaluated analytically (that is not usually the case: Gaussians make our life a lot easier here!) and the predictive distribution is also Gaussian with mean $m(x)$ and variance $s^2(x)$.

In the equation for $s^2(x)$, the first term is the noise on the target variables but the second term arises from the uncertainty in the parameters $\boldsymbol{w}$ and is a consequence of the Bayesian treatment. The consequence of this additional uncertainty can be seen on the graph below. Here the hyperparameters $\alpha=5\times10^{-3}$ and $\beta=1$ (which is cheating a bit since it is the known noise variance). The red curve denotes the mean of the predictive distribution and the red region corresponds to $\pm1$ standard deviation around the mean.
$$
p(t|x,\boldsymbol{x,t})=\mathcal{N}(t|m(x),s^2(x))
$$
<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677935777.png" alt="image-20230304131615718" style="zoom:50%;" />

#### Curse of Demensionality

The curve fitting approach we have described works pretty well with a single input variable $x$. In fact, there probably wouldn’t be such a subject as machine learning.

But in practice, we usually have to deal with spaces of high dimensionality comprising many input variables. We can see the problem that arises by considering a region of space divided into regular cells. The number of such cells grows exponentially with the dimensionality of the space. The problem with an exponential number of cells is that would need an exponentially large quantity of training data to ensure that the cells are not empty.

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677935856.png" alt="image-20230304131735295" style="zoom:50%;" />

A general polynomial with coefficients up to order 3 has the form given on the slide. As $D$ increases, the number of independent coefficients grows proportionally to $D^3$. In practice, we might need to use a higher-order polynomial. For a polynomial of order $M$ , the growth in the number of coefficients as like $D^M$ . Although this is a power law rather than exponential, as $D$ gets larger, we might have to increase $M$ as well. At any rate, the method will rapidly become unwieldy and impractical.

**Polynomial curve fitting, M=3**
$$
y(\boldsymbol{x},\boldsymbol{w})=w_0+\sum_{i=1}^D w_ix_i+\sum_{i=1}^D \sum_{j=1}^D w_{ij}x_ix_j+\sum_{i=1}^D\sum_{j=1}^D\sum_{k=1}^D w_{ijk}x_ix_jx_k
$$
**Gaussian Densities in higher dimensions**

<img src="https://raw.githubusercontent.com/RooNat/Myimages/main/2023/03/upgit_20230304_1677936169.png" alt="image-20230304132247876" style="zoom:50%;" />

Consider the behaviour of a Gaussian distribution in a high-dimensional space. If we transform from Cartesian to polar coordinates and integrate out the directional variables, we obtain an expression for the density $p(r)$ as a function of radius $r$ from the origin. Thus $p(r)\delta r$ is the probability mass in a thin shell of thickness $\delta r$ located at radius $r$. This distirbution is plotted for various values of $D$ in the figure. We see that for large $D$ the probability mass of the Gaussian is concentrated in a thin shell quite a long way from the origin.

#### Pattern analysis in high diemensions

- We can still find effective techniques in high dimensions for two reasons:
- Real data is often confined to a region of the space having lower effective dimensionality. In particular, the directions over which important variations in the target variables occur may be so confined.
- Real data typically exhibits some smoothness properties (at least locally) so that for the most part small changes in input variables produce small changes in the target variables. We can exploit local interpolation-like techniques to allow us to make predictions of the target variables.
