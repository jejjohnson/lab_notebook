---
## Inducing Inputs

**Resources**

* Neil - [Deep GPs](http://inverseprobability.com/talks/notes/deep-gaussian-processes.html)
  > A good long lecture about variational compression and how it relates to induced inputs and a variational bound.
* Power EP - [Paper](http://jmlr.org/papers/volume18/16-603/16-603.pdf)
  > Paper helps explain the full circle connection between all of the methods.

![alt text](pics/graphical_model_gp_inducing.png)

So an idea was proposed to use pseudo data where we augment the distribution of our data
$$p(f,u)=\mathcal{R}\left( 
    \begin{bmatrix}  
    f \\ u
    \end{bmatrix}; 
    \begin{bmatrix}
    0 \\ 0
    \end{bmatrix},
    \begin{bmatrix}
    K_{ff} & K_{fu} \\
    K_{uf} & K_{uu}
    \end{bmatrix} \right)$$

---
### Fully Independent Training Conditional (FITC) Approximation

**Resources**
* Zhenwen Dai - [GP2018](http://zhenwendai.github.io/slides/gpss2018_slides.pdf) | [Lecture](https://www.youtube.com/watch?list=PLFfvLE9TGnegjHFetV-zjPztaM_1UQk9B&v=GXEx6exAXrQ)
* James Hensman - [GPSS15](http://gpss.cc/gpss15/talks/talk_james.pdf) | [Blog](https://www.prowler.io/blog/sparse-gps-approximate-the-posterior-not-the-model)
  > Makes a good argument for approximating the inference instead of the model. Presentation breaks down the equations for inducing points fairly well.
* Matthias Bauer - [Understanding Prob. Sparse GPs Approx.](https://arxiv.org/pdf/1606.04820.pdf)
* GPSS 2014 - [Prezi](http://gpss.cc/gprs15a/assets/session4.pdf) | [Summer School](https://nbviewer.jupyter.org/github/gpschool/gprs15a/blob/master/index.ipynb)
  > A nice overview of the inducing points model, links power expectation algorithm.
* Implementations - [NeuGP](https://github.com/Alaya-in-Matrix/NeuGaP/blob/master/Model.py) | [PyGP](https://github.com/marionmari/pyGP_PR/blob/master/src/Core/inferences.py#L884)

This is one of the many number of methods people have used to approximate GPs. This method in particular seeks to do so by approximating the generative model and then uses exact inference. 

Keep in mind the notion of **factor graphs**.

1. Augment Model with $M<T$ pseudo data
   $$p(f,u)=\mathcal{R}\left( 
       \begin{bmatrix}  
       f \\ u
       \end{bmatrix}; 
       \begin{bmatrix}
       0 \\ 0
       \end{bmatrix},
       \begin{bmatrix}
       K_{ff} & K_{fu} \\
       K_{uf} & K_{uu}
       \end{bmatrix} \right)$$
2. Remove some of the dependencies (results in simpler models)
3. Calibrate the model (e.g. we use KL divergence below, but there are many other choices.)
   $$\underset{q(u), {q(f_t|u)^T_{t=1}}}{\text{argmin}}
   KL\left( p(f,u) ||q(u) \prod_{t=1}^Tq(f_t|u) \right)$$
   where:
   * $p(f,u)$ - GP prior over observed function values, f and inducing points, u
   * $q(u)\prod_{t=1}^Tq(f_t|u)$ - approximate model
   * $q(u)$ - correlated u's
   * $\prod_{t=1}^Tq(f_t|u)$ - conditionally independent f's given the u's.
  
   where we have the exact conditionals:
   * $q(u)=p(u)$
   * $q(f_t|u)=p(f_t|u)$

    Here we have an indirect an **indirect posterior approximation**. We construct our new generative model with pseudo-data. This allows us to cheaply perform exact learning and inference which is calibrated to the original model.


#### Factor Graphs

We can make the complex factor graph simpler by marginalizing out the functions which leaves us with just our inducing points linked to our observations. But we can write each of the layers of our factor graph accordingly to specify our new generative model:

First we have the correlated u's
$$q(u)=p(u)=\mathcal{N}(u;0, K_{uu})$$

Now look at the conditionally independent f's given the u's

$$
\begin{aligned}
q(f_t|u) &=p(f_t|u) \\
&=\mathcal{N}(y_1; K_{f_t u}K^{-1}, K_{f_tf_t}-K_{f_t u}K_{uu}^{-1}K_{uf_t})
\end{aligned}$$

Right away we can see that we save a LOT of computational time by inverting the matrix with the inducing points.

Let $D_{tt}=K_{f_tf_t}-K_{f_t u}K_{uu}^{-1}K_{uf_t}$ (some booking).

Lastly we need to write our observations, $y$ given the $f$'s:

$$q(y_t|f_t)=p(y_t|f_t)=\mathcal{N}(y_t;f_t,\sigma^2_y)$$

**Likelihood Computing Cost**: $\mathcal{O}(TM^2)$

Now we are looking at hyperparameter optimization. Everything above was Gaussian, so that means that the MLE kind of has to be Gaussian. In the end, we have the following formulation

$$
\begin{aligned}
p(y_t|\theta) &= \mathcal{N}(y;0, K_{fu}K_{uu}^{-1}K_{uu}K^{-1}_{uu}K_{uf} + D + \sigma_y^2I) \\
&= \mathcal{N}(y; 0, K_{fu}K_{uu}^{-1}K_{uf}+D+\sigma_y^2I
\end{aligned}$$

We want to find the $cov(y)$. We are just going to assume that this is Gaussian. Computing all of the moments. The first moment is zero becausee the mean is zero. So we're good. Now we need to compute the second moment which is the covariance (expected value of y's).

As usual, let $y=f+\sigma_y\epsilon$ and $\epsilon\sim(0, I)$. At length we have:

$$
\begin{aligned}
cov(y) &= \underset{q(u)q(f|u)}{\mathbb{E}(yy^T)} \\
&= \underset{u,f,\epsilon}{\mathbb{E}}\left[(f+\sigma_y \epsilon)(f+\sigma_y\epsilon)^T \right] \\
&= \underset{u,f}{\mathbb{E}}(ff^T) + \epsilon_y^2I 
\end{aligned}$$

So we've gotten the noise term for the $y$. Now for the other term. Let $f_t=K_{f_tu}K_{uu}^{-1}u+D_{tt}\epsilon'$ where $\epsilon \sim \mathcal{N}(0, 1)$.

$$\begin{aligned}
\mathbb{E}(ff^T) &= \mathbb{E}\left[ K_{f_tu}K_{uu}^{-1}uu^TK_{f_tu}^{-1}K_{uu} + D\right] \\
&= \mathbb{E}\left[ K_{f_tu}K_{uu}^{-1}K_{uu}K_{f_tu}^{-1}K_{uu} + D\right]
\end{aligned}$$

**Note**: The matrix $D$ has the original data which stops the variances from collapsing.

**Conclusion**
We are effectively fitting a clever parametric model to the GP. We've lost the nice separation of model, inference and approximation. So for example if we add more data, we might have to add more pseudo-points as well.

---
### Variational Free Energy Method (VFE)

**Resources**
* VFE for Sparse GPs - [Blog](https://www.uv.es/gonmagar/blog/2018/04/19/VariationalFreeEnergy)

Take the log probability of our data given our parameters.

$$\begin{aligned}
L(\theta) &= \log p(y|\theta)=\log \int df p(y,f|\theta) \\
&= 
\end{aligned}$$

I can use a [identity trick](https://www.shakirm.com/slides/MLSS2018-Madrid-ProbThinking.pdf) to multiply and divide by an arbitrary distribution



$$\begin{aligned}
L(\theta) &= \log p(y|\theta)=\log \int df p(y,f|\theta) \frac{q(f)}{q(f)}
\end{aligned}$$

We can use Jensen's inequality to turn take the log of an average instead of an average of a log.

$$\begin{aligned}
L(\theta) &= \log \int df p(y,f|\theta) \frac{q(f)}{q(f)} \leq 
\int df q(f) \log \frac{p(y,f|\theta)}{q(f)}
\end{aligned}$$

We can rewrite this in a slightly different way: the log of the joint dist. of $y,f$. We rewrite this as the product rule.

$$\begin{aligned}
\mathcal{F}(\theta) &= \int df q(f) \log \frac{p(f|y, \theta)p(y|\theta)}{q(f)}
\end{aligned}$$

Expanding the terms on the numerator of the log using the log sum rule, we get:

$$\begin{aligned}
\mathcal{F}(\theta) &= \int df q(f) \left( \log \frac{p(f|y, \theta)}{q(f)} + \log p(y|\theta)\right)
\end{aligned}$$

The second term does not depend of f so multiplying that term with the integral will just be 1.

$$\begin{aligned}
\mathcal{F}(\theta) &= \int df q(f) \log \frac{p(f|y, \theta)}{q(f)} + \log p(y|\theta)
\end{aligned}$$

Now we have the KL divergence for the first term and the likelihood for the second.


$$\begin{aligned}
\mathcal{F}(\theta) &= \log p(y|\theta) - KL\left[q(f)||p(f|y)\right]
\end{aligned}$$

If we optimize $\mathcal{F}$ with respect to $q(f)$, the KL is minimized and we just get the likelihood, but that's nothing new and we've done nothing useful. If we introduce some special structure in $q(f)$ by introducing sparsity, then we can achieve something useful with this formulation.

Looking at the infinite subspace of data points, let's restrict it to be $u$ that summarize our function data.

$$\begin{aligned}
q(f) &= q(u, f_{\neq u})
\end{aligned}$$

Using the product rule:

$$\begin{aligned}
q(f) &= q(f_{\neq u}|u)q(u)
\end{aligned}$$

Let's assume that $q(f_{\neq u}|u)=p(f_{\neq u}|y, u)$. All of the function values not at the U's are given by their GP prior. We only optimize our $q(u)$

$$\begin{aligned}
q(f) &= p(f_{\neq u}|u)q(u)
\end{aligned}$$



---
#### Full Proof -Variational Bound on $\mathcal P (y|u)$ (**Incomplete...**)

We've shown the difficulties of actually obtaining the probability density function of $\mathcal{P}(y)$ but in this section we're just going to show that we can obtain a lower bound for the conditional density function $\mathcal{P}(y|u)$

$$\mathcal{P}(y|u)=\int_f \mathcal P (y|f) \cdot \mathcal{P}(f|u) \cdot df$$

I'll do the 4.5 classic steps in order to arrive at a variational lower bound:

1. Take the $\log$ of both sides of the function.
   
$$\log \mathcal P (y|u) = \log \int_f \mathcal P (y|f) \cdot \mathcal{P}(f|u) \cdot df$$

2. Introduce the variational parameter $q(f)$ with the Identity trick.

$$\log \mathcal P (y|u) = \log \int_f \mathcal P (y|f) \cdot \mathcal{P}(f|u) \cdot \frac{q(f)}{q(f)} \cdot \frac{\mathcal{P}(f|y,u)}{\mathcal{P}(f|y,u)} \cdot df$$

$$\log \mathcal P (y|u) = \log \int_f \mathcal P (y|f) \cdot \mathcal{P}(f|u) \cdot \frac{q(f)}{q(f)} \cdot \frac{1}{\mathcal{P}(f|y,u)} \cdot \mathcal{P}(f|y,u)\cdot df$$

$$\log \mathcal P (y|u) = \log \int_f \mathcal P (y|f) \cdot \mathcal{P}(f|u) \cdot \frac{q(f)}{q(f)} \cdot \frac{1}{\mathcal{P}(f|y,u)} \cdot \frac{\mathcal P (y|f) \cdot \mathcal P (f|u)}{\mathcal P (y|u)}\cdot df$$

$$\log \mathcal P (y|u) = \log \int_f \left(  \frac{\mathcal P (y|f) \cdot \mathcal{P}(f|u)}{q(f)} \right)  \cdot \frac{q(f)}{} \cdot \frac{1}{\mathcal{P}(f|y,u)} \cdot \frac{\mathcal P (y|f) \cdot \mathcal P (f|u)}{\mathcal P (y|u)}\cdot df$$


3. Use Jensen's inequality for the log function to rearrange the formula and provide a bound for $\mathcal{F}(q)$:

$$\mathcal L () = \log \mathcal P (y|u) \geq  \int_f q(f)  \cdot \log \frac{\mathcal P (y|f) \cdot \mathcal{P}(f|u)}{q(f) } \cdot df = \mathcal F (q)$$

4. Rearrange to look like an expectation and KL divergence using targeted $\log$ rules:

$$\mathcal F (q) = \int_f q(f) \cdot \log \mathcal P(y|f) \cdot df - \int_f q(f) \cdot \log \frac{\mathcal{P}(f|u)}{q(f)} \cdot df$$

5. Simplify notation to look like every paper in ML that uses VI.

$$\mathcal F (q) = \mathbb E_{q(f)} \left[ \log \mathcal P(y|f) \right]  - \text{D}_{\text{KL}} \left[ q(f) || \mathcal{P}(f|u)\right]$$

#### Some Relations

$$\mathcal P (y,u,f) = \mathcal P (y|f) \cdot \mathcal P (f|u) \cdot \mathcal P (u)$$
$$\mathcal P (y,u,f) = \mathcal P (y,f|u) \cdot \mathcal P (u) = \mathcal P (f|y, u)  \cdot \mathcal P (y|u) \cdot \mathcal P (u)$$
$$\mathcal P (y|f) \cdot \mathcal P (f|u) \cdot \cancel{\mathcal P (u)} = \mathcal P (f|y, u)  \cdot \mathcal P (y|u) \cdot \cancel{\mathcal P (u)} $$
$$\mathcal P (y|f) \cdot \mathcal P (f|u) \cdot  = \mathcal P (f|y, u)  \cdot \mathcal P (y|u)  $$

$$\mathcal P (f|y, u) = \frac{\mathcal P (y|f) \cdot \mathcal P (f|u)}{\mathcal P (y|u)}$$