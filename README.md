# **Gaussian Mixture Models, Expectation Maximization and KL Divergence**

## **Introduction**
This is an implementation of the Expectation Maximization algorithm which helps fitting a Gaussian Model to a set of data points. The algorithm is designed to estimate parameters of multiple Gaussian distributions that best fit the data. The algorithm is responsible for finding the best estimates for the parameters. At the same time, once the fitting is completed, the KL Divergence algorithm is a method that helps measure the error between the predicted fit and the actual curve. 

## **Why Expectation Maximization**
The formula for the log likelihood of the Gaussian Mixture Model can be derived as follows :

Gaussian Mixture Distribution : 

$$ p(x) = \sum_k \pi_k \cdot \mathcal{N}(X \mid \mu_k, \Sigma_k)$$

**Proof:**\
Assume $z$ is a latent random variable such that  

$$p(z_k = 1) = \pi_k$$

i.e, The probability of the component chosen being $k$ is equal to $\pi_k$, where 

$$0 \leq \pi_k \leq 1$$ 

and

$$\sum_{k=1}^{K} \pi_k = 1$$

Furthermore, the conditional distribution of $X$ given a particular value for $z$ is a Gaussian :

$$ p(X \mid z_k = 1) = \mathcal{N}(X \mid \mu_k, \Sigma_k)$$

Which is the same as saying, given that the datapoint originates from the kth gaussian, it will be distributed as a gaussian.

Now, 

$$p(X) = \sum_z p(z) \cdot p(x \mid z)$$

$$\implies p(X) = \sum_{k = 1}^K \pi_k \cdot \mathcal{N} (X \mid \mu_k, \Sigma_k)$$

Moving to the need for need for EM Algorithm, Let us look at the calculation of the Log Likelihood function :

$$ \mathcal{L}(\theta) = \prod p(X_n)$$
$$\log \mathcal{L}(\theta) = \sum \log p(X_n)$$
So, 
$$ \log \mathcal{L}(\theta) = \sum_{n = 1}^N \log \sum_{k = 1}^K \pi_k \cdot \mathcal{N} (X_n \mid \mu_k, \Sigma_k)$$

Here, the problem that comes up due to the presence of the summation over $k$ that is present within the logarithm. This prevents the log function acting directly over the Gaussian. So, if we set the derivatives as $0$ we will no longer obtain a closed form solution, which is undesirable.

Let us look at the conditions that must be satisfied at the max of the likelihood function, i.e, computing $\frac{\partial\mathcal{L}(\theta)}{\partial\mu_k}$ :


Before anything else, the probability density function of a multivariate normal function [ $\mathcal{N}(X \mid \theta) $ ] is:

$$
\mathcal{N}(X_n \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (X_n - \mu_k)^T \Sigma_k^{-1} (X_n - \mu_k) \right)
$$

Where:
- $X_n \in \mathbb{R}^d$ is a data point,
- $\mu_k \in \mathbb{R}^d$ is the mean of component $ k $,
- $\Sigma_k \in \mathbb{R}^{d \times d}$ is the covariance matrix (positive definite).

Let:
$$ 
Z = \mathcal{N}(X_n \mid \mu_k, \Sigma_k) 
$$

$$ 
Q = (X_n - \mu_k)^T \Sigma_k^{-1} (X_n - \mu_k) 
$$

$$ 
C = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} 
$$

Then:
$$
Z = C \cdot \exp\left(-\frac{1}{2} Q\right)
$$


$$
\frac{\partial Z}{\partial \mu_k} 
= \frac{\partial}{\partial \mu_k} \left( C \cdot \exp\left( -\frac{1}{2} Q \right) \right)
= Z \cdot \left( -\frac{1}{2} \cdot \frac{\partial Q}{\partial \mu_k} \right)
$$



Let $a = X_n - \mu_k$, then:

$$
Q = a^T \Sigma_k^{-1} a
$$

So, 

$$
\frac{\partial Q}{\partial \mu_k} 
= \frac{\partial}{\partial \mu_k} \left( (X_n - \mu_k)^T \Sigma_k^{-1} (X_n - \mu_k) \right)
= -2 \Sigma_k^{-1} (X_n - \mu_k)
$$

Finally, 

Substitute back into the expression for $\frac{\partial Z}{\partial \mu_k}$

$$
\frac{\partial}{\partial \mu_k} \mathcal{N}(X_n \mid \mu_k, \Sigma_k)
= Z \cdot \left( -\frac{1}{2} \cdot (-2 \Sigma_k^{-1} (X_n - \mu_k)) \right)
= \mathcal{N}(X_n \mid \mu_k, \Sigma_k) \cdot \Sigma_k^{-1} (X_n - \mu_k)
$$

Hence, 

$$
\boxed{
\frac{\partial}{\partial \mu_k} \mathcal{N}(X_n \mid \mu_k, \Sigma_k) = \mathcal{N}(X_n \mid \mu_k, \Sigma_k) \cdot \Sigma_k^{-1} (X_n - \mu_k)
}
$$


Using this result, we can now solve for our required result :


$$
\log L(\theta) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \cdot \mathcal{N}(X_n \mid \mu_k, \Sigma_k) \right)
$$

Let:

$$
\gamma_{nk} = \frac{\pi_k \mathcal{N}(X_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(X_n \mid \mu_j, \Sigma_j)}
$$

This is the **responsibility** of component $k$ for data point $X_n$.

$$
\frac{\partial \log L(\theta)}{\partial \mu_k} = \sum_{n=1}^N \frac{\partial}{\partial \mu_k} \log \left( \sum_{j=1}^K \pi_j \mathcal{N}(X_n \mid \mu_j, \Sigma_j) \right)
$$

Since only the $k$-th term depends on $\mu_k$, apply the chain rule:

$$
= \sum_{n=1}^N \frac{1}{\sum_{j=1}^K \pi_j \mathcal{N}(X_n \mid \mu_j, \Sigma_j)} \cdot \pi_k \frac{\partial \mathcal{N}(X_n \mid \mu_k, \Sigma_k)}{\partial \mu_k}
$$

From above :

$$
\frac{\partial}{\partial \mu_k} \mathcal{N}(X_n \mid \mu_k, \Sigma_k) = \mathcal{N}(X_n \mid \mu_k, \Sigma_k) \cdot \Sigma_k^{-1} (X_n - \mu_k)
$$

Substitute this into the previous expression:

$$
\frac{\partial \log L(\theta)}{\partial \mu_k} 
= \sum_{n=1}^N \frac{\pi_k \mathcal{N}(X_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(X_n \mid \mu_j, \Sigma_j)} \cdot \Sigma_k^{-1} (X_n - \mu_k)
$$

Hence, 

$$
\boxed{
\frac{\partial \log L(\theta)}{\partial \mu_k} 
= \sum_{n=1}^N \gamma_{nk} \Sigma_k^{-1} (X_n - \mu_k)
}
$$

And, for maximization, we equate the same to zero :

$$
0 = \sum_{n=1}^N \gamma_{nk} \Sigma_k^{-1} (X_n - \mu_k)
$$
}


**NOTE:** This is the gradient used in the **M-step** of the EM algorithm for updating $\mu_k$.


%% ####################################################################################################################################################################### %%

## **Proof of Correctness**
We wish to find a $\theta$ (Parameter Vector) such that $P(X|\theta)$ is maximized, i.e, Maximum Likelihood. Alternatively to simplify the calculation, we can maximize 

$$ \mathcal{L}(\theta) = \ln P(X \mid \theta) $$ 

This is known as the log maximum likelihood measure.

Now, as $\ln(x)$ is an increasing function, the value of $\theta$ that maximizes $P(X|\theta)$ will also maximize $\mathcal{L}(\theta)$.
Assume that after the $n^{th}$ iteration, the estimate is $\theta_n$ and this is updated to become $theta$ after the algorithm.

So, our task would be to prove that $$\mathcal{L}(\theta) > \mathcal{L}(\theta_n)$$ i.e, $$\ln P(X \mid \theta) > \ln P(X \mid \theta_n)$$

Given that $Z$ is a latent variable:

$$
P(X \mid \theta) = \sum_Z P(X \mid Z, \theta) \cdot P(Z \mid \theta)
$$

Then,

$$
\mathcal{L}(\theta) - \mathcal{L}(\theta_n) = \ln \sum_Z P(X \mid Z, \theta) \cdot P(Z \mid \theta) - \ln P(X \mid \theta_n)
$$

$$
\mathcal{L}(\theta) - \mathcal{L}(\theta_n) = \ln \sum_Z P(X \mid Z, \theta) \cdot P(Z \mid \theta) \cdot \frac{P(Z \mid X, \theta_n)}{P(Z \mid X, \theta_n)} - \ln P(X \mid \theta_n)                              
$$
Rearranging the above, we get :
$$
\mathcal{L}(\theta) - \mathcal{L}(\theta_n) = \ln \sum_Z P(Z \mid X, \theta_n) \cdot \frac{P(X \mid Z, \theta) \cdot P(Z \mid \theta)}{P(Z \mid X, \theta_n)} - \ln P(X \mid \theta_n)  
$$

Now, applying Jensen's Inequality on the above equation, we get :

$$\ln \sum_Z P(Z \mid X, \theta_n) \cdot \frac{P(X \mid Z, \theta) \cdot P(Z \mid \theta)}{P(Z \mid X, \theta_n)} - \ln P(X \mid \theta_n) \geq \sum_Z P(Z \mid X, \theta_n) \ln \frac{P(X \mid Z, \theta) \cdot P(Z \mid \theta)}{P(Z \mid X, \theta_n)} - \ln P(X \mid \theta_n)$$


Now, $$-\ln P(X \mid \theta_n) > - \sum \ln P(X \mid \theta_n)$$ so we can replace it and then take $-\ln P(X \mid \theta_n)$ into the summation.


So,

$$
 \mathcal{L}(\theta) - \mathcal{L}(\theta_n) \geq \sum_Z P(Z \mid X, \theta_n) \cdot \ln \frac{P(X \mid Z, \theta) \cdot P(Z \mid \theta)}{P(Z \mid X, \theta_n) \cdot P(X \mid \theta_n)} 
$$

Now, Let, 

$$ 
\Delta(\theta \mid \theta_n) = \sum_Z P(Z \mid X, \theta_n) \cdot \ln \frac{P(X \mid Z, \theta) \cdot P(Z \mid \theta)}{P(Z \mid X, \theta_n) \cdot P(X \mid \theta_n)}  
$$

Also, Let, 

$$
\mathcal{l}(\theta \mid \theta_n) = \mathcal{L}(\theta_n) + \Delta(\theta \mid \theta_n)
$$

$$
\implies \mathcal{L}(\theta)  \geq  \mathcal{l}(\theta \mid \theta_n)
$$

So, we can say that $\mathcal{l}(\theta \mid \theta_n)$ is a lowerbound to $\mathcal{L}(\theta)$.

So, any $\theta$ which increases $\mathcal{l}(\theta \mid \theta_n)$ would lead to the increase of $\mathcal{L}(\theta)$.

$\therefore$ To maximize $\mathcal{L}(\theta)$, we can maximize $\mathcal{l}(\theta \mid \theta_n)$. This is more helpful as $\mathcal{l}(\theta \mid \theta_n)$ .

Finally,

$$\theta_{n + 1} = argmax_{\theta} [ \mathcal{l}(\theta \mid \theta_n) ]$$

$$\implies \theta_{n + 1} = argmax_{\theta} [\mathcal{L}(\theta_n) + \Delta(\theta \mid \theta_n)]$$

Now, expand the terms inside the argmax and drop the non - $\theta$ terms.

$$
\theta_{n + 1} = argmax_{\theta} [\sum_Z P(Z \mid X, \theta_n) \ln P(X \mid Z, \theta) \cdot P(Z \mid \theta)]
$$

Now simplify $P(X \mid Z, \theta) \cdot P(Z \mid \theta)$ to $P(X, Z \mid \theta)$ as we know, E = $\sum$ P . So, 

$$
\theta_{n + 1} = argmax_{\theta} [E_{Z \mid X, \theta_n} (\ln P(X, Z \mid \theta))]
$$

Hence, this shows that the Expectation step leads to the maximization of $\mathcal{L}(\theta)$ on the $n + 1^{th}$ iteration. That is, on eac hiteration the likelihood is non - decreasing.
