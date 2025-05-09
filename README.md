# **Gaussian Mixture Models, Expectation Maximization and KL Divergence**

## **Introduction**
This is an implementation of the Expectation Maximization algorithm which helps fitting a Gaussian Model to a set of data points. The algorithm is designed to estimate parameters of multiple Gaussian distributions that best fit the data. The algorithm is responsible for finding the best estimates for the parameters. At the same time, once the fitting is completed, the KL Divergence algorithm is a method that helps measure the error between the predicted fit and the actual curve. 

## **Proof of Correctness**
We wish to find a $\theta$ (Parameter Vector) such that $P(X|\theta)$ is maximized, i.e, Maximum Likelihood. Alternatively to simplify the calculation, we can maximize $$ \mathcal{L}(\theta) = \ln P(X \mid \theta) $$ This is known as the log maximum likelihood measure.

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
$$
\theta_{n + 1} = \argmax_{\theta} [ \mathcal{l}(\theta \mid \theta_n) ]
$$
$$
\implies \theta_{n + 1} = \argmax_{\theta} [\mathcal{L}(\theta_n) + \Delta(\theta \mid \theta_n)]
$$
Now, expand the terms inside the argmax and drop the non - $\theta$ terms.

$$
\theta_{n + 1} = argmax_{\theta} [\sum_Z P(Z \mid X, \theta_n) \ln P(X \mid Z, \theta) \cdot P(Z \mid \theta)]
$$

Now simplify $P(X \mid Z, \theta) \cdot P(Z \mid \theta)$ to $P(X, Z \mid \theta)$ as we know, E = $\sum$ P . So, 

$$
\theta_{n + 1} = argmax_{\theta} [E_{Z \mid X, \theta_n} (\ln P(X, Z \mid \theta))]
$$

Hence, this shows that the Expectation step leads to the maximization of $\mathcal{L}(\theta)$ on the $n + 1^{th}$ iteration. That is, on eac hiteration the likelihood is non - decreasing.
