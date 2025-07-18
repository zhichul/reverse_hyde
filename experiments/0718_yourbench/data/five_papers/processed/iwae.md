6
1
0
2

v
o
N
7

]

G
L
.
s
c
[

4
v
9
1
5
0
0
.
9
0
5
1
:
v
i
X
r
a

Under review as a conference paper at ICLR 2016

IMPORTANCE WEIGHTED AUTOENCODERS

Yuri Burda, Roger Grosse & Ruslan Salakhutdinov
Department of Computer Science
University of Toronto
Toronto, ON, Canada
{yburda,rgrosse,rsalakhu}@cs.toronto.edu

ABSTRACT

The variational autoencoder (VAE; Kingma & Welling (2014)) is a recently pro-
posed generative model pairing a top-down generative network with a bottom-up
recognition network which approximates posterior inference. It typically makes
strong assumptions about posterior inference, for instance that the posterior dis-
tribution is approximately factorial, and that its parameters can be approximated
with nonlinear regression from the observations. As we show empirically, the
VAE objective can lead to overly simpliﬁed representations which fail to use the
network’s entire modeling capacity. We present the importance weighted autoen-
coder (IWAE), a generative model with the same architecture as the VAE, but
which uses a strictly tighter log-likelihood lower bound derived from importance
weighting. In the IWAE, the recognition network uses multiple samples to ap-
proximate the posterior, giving it increased ﬂexibility to model complex posteri-
ors which do not ﬁt the VAE modeling assumptions. We show empirically that
IWAEs learn richer latent space representations than VAEs, leading to improved
test log-likelihood on density estimation benchmarks.

1

INTRODUCITON

In recent years, there has been a renewed focus on learning deep generative models (Hinton et al.,
2006; Salakhutdinov & E., 2009; Gregor et al., 2014; Kingma & Welling, 2014; Rezende et al.,
2014). A common difﬁculty faced by most approaches is the need to perform posterior inference
during training: the log-likelihood gradients for most latent variable models are deﬁned in terms
of posterior statistics (e.g. Salakhutdinov & E. (2009); Neal (1992); Gregor et al. (2014)). One
approach for dealing with this problem is to train a recognition network alongside the generative
model (Dayan et al., 1995). The recognition network aims to predict the posterior distribution over
latent variables given the observations, and can often generate a rough approximation much more
quickly than generic inference algorithms such as MCMC.

The variational autoencoder (VAE; Kingma & Welling (2014); Rezende et al. (2014)) is a recently
proposed generative model which pairs a top-down generative network with a bottom-up recognition
network. Both networks are jointly trained to maximize a variational lower bound on the data log-
likelihood. VAEs have recently been successful at separating style and content (Kingma et al., 2014;
Kulkarni et al., 2015) and at learning to “draw” images in a realistic manner (Gregor et al., 2015).

VAEs make strong assumptions about the posterior distribution. Typically VAE models assume that
the posterior is approximately factorial, and that its parameters can be predicted from the observables
through a nonlinear regression. Because they are trained to maximize a variational lower bound
on the log-likelihood, they are encouraged to learn representations where these assumptions are
satisﬁed, i.e. where the posterior is approximately factorial and predictable with a neural network.
While this effect is beneﬁcial, it comes at a cost: constraining the form of the posterior limits the
expressive power of the model. This is especially true of the VAE objective, which harshly penalizes
approximate posterior samples which are unlikely to explain the data, even if the recognition network
puts much of its probability mass on good explanations.

In this paper, we introduce the importance weighted autoencoder (IWAE), a generative model which
shares the VAE architecture, but which is trained with a tighter log-likelihood lower bound de-

1

Under review as a conference paper at ICLR 2016

rived from importance weighting. The recognition network generates multiple approximate pos-
terior samples, and their weights are averaged. As the number of samples is increased, the lower
bound approaches the true log-likelihood. The use of multiple samples gives the IWAE additional
ﬂexibility to learn generative models whose posterior distributions do not ﬁt the VAE modeling as-
sumptions. This approach is related to reweighted wake sleep (Bornschein & Bengio, 2015), but
the IWAE is trained using a single uniﬁed objective. Compared with the VAE, our IWAE is able to
learn richer representations with more latent dimensions, which translates into signiﬁcantly higher
log-likelihoods on density estimation benchmarks.

2 BACKGROUND

In this section, we review the variational autoencoder (VAE) model of Kingma & Welling (2014). In
particular, we describe a generalization of the architecture to multiple stochastic hidden layers. We
note, however, that Kingma & Welling (2014) used a single stochastic hidden layer, and there are
other sensible generalizations to multiple layers, such as the one presented by Rezende et al. (2014).

The VAE deﬁnes a generative process in terms of ancestral sampling through a cascade of hidden
layers:

p(x|θ) =

(cid:88)

p(hL|θ)p(hL−1|hL, θ) · · · p(x|h1, θ).

(1)

h1,...,hL

Here, θ is a vector of parameters of the variational autoencoder, and h = {h1, . . . , hL} denotes the
stochastic hidden units, or latent variables. The dependence on θ is often suppressed for clarity. For
convenience, we deﬁne h0 = x. Each of the terms p(h(cid:96)|h(cid:96)+1) may denote a complicated nonlinear
relationship, for instance one computed by a multilayer neural network. However, it is assumed
that sampling and probability evaluation are tractable for each p(h(cid:96)|h(cid:96)+1). Note that L denotes
the number of stochastic hidden layers; the deterministic layers are not shown explicitly here. We
assume the recognition model q(h|x) is deﬁned in terms of an analogous factorization:

q(h|x) = q(h1|x)q(h2|h1) · · · q(hL|hL−1),

(2)

where sampling and probability evaluation are tractable for each of the terms in the product.

In this work, we assume the same families of conditional probability distributions as Kingma &
Welling (2014). In particular, the prior p(hL) is ﬁxed to be a zero-mean, unit-variance Gaussian.
In general, each of the conditional distributions p(h(cid:96)| h(cid:96)+1) and q(h(cid:96)|h(cid:96)−1) is a Gaussian with
diagonal covariance, where the mean and covariance parameters are computed by a deterministic
feed-forward neural network. For real-valued observations, p(x|h1) is also deﬁned to be such a
Gaussian; for binary observations, it is deﬁned to be a Bernoulli distribution whose mean parameters
are computed by a neural network.

The VAE is trained to maximize a variational lower bound on the log-likelihood, as derived from
Jensen’s Inequality:

log p(x) = log Eq(h|x)

(cid:21)

(cid:20) p(x, h)
q(h|x)

≥ Eq(h|x)

(cid:20)

log

(cid:21)

p(x, h)
q(h|x)

= L(x).

(3)

Since L(x) = log p(x) − DKL(q(h|x)||p(h|x)), the training procedure is forced to trade off the
data log-likelihood log p(x) and the KL divergence from the true posterior. This is beneﬁcial, in that
it encourages the model to learn a representation where posterior inference is easy to approximate.

If one computes the log-likelihood gradient for the recognition network directly from Eqn. 3, the re-
sult is a REINFORCE-like update rule which trains slowly because it does not use the log-likelihood
Instead,
gradients with respect to latent variables (Dayan et al., 1995; Mnih & Gregor, 2014).
Kingma & Welling (2014) proposed a reparameterization of the recognition distribution in terms
of auxiliary variables with ﬁxed distributions, such that the samples from the recognition model are
a deterministic function of the inputs and auxiliary variables. While they presented the reparameter-
ization trick for a variety of distributions, for convenience we discuss the special case of Gaussians,
since that is all we require in this work. (The general reparameterization trick can be used with our
IWAE as well.)
In this paper, the recognition distribution q(h(cid:96)|h(cid:96)−1, θ) always takes the form of a Gaussian
N (h(cid:96)|µ(h(cid:96)−1, θ), Σ(h(cid:96)−1, θ)), whose mean and covariance are computed from the the states of

2

Under review as a conference paper at ICLR 2016

the hidden units at the previous layer and the model parameters. This can be alternatively expressed
by ﬁrst sampling an auxiliary variable (cid:15)(cid:96) ∼ N (0, I), and then applying the deterministic mapping
h(cid:96)((cid:15)(cid:96), h(cid:96)−1, θ) = Σ(h(cid:96)−1, θ)1/2(cid:15)(cid:96) + µ(h(cid:96)−1, θ).
(4)
The joint recognition distribution q(h|x, θ) over all latent variables can be expressed in terms of
a deterministic mapping h((cid:15), x, θ), with (cid:15) = ((cid:15)1, . . . , (cid:15)L), by applying Eqn. 4 for each layer in
sequence. Since the distribution of (cid:15) does not depend on θ, we can reformulate the gradient of the
bound L(x) from Eqn. 3 by pushing the gradient operator inside the expectation:
(cid:21)

(cid:20)

(cid:21)

∇θ log Eh∼q(h|x,θ)

(cid:20) p(x, h|θ)
q(h|x, θ)

= ∇θE(cid:15)1,...,(cid:15)L∼N (0,I)
(cid:20)
∇θ log

= E(cid:15)1,...,(cid:15)L∼N (0,I)

log

p(x, h((cid:15), x, θ)|θ)
q(h((cid:15), x, θ)|x, θ)
p(x, h((cid:15), x, θ)|θ)
q(h((cid:15), x, θ)|x, θ)

(5)

(6)

(cid:21)

.

Assuming the mapping h is represented as a deterministic feed-forward neural network, for a ﬁxed
(cid:15), the gradient inside the expectation can be computed using standard backpropagation. In practice,
one approximates the expectation in Eqn. 6 by generating k samples of (cid:15) and applying the Monte
Carlo estimator

1
k

k
(cid:88)

i=1

∇θ log w (x, h((cid:15)i, x, θ), θ)

(7)

with w(x, h, θ) = p(x, h|θ)/q(h|x, θ). This is an unbiased estimate of ∇θL(x). We note that
the VAE update and the basic REINFORCE-like update are both unbiased estimators of the same
gradient, but the VAE update tends to have lower variance in practice because it makes use of the
log-likelihood gradients with respect to the latent variables.

3

IMPORTANCE WEIGHTED AUTOENCODER

The VAE objective of Eqn. 3 heavily penalizes approximate posterior samples which fail to explain
the observations. This places a strong constraint on the model, since the variational assumptions
must be approximately satisﬁed in order to achieve a good lower bound. In particular, the posterior
distribution must be approximately factorial and predictable with a feed-forward neural network.
This VAE criterion may be too strict; a recognition network which places only a small fraction
(e.g. 20%) of its samples in the region of high posterior probability region may still be sufﬁcient for
performing accurate inference. If we lower our standards in this way, this may give us additional
ﬂexibility to train a generative network whose posterior distributions do not ﬁt the VAE assump-
tions. This is the motivation behind our proposed algorithm, the Importance Weighted Autoencoder
(IWAE).

Our IWAE uses the same architecture as the VAE, with both a generative network and a recognition
network. The difference is that it is trained to maximize a different lower bound on log p(x). In
particular, we use the following lower bound, corresponding to the k-sample importance weighting
estimate of the log-likelihood:

Lk(x) = Eh1,...,hk∼q(h|x)

log

(cid:34)

1
k

k
(cid:88)

i=1

p(x, hi)
q(hi|x)

(cid:35)

.

(8)

Here, h1, . . . , hk are sampled independently from the recognition model. The term inside the sum
corresponds to the unnormalized importance weights for the joint distribution, which we will denote
as wi = p(x, hi)/q(hi|x).

This is a lower bound on the marginal log-likelihood, as follows from Jensen’s Inequality and the
fact that the average importance weights are an unbiased estimator of p(x):
k
(cid:88)

k
(cid:88)

(cid:34)

(cid:34)

(cid:35)

(cid:35)

Lk = E

log

wi

≤ log E

wi

= log p(x),

(9)

1
k

i=1

1
k

i=1

where the expectations are with respect to q(h|x).

It is perhaps unintuitive that importance weighting would be a reasonable estimator in high dimen-
sions. Observe, however, that the special case of k = 1 is equivalent to the standard VAE objective
shown in Eqn. 3. Using more samples can only improve the tightness of the bound:

3

Under review as a conference paper at ICLR 2016

Theorem 1. For all k, the lower bounds satisfy

Moreover, if p(h, x)/q(h|x) is bounded, then Lk approaches log p(x) as k goes to inﬁnity.

log p(x) ≥ Lk+1 ≥ Lk.

(10)

Proof. See Appendix A.

The bound Lk can be estimated using the straightforward Monte Carlo estimator, where we generate
samples from the recognition network and average the importance weights. One might worry about
the variance of this estimator, since importance weighting famously suffers from extremely high
variance in cases where the proposal and target distributions are not a good match. However, as
our estimator is based on the log of the average importance weights, it does not suffer from high
variance. This argument is made more precise in Appendix B.

3.1 TRAINING PROCEDURE

To train an IWAE with a stochastic gradient based optimizer, we use an unbiased estimate of the
gradient of Lk, deﬁned in Eqn. 8. As with the VAE, we use the reparameterization trick to derive a
low-variance upate rule:

∇θLk(x) = ∇θEh1,...,hk

log

(cid:34)

(cid:35)

(cid:34)

wi

= ∇θE(cid:15)1,...,(cid:15)k

log

1
k

k
(cid:88)

i=1

(cid:34)

= E(cid:15)1,...,(cid:15)k

∇θ log

1
k

1
k

k
(cid:88)

w(x, h(x, (cid:15)i, θ), θ)

(11)

(cid:35)

i=1

k
(cid:88)

i=1

w(x, h(x, (cid:15)i, θ), θ)

(12)

(cid:35)

= E(cid:15)1,...,(cid:15)k

(cid:34) k

(cid:88)

i=1

(cid:102)wi∇θ log w(x, h(x, (cid:15)i, θ), θ)

,

(13)

(cid:35)

where (cid:15)1, . . . , (cid:15)k are the same auxiliary variables as deﬁned in Section 2 for the VAE, wi =
w(x, h(x, (cid:15)i, θ), θ) are the importance weights expressed as a deterministic function, and (cid:102)wi =
wi/ (cid:80)k
In the context of a gradient-based learning algorithm, we draw k samples from the recognition
network (or, equivalently, k sets of auxiliary variables), and use the Monte Carlo estimate of Eqn. 13:

i=1 wi are the normalized importance weights.

k
(cid:88)

i=1

(cid:102)wi∇θ log w (x, h((cid:15)i, x, θ), θ) .

(14)

In the special case of k = 1, the single normalized weight (cid:102)w1 takes the value 1, and one obtains the
VAE update rule.

We unpack this update because it does not quite parallel that of the standard VAE.1 The gradient of
the log weights decomposes as:

∇θ log w(x, h(x, (cid:15)i, θ), θ) = ∇θ log p(x, h(x, (cid:15)i, θ)|θ) − ∇θ log q(h(x, (cid:15)i, θ)|x, θ).

(15)

The ﬁrst term encourages the generative model to assign high probability to each h(cid:96) given h(cid:96)+1
(following the convention that x = h0). It also encourages the recognition network to adjust the
hidden representations so that the generative network makes better predictions. In the case of a single
stochastic layer (i.e. L = 1), the combination of these two effects is equivalent to backpropagation
in a stochastic autoencoder. The second term of this update encourages the recognition network to
have a spread-out distribution over predictions. This update is averaged over the samples with weight
proportional to the importance weights, motivating the name “importance weighted autoencoder.”

1Kingma & Welling (2014) separated out the KL divergence in the bound of Eqn. 3 in order to achieve a
simpler and lower-variance update. Unfortunately, no analogous trick applies for k > 1. In principle, the IWAE
updates may be higher variance for this reason. However, in our experiments, we observed that the performance
of the two update rules was indistinguishable in the case of k = 1.

4

Under review as a conference paper at ICLR 2016

The dominant computational cost in IWAE training is computing the activations and parameter gra-
dients needed for ∇θ log w(x, h(x, (cid:15)i, θ), θ). This corresponds to the forward and backward passes
in backpropagation. In the basic IWAE implementation, both passes must be done independently for
each of the k samples. Therefore, the number of operations scales linearly with k. In our GPU-based
implementation, the samples are processed in parallel by replicating each training example k times
within a mini-batch.

One can greatly reduce the computational cost by adding another form of stochasticity. Speciﬁcally,
only the forward pass is needed to compute the importance weights. The sum in Eqn. 14 can be
stochastically approximated by choosing a single sample (cid:15)i proprtional to its normalized weight (cid:102)wi
and then computing ∇θ log w(x, h(x, (cid:15)i, θ), θ). This method requires k forward passes and one
backward pass per training example. Since the backward pass requires roughly twice as many add-
multiply operations as the forward pass, for large k, this trick reduces the number of add-multiply
operations by roughly a factor of 3. This comes at the cost of increased variance in the updates, but
empirically we have found the tradeoff to be favorable.

4 RELATED WORK

There are several broad families of approaches to training deep generative models. Some models
are deﬁned in terms of Boltzmann distributions (Smolensky, 1986; Salakhutdinov & E., 2009). This
has the advantage that many of the conditional distributions are tractable, but the inability to sample
from the model or compute the partition function has been a major roadblock (Salakhutdinov &
Murray, 2008). Other models are deﬁned in terms of belief networks (Neal, 1992; Gregor et al.,
2014). These models are tractable to sample from, but the conditional distributions become tangled
due to the explaining away effect.

One strategy for dealing with intractable posterior inference is to train a recognition network
which approximates the posterior. A classic approach was the wake-sleep algorithm, used to train
Helmholtz machines (Dayan et al., 1995). The generative model was trained to model the condi-
tionals inferred by the recognition net, and the recognition net was trained to explain synthetic data
generated by the generative net. Unfortunately, wake-sleep trained the two networks on different ob-
jective functions. Deep autoregressive networks (Gregor et al., 2014) consisted of deep generative
and recognition networks trained using a single variational lower bound. Neural variational infer-
ence and learning (Mnih & Gregor, 2014) is another algorithm for training recognition networks
which reduces stochasticity in the updates by training a third network to predict reward baselines
in the context of the REINFORCE algorithm (Williams, 1992). Salakhutdinov & Larochelle (2010)
used a recognition network to approximate the posterior distribution in deep Boltzmann machines.

Variational autoencoders (Kingma & Welling, 2014; Rezende et al., 2014), as described in detail in
Section 2, are another combination of generative and recognition networks, trained with the same
variational objective as DARN and NVIL. However, in place of REINFORCE, they reduce the
variance of the updates through a clever reparameterization of the random choices. The reparame-
terization trick is also known as “backprop through a random number generator” (Williams, 1992).

One factor distinguishing VAEs from the other models described above is that the model is described
in terms of a simple distribution followed by a deterministic mapping, rather than a sequence of
stochastic choices. Similar architectures have been proposed which use different training objectives.
Generative adversarial networks (Goodfellow et al., 2014) train a generative network and a recog-
nition network which act in opposition: the recognition network attempts to distinguish between
training examples and generated samples, and the generative model tries to generate samples which
fool the recognition network. Maximum mean discrepancy (MMD) networks (Li et al., 2015; Dziu-
gaite et al., 2015) attempt to generate samples which match a certain set of statistics of the training
data. They can be viewed as a kind of adversarial net where the adversary simply looks at the set of
pre-chosen statistics (Dziugaite et al., 2015). In contrast to VAEs, the training criteria for adversarial
nets and MMD nets are not based on the data log-likelihood.

Other researchers have derived log-probability lower bounds by way of importance sampling. Tang
& Salakhutdinov (2013) and Ba et al. (2015) avoided recognition networks entirely, instead perform-
ing inference using importance sampling from the prior. Gogate et al. (2007) presented a variety
of graphical model inference algorithms based on importance weighting. Reweighted wake-sleep

5

Under review as a conference paper at ICLR 2016

(RWS) of Bornschein & Bengio (2015) is another recognition network approach which combines
the original wake-sleep algorithm with updates to the generative network equivalent to gradient as-
cent on our bound Lk. However, Bornschein & Bengio (2015) interpret this update as following a
biased estimate of ∇θ log p(x), whereas we interpret it as following an unbiased estimate of ∇θLk.
The IWAE also differs from RWS in that the generative and recognition networks are trained to
maximize a single objective, Lk. By contrast, the q-wake and sleep steps of RWS do not appear to
be related to Lk. Finally, the IWAE differs from RWS in that it makes use of the reparameterization
trick.

Apart from our approach of using multiple approximate posterior samples, another way to improve
the ﬂexibility of posterior inference is to use a more sophisticated algorithm than importance sam-
pling. Examples of this approach include normalizing ﬂows (Rezende & Mohamed, 2015) and the
Hamiltonian variational approximation of Salimans et al. (2015).

After the publication of this paper the authors learned that the idea of using an importance weighted
lower bound for training variational autoencoders has been independently explored by Laurent Dinh
and Vincent Dumoulin, and preliminary results of their work were presented at the 2014 CIFAR
NCAP Deep Learning summer school.

5 EXPERIMENTAL RESULTS

We have compared the generative performance of the VAE and IWAE in terms of their held-out log-
likelihoods on two density estimation benchmark datasets. We have further investigated a particular
issue we have observed with VAEs and IWAEs, namely that they learn latent spaces of signiﬁcantly
lower dimensionality than the modeling capacity they are allowed. We tested whether the IWAE
training method ameliorates this effect.

5.1 EVALUATION ON DENSITY ESTIMATION

We evaluated the models on two benchmark datasets: MNIST, a dataset of images of handwritten
digits (LeCun et al., 1998), and Omniglot, a dataset of handwritten characters in a variety of world
alphabets (Lake et al., 2013). In both cases, the observations were binarized 28 × 28 images.2 We
used the standard splits of MNIST into 60,000 training and 10,000 test examples, and of Omniglot
into 24,345 training and 8,070 test examples.

We trained models with two architectures:

1. An architecture with a single stochastic layer h1 with 50 units. In between the observations

and the stochastic layer were two deterministic layers, each with 200 units.

2. An architecture with two stochastic layers h1 and h2, with 100 and 50 units, respectively.
In between x and h1 were two deterministic layers with 200 units each. In between h1 and
h2 were two deterministic layers with 100 units each.

All deterministic hidden units used the tanh nonlinearity. All stochastic layers used Gaussian dis-
tributions with diagonal covariance, with the exception of the visible layer, which used Bernoulli
distributions. An exp nonlinearity was applied to the predicted variances of the Gaussian distribu-
tions. The network architectures are summarized in Appendix C.

All models were initialized with the heuristic of Glorot & Bengio (2010). For optimization, we used
Adam (Kingma & Ba, 2015) with parameters β1 = 0.9, β2 = 0.999, (cid:15) = 10−4 and minibaches of
size 20. The training proceeded for 3i passes over the data with learning rate of 0.001 · 10−i/7 for
i = 0 . . . 7 (for a total of (cid:80)7
i=0 3i = 3280 passes over the data). This learning rate schedule was
chosen based on preliminary experiments training a VAE with one stochastic layer on MNIST.

2Unfortunately, the generative modeling literature is inconsistent about the method of binarization, and
different choices can lead to considerably different log-likelihood values. We follow the procedure of Salakhut-
dinov & Murray (2008): the binary-valued observations are sampled with expectations equal to the real values
in the training set. See Appendix D for an alternative binarization scheme.

6

Under review as a conference paper at ICLR 2016

MNIST

OMNIGLOT

VAE

IWAE

VAE

IWAE

# stoch.
layers

1

2

k

1
5
50

1
5
50

NLL

86.76
86.47
86.35

85.33
85.01
84.78

active
units

19
20
20

16+5
17+5
17+5

NLL

86.76
85.54
84.78

85.33
83.89
82.90

active
units

19
22
25

16+5
21+5
26+7

NLL

108.11
107.62
107.80

107.58
106.31
106.30

active
units

28
28
28

28+4
30+5
30+5

NLL

108.11
106.12
104.67

107.56
104.79
103.38

active
units

28
34
41

30+5
38+6
44+7

Table 1: Results on density estimation and the number of active latent dimensions. For models with two latent
layers, “k1+k2” denotes k1 active units in the ﬁrst layer and k2 in the second layer. The generative performance
of IWAEs improved with increasing k, while that of VAEs beneﬁtted only slightly. Two-layer models achieved
better generative performance than one-layer models.

For each number of samples k ∈ {1, 5, 50} we trained a VAE with the gradient of L(x) estimted
as in Eqn. 7 and an IWAE with the gradient estimated as in Eqn. 14. For each k, the VAE and the
IWAE were trained for approximately the same length of time.

All log-likelihood values were estimated as the mean of L5000 on the test set. Hence, the reported
values are stochastic lower bounds on the true value, but are likely to be more accurate than the
lower bounds used for training.

The log-likelihood results are reported in Table 1. Our VAE results are comparable to those previ-
ously reported in the literature. We observe that training a VAE with k > 1 helped only slightly. By
contrast, using multiple samples improved the IWAE results considerably on both datasets. Note that
the two algorithms are identical for k = 1, so the results ought to match up to random variability.

On MNIST, IWAE with two stochastic layers and k = 50 achieves a log-likelihood of -82.90 on
the permutation-invariant model on this dataset. By comparison, deep belief networks achieved log-
likelihood of approximately -84.55 nats (Murray & Salakhutdinov, 2009), and deep autoregressive
networks achieved log-likelihood of -84.13 nats (Gregor et al., 2014). Gregor et al. (2015), who
exploited spatial structure, achieved a log-likelihood of -80.97. We did not ﬁnd overﬁtting to be a
serious issue for either the VAE or the IWAE: in both cases, the training log-likelihood was 0.62 to
0.79 nats higher than the test log-likelihood. We present samples from our models in Appendix E.

For the OMNIGLOT dataset, the best performing IWAE has log-likelihood of -103.38 nats, which is
slightly worse than the log-likelihood of -100.46 nats achieved by a Restricted Boltzmann Machine
with 500 hidden units trained with persistent contrastive divergence (Burda et al., 2015). RBMs
trained with centering or FANG methods achieve a similar performance of around -100 nats (Grosse
& Salakhudinov, 2015). The training log-likelihood for the models we trained was 2.39 to 2.65 nats
higher than the test log-likelihood.

5.2 LATENT SPACE REPRESENTATION

We have observed that both VAEs and IWAEs tend to learn latent representations with effective
dimensions far below their capacity. Our next set of experiments aimed to quantify this effect and
determine whether the IWAE objective ameliorates this effect.

If a latent dimension encodes useful information about the data, we would expect its distribution
to change depending on the observations. Based on this intuition, we measured activity of a latent
(cid:0)Eu∼q(u|x)[u](cid:1). We deﬁned the dimension u to be active
dimension u using the statistic Au = Covx
if Au > 10−2. We have observed two pieces of evidence that this criterion is both well-deﬁned and
meaningful:

1. The distribution of Au for a trained model consisted of two widely separated modes, as

shown in Appendix C.

7

Under review as a conference paper at ICLR 2016

First stage

Second stage

trained as

NLL

active units

trained as

NLL

active units

Experiment 1

VAE

86.76

Experiment 2

IWAE, k = 50

84.78

19

25

IWAE, k = 50

84.88

VAE

86.02

22

23

Table 2: Results of continuing to train a VAE model with the IWAE objective, and vice versa. Training the
VAE with the IWAE objective increased the latent dimension and test log-likelihood, while training the IWAE
with the VAE objective had the opposite effect.

2. To conﬁrm that the inactive dimensions were indeed insigniﬁcant to the predictions, we
evaluated all models with the inactive dimensions removed. In all cases, this changed the
test log-likelihood by less than 0.06 nats.

In Table 1, we report the numbers of active units for all conditions. In all conditions, the number of
active dimensions was far less than the total number of dimensions. Adding more latent dimensions
did not increase the number of active dimensions. Interestingly, in the two-layer models, the second
layer used very little of its modeling capacity: the number of active dimensions was always less
than 10. In all cases with k > 1, the IWAE learned more latent dimensions than the VAE. Since this
coincided with higher log-likelihood values, we speculate that a larger number of active dimensions
reﬂects a richer latent representation.

Superﬁcially, the phenomenon of inactive dimensions appears similar to the problem of “units dying
out” in neural networks and latent variable models, an effect which is often ascribed to difﬁculties
in optimization. For example, if a unit is inactive, it may never receive a meaningful gradient signal
because of a plateau in the optimization landscape. In such cases, the problem may be avoided
through a better initialization. To determine whether the inactive units resulted from an optimization
issue or a modeling issue, we took the best-performing VAE and IWAE models from Table 1, and
continued training the VAE model using the IWAE objective and vice versa. In both cases, the model
was trained for an additional 37 passes over the data with a learning rate of 10−4.

The results are shown in Table 2. We found that continuing to train the VAE with the IWAE objective
increased the number of active dimensions and the test log-likelihood, while continuing to train
the IWAE with the VAE objective did the opposite. The fact that training with the VAE objective
actively reduces both the number of active dimensions and the log-likelihood strongly suggests that
inactivation of the latent dimensions is driven by the objective functions rather than by optimization
issues. On the other hand, optimization also appears to play a role, as the results in Table 2 are not
quite identical to those in Table 1.

6 CONCLUSION

In this paper, we presented the importance weighted autoencoder, a variant on the VAE trained by
maximizing a tighter log-likelihood lower bound derived from importance weighting. We showed
empirically that IWAEs learn richer latent representations and achieve better generative performance
than VAEs with equivalent architectures and training time. We believe this method may improve the
ﬂexibility of other generative models currently trained with the VAE objective.

7 ACKNOWLEDGEMENTS

This research was supported by NSERC, the Fields Institute, and Samsung.

REFERENCES
Ba, J. L., Mnih, V., and Kavukcuoglu, K. Multiple object recognition with visual attention. In International

Conference on Learning Representations, 2015.

Bornschein, J. and Bengio, Y. Reweighted wake-sleep. International Conference on Learning Representations,

2015.

Burda, Y., Grosse, R. B., and Salakhutdinov, R. Accurate and conservative estimates of MRF log-likelihood

using reverse annealing. Artiﬁcial Intelligence and Statistics, pp. 102–110, 2015.

8

Under review as a conference paper at ICLR 2016

Dayan, P., Hinton, G. E., Neal, R. M., and Zemel, R. S. The Helmholtz machine. Neural Computation, 7:

889–904, 1995.

Dziugaite, K. G., Roy, D. M., and Ghahramani, Z. Training generative neural networks via maximum mean

discrepancy optimization. In Uncertainty in Artiﬁcial Intelligence, 2015.

Glorot, X. and Bengio, Y. Understanding the difﬁculty of training deep feedforward neural networks.

In

Artiﬁcial Intelligence and Statistics, pp. 249–256, 2010.

Gogate, V., Bidyuk, B., and Dechter, R. Studies in lower bounding probability of evidence using the Markov

inequality. In Uncertainty in Artiﬁcial Intelligence, 2007.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio,

Y. Generative adversarial nets. In Neural Information Processing Systems, 2014.

Gregor, K., Danihelka, I., Mnih, A., Blundell, C., and Wierstra, D. Deep autoregressive networks. International

Conference on Machine Learning, 2014.

Gregor, K., Danihelka, I., Graves, A., Rezende, D. J., and Wierstra, D. DRAW: A recurrent neural network for

image generation. In International Conference on Machine Learning, pp. 1462–1471, 2015.

Grosse, R. and Salakhudinov, R. Scaling up natural gradient by sparsely factorizing the inverse ﬁsher matrix.

In International Conference on Machine Learning, 2015.

Hinton, G. E., Osindero, S., and Teh, Y. A fast learning algorithm for deep belief nets. Neural Computation,

2006.

Kingma, D. and Ba, J. L. Adam: A method for stochastic optimization.

In International Conference on

Learning Representations, 2015.

Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes.

International Conference on Learning

Representations, 2014.

Kingma, D. P., Mohamed, S., Rezende, D. J., and Welling, M. Semi-supervised learning with deep generative

models. In Neural Information Processing Systems, 2014.

Kulkarni, T. D., Whitney, W., Kohli, P., and Tenenbaum, J. B. Deep convolutional inverse graphics network.

arXiv:1503.03167, 2015.

Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. One-shot learning by inverting a compositional causal

process. In Neural Information Processing Systems, 2013.

Larochelle, H., Murray I. The neural autoregressive distribution estimator. In Artiﬁcial Intelligence and Statis-

tics, pp. 29–37, 2011.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition.

Proceedings of the IEEE, 86(11):2278–2324, 1998.

Li, Y., Swersky, K., and Zemel, R. Generative moment matching networks. In International Conference on

Machine Learning, pp. 1718–1727, 2015.

Mnih, A. and Gregor, K. Neural variational inference and learning in belief networks. In International Confer-

ence on Machine Learning, pp. 1791–1799, 2014.

Murray, I. and Salakhutdinov, R. Evaluating probabilities under high-dimensional latent variable models. In

Neural Information Processing Systems, pp. 1137–1144, 2009.

Neal, R. M. Connectionist learning of belief networks. Artiﬁcial Intelligence, 1992.

Rezende, D. J. and Mohamed, S. Variational inference with normalizing ﬂows. In International Conference on

Machine Learning, pp. 1530–1538, 2015.

Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep

generative models. International Conference on Machine Learning, pp. 1278–1286, 2014.

Salakhutdinov, R. and E., Hinton G. Deep Boltzmann machines. In Neural Information Processing Systems,

2009.

Salakhutdinov, R. and Larochelle, H. Efﬁcient learning of deep Boltzmann machines. In Artiﬁcial Intelligence

and Statistics, 2010.

9

Under review as a conference paper at ICLR 2016

Salakhutdinov, R. and Murray, I. On the quantitative analysis of deep belief networks. In International Con-

ference on Machine Learning, 2008.

Salimans, T., Kingma, D. P., and Welling, M. Markov chain Monte Carlo and variational inference: bridging

the gap. In International Conference on Machine Learning, pp. 1218–1226, 2015.

Smolensky, P. Information processing in dynamical systems: foundations of harmony theory. In Rumelhart,
D. E. and McClelland, J. L. (eds.), Parallel Distributed Processing: Explorations in the Microstructure of
Cognition. MIT Press, 1986.

Tang, Y. and Salakhutdinov, R. Learning stochastic feedforward neural networks.

In Neural Information

Processing Systems, 2013.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Ma-

chine Learning, 8:229–256, 1992.

APPENDIX A

Proof of Theorem 1. We need to show the following facts about the log-likelihood lower bound Lk:

1. log p(x) ≥ Lk,

2. Lk ≥ Lm for k ≥ m,

3. log p(x) = limk→∞ Lk, assuming p(h, x)/q(h|x) is bounded.

We prove each in turn:

1. It follows from Jensen’s inequality that

(cid:34)

Lk = E

log

(cid:35)

1
k

k
(cid:88)

i=1

p(x, hi)
q(hi|x)

≤ log E

(cid:34)

1
k

k
(cid:88)

i=1

(cid:35)

p(x, hi)
q(hi|x)

= log p(x)

(16)

2. Let I ⊂ {1, . . . , k} with |I| = m be a uniformly distributed subset of distinct indices from
=

{1, . . . , k}. We will use the following simple observation: EI={i1,...,im}
a1+...+ak
k

for any sequence of numbers a1, . . . , ak.

(cid:104) ai1 +...+aim
m

(cid:105)

Using this observation and Jensen’s inequality, we get

(cid:34)

Lk = Eh1,...,hk

log

1
k

k
(cid:88)

i=1

(cid:35)

p(x, hi)
q(hi|x)


= Eh1,...,hk


log EI={i1,...,im}



≥ Eh1,...,hk


EI={i1,...,im}



log

1
m

1
m

m
(cid:88)

j=1

m
(cid:88)

j=1

p(x, hij )
q(hij |x)

p(x, hij )
q(hij |x)

















(cid:34)

= Eh1,...,hm

log

(cid:35)

1
m

m
(cid:88)

i=1

p(x, hi)
q(hi|x)

= Lm

(17)

(18)

(19)

(20)

3. Consider the random variable Mk = 1
k
it follows from the strong law of large numbers that Mk converges to Eq(hi|x)
p(x) almost surely. Hence Lk = E log[Mk] converges to log p(x) as k → ∞.

p(x,hi)
q(hi|x) . If p(h, x)/q(h|x) is bounded, then
=

(cid:104) p(x,hi)
q(hi|x)

i=1

(cid:105)

(cid:80)k

10

Under review as a conference paper at ICLR 2016

APPENDIX B

It is well known that the variance of an unnormalized importance sampling based estimator can
be extremely large, or even inﬁnite, if the proposal distribution is not well matched to the target
distribution. Here we argue that the Monte Carlo estimator of Lk, described in Section 3, does not
suffer from large variance. More precisely, we bound the mean absolute deviation (MAD). While
this does not directly bound the variance, it would be surprising if an estimator had small MAD yet
extremely large variance.
Suppose we have a strictly positive unbiased estimator ˆZ of a positive quantity Z, and we wish to
use log ˆZ as an estimator of log Z. By Jensen’s inequality, this is a biased estimator, i.e. E[log ˆZ] ≤
log Z. Denote the bias as δ = log Z − E[log ˆZ]. We start with the observation that log ˆZ is unlikely
to overestimate log Z by very much, as can be shown with Markov’s Inequality:

Pr(log ˆZ > log Z + b) ≤ e−b.

Let (X)+ denote max(X, 0). We now use the above facts to bound the MAD:

E

(cid:104)(cid:12)
(cid:12)log ˆZ − E[log ˆZ]
(cid:12)

(cid:12)
(cid:12)
(cid:12)

(cid:105)

= 2E

= 2E

≤ 2E

= 2E

(cid:20)(cid:16)

(cid:20)(cid:16)

(cid:20)(cid:16)

(cid:20)(cid:16)

(cid:17)
log ˆZ − E[log ˆZ]

(cid:21)

+

log ˆZ − log Z + log Z − E[log ˆZ]

(cid:21)

(cid:17)

+

log ˆZ − log Z

log ˆZ − log Z

(cid:17)

(cid:17)

+

+

(cid:16)

(cid:17)
log Z − E[log ˆZ]

+

(cid:21)

+

(cid:21)

+ 2δ

= 2

≤ 2

(cid:90) ∞

0
(cid:90) ∞

0

(cid:16)
log ˆZ − log Z > t

(cid:17)

Pr

dt + 2δ

e−tdt + 2δ

(21)

(22)

(23)

(24)

(25)

(26)

(27)

= 2 + 2δ
Here, (22) is a general formula for the MAD, (26) uses the formula E[Y ] = (cid:82) ∞
0 Pr(Y > t) dt for
a nonnegative random variable Y , and (27) applies the bound (21). Hence, the MAD is bounded by
2 + 2δ. In the context of IWAE, δ corresponds to the gap between Lk and log p(x).

(28)

APPENDIX C

NETWORK ARCHITECTURES

Here is a summary of the network architectures used in the experiments:

q(h1|x) = N (h1|µq,1, diag(σq,1))

x

lin+tanh

200d

lin+tanh

200d

lin

lin+exp

q(h2|h1) = N (h2|µq,2, diag(σq,2))

h1

lin+tanh

100d

lin+tanh

100d

lin

lin+exp

p(h1|h2) = N (h1|µp,1, diag(σp,1))

h2

lin+tanh

100d

lin+tanh

100d

lin

lin+exp

µq,1
σq,1

µq,2
σq,2

µp,1
σp,1

p(x|h1) = Bernoulli(x|µp,0)

lin+tanh

h1

200d

lin+tanh

200d

lin+sigm

µp,0

11

Under review as a conference paper at ICLR 2016

DISTRIBUTION OF ACTIVITY STATISTIC

(cid:0)Eu∼q(u|x)[u](cid:1), and chose a threshold
In Section 5.2, we deﬁned the activity statistic Au = Covx
of 10−2 for determining if a unit is active. One justiﬁcation for this is that the distribution of this
statistic consisted of two widely separated modes in every case we looked at. Here is the histogram
of log Au for a VAE with one stochastic layer:

VISUALIZATION OF POSTERIOR DISTRIBUTIONS

We show some examples of true and approximate posteriors for VAE and IWAE models trained with
two latent dimensions. Heat maps show true posterior distributions for 6 training examples, and the
pictures in the bottom row show the examples and their reconstruction from samples from q(h|x).
Left: VAE. Middle: IWAE, with k = 5. Right: IWAE, with k = 50. The IWAE prefers less regular
posteriors and more spread out posterior predictions.

12

8765432101log variance of µ024681012number of unitsUnder review as a conference paper at ICLR 2016

APPENDIX D

RESULTS FOR A FIXED MNIST BINARIZATION

Several previous works have used a ﬁxed binarization of the MNIST dataset deﬁned by Larochelle
(2011). We repeated our experiments training the models on the 50000 examples from the training
dataset, and evaluating them on the 10000 examples from the test dataset. Otherwise we used the
same training procedure and hyperparameters as in the experiments in the main part of the paper.
The results in table 3 indicate that the conclusions about the relative merits of VAEs and IWAEs are
unchanged in the new experimental setup. In this setup we noticed signiﬁcantly larger amounts of
overﬁtting.

VAE

IWAE

# stoch.
layers

1

2

k

1
5
50

1
5
50

NLL

88.71
88.83
89.05

88.08
87.63
87.86

active
units

19
19
20

16+5
17+5
17+6

NLL

88.71
87.63
87.10

88.08
86.17
85.32

active
units

19
22
24

16+5
21+5
24+7

Table 3: Results on density estimation and the number of active latent dimensions on the ﬁxed binarization
MNIST dataset. For models with two latent layers, “k1 + k2” denotes k1 active units in the ﬁrst layer and k2
in the second layer. The generative performance of IWAEs improved with increasing k, while that of VAEs
beneﬁtted only slightly. Two-layer models achieved better generative performance than one-layer models.

APPENDIX E

SAMPLES

13

Under review as a conference paper at ICLR 2016

Table 4: Random samples from VAE (left column) and IWAE with k = 50 (right column) models. Row 1:
models with one stochastic layer. Row 2: models with two stochastic layers. Samples are represented as the
means of the corresponding Bernoulli distributions.

14

