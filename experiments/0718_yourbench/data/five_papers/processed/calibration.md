4
2
0
2

n
u
J

4

]

G
L
.
s
c
[

2
v
4
7
4
0
0
.
4
0
4
2
:
v
i
X
r
a

Linguistic Calibration of Long-Form Generations

Neil Band, Xuechen Li, Tengyu Ma, Tatsunori Hashimoto
Stanford University
{nband, lxuechen, tengyuma, thashim}@cs.stanford.edu

Abstract

Language models (LMs) may lead their users to make suboptimal downstream decisions when they
confidently hallucinate. This issue can be mitigated by having the LM verbally convey the probability that its
claims are correct, but existing models cannot produce long-form text with calibrated confidence statements.
Through the lens of decision-making, we define linguistic calibration for long-form generations: an LM
is linguistically calibrated if its generations enable its users to make calibrated probabilistic predictions.
This definition enables a training framework where a supervised finetuning step bootstraps an LM to emit
long-form generations with confidence statements such as “I estimate a 30% chance of...” or “I am certain
that...”, followed by a reinforcement learning step which rewards generations that enable a user to provide
calibrated answers to related questions. We linguistically calibrate Llama 2 7B and find in automated and
human evaluations of long-form generations that it is significantly more calibrated than strong finetuned
factuality baselines with comparable accuracy. These findings generalize under significant domain shifts to
scientific and biomedical questions and to an entirely held-out person biography generation task. Our results
demonstrate that long-form generations may be calibrated end-to-end by constructing an objective in the
space of the predictions that users make in downstream decision-making.

1 Introduction

The claims made by language models (LMs) are increasingly used to inform real-world decisions,
e.g., what to order at a restaurant, what information to provide someone else about a topic, or which
code completion to accept. However, LMs have knowledge gaps which manifest as hallucinations
(Huang et al., 2023, Ji et al., 2023). Currently, when an LM lacks knowledge about a topic, it will do
one of two things: hallucinate incorrect claims with complete confidence, or, in the case of a few
strong closed-source models (Anthropic, 2023, OpenAI et al., 2023), abstain from making claims.
Confident hallucinations are especially harmful. They decrease users’ trust in the errant LM
and broadly make LMs unsuitable for settings where factuality is paramount such as medicine
(Thirunavukarasu et al., 2023) and law (Dahl et al., 2024). Perhaps most importantly, they lead the
user to confidently make poor decisions (Fig. 1). However, even abstentions are suboptimal, because
they provide the user with no plausible claims and their likelihoods.
Linguistic calibration (Mielke et al., 2022)—conveying confidence levels in natural language that
equal the likelihood that one’s claims are correct—could mitigate the harms of hallucination. If
an LM was linguistically calibrated in a manner interpretable to its users, they could make good
decisions regardless of the LM’s underlying knowledge.
For example, suppose a clinical LM generates a patient’s case report, providing a diagnosis. If the
LM was unsure of the correct diagnosis, it could use numerical confidence with its corresponding
claim (“I estimate that the correct diagnosis is Condition A with 60% confidence”). Then, when the
doctor faces a decision—deciding the patient’s treatment—they have access to both a salient claim

1

Figure 1: Illustrative example of linguistic calibration. We define linguistic calibration of long-form
generations (LC) as calibrating an LM’s generations in a way that leads to calibrated downstream
user forecasts. We apply LC to train an LM that emits calibrated statements of confidence in natural
language, enabling better downstream decisions. Left: users read long-form generations (e.g., a
doctor reading an LM-generated clinical note). Middle: to decide the patient’s treatment, the doctor
first forecasts the patient’s underlying condition. Upper Right: when standard LMs lack knowledge,
they hallucinate confidently, leading to a suboptimal decision (treating the wrong condition). Lower
Right: even if the base LM cannot be confidently correct, linguistic calibration encourages the LM to
spread probability over plausible claims, enabling a better decision.

and an approximate likelihood of its correctness. The manner of conveying confidence is limited
only by the use of language: e.g., the LM could provide linguistic confidence statements (“I am fairly
sure that the correct diagnosis is A”) or many mutually exclusive claims (“I estimate that the correct
diagnosis is A with 60% confidence, though B or C are also possibilities.”). However, both classic
calibration methods like temperature scaling (Guo et al., 2017) and methods for LMs (Jiang et al.,
2021, Kadavath et al., 2022, Kuhn et al., 2023, Lin et al., 2022, Mielke et al., 2022, Tian et al., 2023) are
restricted to classification or short outputs and hence cannot calibrate the many claims made in each
long-form LM generation (cf. §5 for related work).
We make progress on this challenge by leveraging the connection between calibration and decision
theory (Zhao and Ermon, 2021, Zhao et al., 2021). LMs fit cleanly within the framework of decision-
making. Users query LMs, learn from their generations, and later encounter decision-making tasks
with associated questions (“What is the patient’s condition?”). They forecast an answer based on
what they learned, and finally make a decision (which treatment to provide) using their forecast for
which they receive some reward.
Because linguistic calibration improves downstream user decisions, we might hope to calibrate
an LM by directly optimizing real-world downstream rewards. While intuitively appealing, this
process is challenging; it entails tracing the knowledge learnt by users through to their decisions,
and further propagating the associated rewards back to the LM.

Our contributions. We propose a definition of linguistic calibration for long-form generations (LC)
which sidesteps the difficulty of tracing real-world rewards and enables training an LM to emit
calibrated confidence statements in long-form generations. Our contributions are as follows:
• We define an LM emitting long-form generations to be linguistically calibrated if it enables users
to produce calibrated forecasts relevant to their decision-making tasks, which in turn enable optimal
decision-making.

2

… I believe the patient’s symptoms and medical history indicate deconditioning with a 75% likelihood and asthma with a 25% likelihood…… and make decisions using those predictions.… The patient’s symptoms and medical history are consistent with deconditioning, the decline in physical function as a result of inactivity…Standard LMLinguistically Calibrated LMLC Enables Good DecisionsConﬁdent Hallucinations Cause Bad DecisionsScreen patient  for asthmaPrescribe physical therapyUsers learn from long-form LM generations…… make predictions using this knowledge…A: deconditioning (75%) or asthma (25%)A: deconditioning (100%)Q: What is the patient’s underlying condition?• We instantiate this definition in a training objective and framework that calibrates long-form
generations through a decision-theoretic lens. Our training framework first bootstraps an LM to
express confidence statements with supervised finetuning. Then, it optimizes our objective using
reinforcement learning (RL), rewarding the LM policy for generating text that enables calibrated
forecasts on related questions.

• We linguistically calibrate Llama 2 7B using our training framework and find it significantly
improves calibration versus strong baselines finetuned for factuality while matching their accuracy,
in human and API-based LLM evaluation of long-form generations. We also show that linguistic
calibration has zero-shot transfer beyond the training task. Specifically, an LM calibrated using a
single off-the-shelf question-answering dataset is also calibrated on significantly out-of-domain
scientific question-answering datasets, and on an entirely held-out task of biography generation.
Instead of working in the space of text, our decision-based approach constructs an objective in the
space of the predictions that users make in the process of decision-making. This makes the standard
calibration machinery of proper scoring rules (Gneiting and Raftery, 2007) tractable as an objective
for end-to-end calibration of long-form generations.

2 Setup

Our goal is to formulate a tractable objective that enables the end-to-end linguistic calibration
of long-form LM generations. To begin, we define what it means for an LM emitting long-form
generations to be linguistically calibrated.

2.1 Linguistic Calibration of Long-Form Generations

Our definition is motivated by the process through which users learn from LM generations and later
use their knowledge to answer questions. This process of LM-assisted user forecasting allows us to
turn the problem of calibrating long-form text into the problem of calibrating probabilistic forecasts.

LM-assisted user forecasting. First, the user prompts an LM π with an open-ended query q
(e.g., “Generate a clinical note for this patient...”). Next, the LM π generates a long-form context
z ∼ π(z | q), e.g., the clinical note.
At some point in the future, the user encounters a question x associated with the query q, such
as x = “What is the correct diagnosis of the patient?” Let y ∈ Y be the answer to the question
x, and p(y | x) be the ground-truth conditional distribution over answers. We note that given x,
the answer y does not depend on either the query q or the long-form generation z, i.e., we have
the conditional independence statement Y ⊥ (Q, Z) | X. Altogether, we have joint distribution
p(q, x, y, z) := p(q, x, y)π(z | q).
Since p(y | x) is unknown to the user, they form a probabilistic forecast over possible answers y ∈ Y
as a probability distribution in ∆|Y|. In our example, the doctor can make a better forecast over
possible diagnoses by first reading the LM-generated clinical note. In other words, because the
long-form generation z is salient to the question x, the user conditions on it to make their forecast;
the user applies their reader function f : X × Z → ∆|Y| to the question x and long-form generation
z in order to form a forecast f (x, z) ∈ ∆|Y|. For example, we might have f (x, z) = [0.2, 0.8]⊤ placing
20% mass on Condition A and 80% on Condition B. We will denote the probability of answer y
under the forecast as f (x, z)y.

Defining linguistic calibration for long-form generations. LM-assisted user forecasting motivates
a definition of linguistic calibration in terms of the calibration of a downstream reader. Intuitively,

3

Table 1: Each notion of linguistic ϕ-calibration corresponds to a traditional notion of classifier
calibration ϕ applied to the reader f : X × Z → ∆|Y|.

Notion of Classifier Calibration ϕ

Distribution Calibration (Kull and Flach, 2015)

Condition for Linguistic ϕ-Calibration

E [1y | f (x, z) = p] = p,

∀p ∈ ∆|Y|.

Classwise Calibration (Kull et al., 2019)

Confidence Calibration (Guo et al., 2017)

(cid:16)

Y = y

Pr

(cid:12)
(cid:12)
(cid:12) f (x, z)y = β

(cid:17)

= β,

∀y ∈ Y, β ∈ [0, 1].

(cid:16)

Pr

Y = arg maxy f (x, z)y

(cid:12)
(cid:12)
(cid:12) maxy f (x, z)y = β

(cid:17)

= β,

∀β ∈ [0, 1].

an LM is linguistically calibrated with respect to a downstream user if its long-form generations z
enable that user to make calibrated forecasts on question-answer pairs (x, y) using their reader func-
tion f . This notion of linguistic calibration is a property of an LM, and is defined with respect to a
particular user and notion of classifier calibration.

Definition 2.1 (Linguistic Calibration of Long-Form Generations). An LM π emitting long-form
generations z ∼ π(z | q) is linguistically ϕ-calibrated (for ϕ ∈ {distribution, classwise, confidence})
with respect to a readerf : X × Z → ∆|Y| if and only if f is ϕ-calibrated on the joint distribution p(x, y, z).

Prior work established linguistic calibration for single-claim utterances (Mielke et al., 2022). Our
definition generalizes linguistic calibration to long-form generations, which we will show has both
theoretical and empirical benefits. Our focus is on calibrating long-form text, so future references to
linguistic calibration, linguistically calibrated, etc. refer to Defn. 2.1. §5 discusses related work.
Defn. 2.1 reduces the problem of calibrating long-form text to calibrating over a smaller space of
answers Y. Each notion of linguistic ϕ-calibration is equivalent to a traditional notion of classifier
calibration ϕ applied to the reader f . To be self-contained, we next provide examples in our notation.
Readers familiar with classifier calibration could skip to §2.2.

Examples of linguistic ϕ-calibration. Consider distribution calibration as a specific notion of
classifier calibration (Bröcker, 2009, Kull and Flach, 2015). Intuitively, a classifier g : X ′ → ∆|Y| is
distribution calibrated if its forecast g(x′) ∈ ∆|Y| is faithful to the predictive uncertainty (Minderer
et al., 2021); among all inputs x′ ∈ X ′ for which the forecasted probability of class y is 0.4, i.e.,
g(x′)y = 0.4, 40% of those inputs should have label y. Taking 1y ∈ {0, 1}|Y| as the one-hot vector for
y, g is distribution calibrated if

E (cid:2)1y | g(x′) = p (cid:3) = p,

∀p ∈ ∆|Y|.

Now taking our reader f := g as the classifier and both the question and LM generation as the input
(x, z) := x′, we say that an LM π is linguistically distribution calibrated with respect to a reader f if

E [1y | f (x, z) = p ] = p,

∀p ∈ ∆|Y|.

We provide more examples of this correspondence in Table 1. These include the weaker notion of
confidence calibration (Guo et al., 2017), which means that among the examples with top forecasted
probability equal to β, the accuracy over those examples is also β.

2.2 From Calibration to Optimal Decisions

The process of LM-assisted user forecasting models how users learn about a variety of topics by
reading LM generations and forecast answers to related questions. We now describe how they
use their forecasts to inform downstream decisions. This decision-based perspective leads to our
objective for linguistic calibration.

4

LM-assisted user decision-making. The user will make a decision by selecting an action a ∈ A
(which treatment to provide the patient) according to their forecast f (x, z) ∈ ∆|Y| and loss function
L : A × Y → R. For example, the user may make a Bayes-optimal decision by choosing the action
which minimizes their loss under the forecast distribution: a∗ = arg mina E
(cid:98)y∼f (x,z)[L(a, (cid:98)y)]. Based
on the realized answer y ∼ p(y | x), the user suffers a loss of L(a∗, y).
Ideally, one would follow this LM-assisted user decision-making process and directly train an
LM to emit generations z which minimize the user’s downstream loss L(a∗, y). However, it is
difficult to obtain real-world rewards from decision-making, and moreover to obtain a real-world
distribution over queries to LMs and related question-answer pairs p(q, x, y). We overcome both of
these challenges by optimizing for linguistic calibration over a representative surrogate distribution. In
other words, we encourage the LM to emit generations z that enable the user to provide calibrated
answers to questions x.
In the rest of this section, we will introduce our objective for linguistic calibration and demonstrate
its optimality properties. We discuss our surrogate distribution in §3.1.

Linguistic calibration implies informed decision-making. Why is optimizing for linguistic
calibration a reasonable proxy for directly optimizing downstream losses L(a∗, y)? Zhao et al. (2021)
prove that calibrated forecasts f (x, z) enable a user to make informed decisions. Specifically, the user
may use their calibrated forecast to estimate the expected loss of a decision-making strategy as if
they had access to the ground-truth distribution p(y | x). This accurate loss estimation implies that
Bayes-optimal decision-making outperforms other decision-making strategies. Altogether, because
linguistic calibration implies calibrated forecasts by definition (cf. Defn. 2.1), it will provide optimal
decision-making guarantees.

Guarantees for weaker notions of calibration. Each notion of linguistic ϕ-calibration (cf. Table 1)
in fact corresponds to a different decision-making guarantee. Even if our optimization process cannot
achieve linguistic distribution calibration, achieving linguistic ϕ-calibration with respect to a weaker
notion ϕ such as classwise or confidence calibration will still provide a (weaker) decision-making
guarantee. Appendix B provides a proof of this correspondence (Theorem B.4).
Lastly, we note that this connection between linguistic calibration and optimal decision-making
motivates our evaluation metric of reader expected calibration error (defined in §4.1).

2.3 Training Objective for Linguistic Calibration

Because linguistic calibration of long-form generations is equivalent to the calibration of downstream
probabilistic forecasts, we can optimize for it with the standard calibration machinery of scoring
rules. We use a strictly proper scoring rule as our optimization objective, with the guarantee that it
is maximized if and only if user forecasts f (x, z) equal the ground-truth distribution p(y | x).

Proper scoring rules. There is a rich literature on scoring rules (Gneiting and Raftery, 2007, Savage,
1971), which measure the quality of a forecast. Specifically, a scoring rule R scores a forecast (cid:98)p ∈ ∆|Y|
and outcome y ∈ Y, i.e., R : ∆|Y| × Y → R. Proper scoring rules have the desirable property that the
true distribution of y, denoted by p, is a maximizer. Formally, R is proper if

Ey∼p[R(p, y)] ≥ Ey∼p[R((cid:98)p, y)], ∀p, (cid:98)p ∈ ∆|Y|.
R is strictly proper if the unique maximizer of the expected reward is the true probability p. Strictly
proper scoring rules are ubiquitous in ML, e.g., the (negative) log loss or Brier score (Brier, 1950),
and are natural objectives for calibration methods such as temperature (Guo et al., 2017) or Platt
scaling (Platt et al., 1999).

5

Figure 2: Our training framework for linguistic calibration of long-form generations (LC) calibrates
the long-form generations of an LM by calibrating downstream user forecasts. It involves two steps:
summary distillation (Upper) and decision-based RL (Lower). Datasets are in white, LMs in blue,
and steps involving user or surrogate forecasts are in green.

Linguistic calibration objective. We simply choose the negative log loss as our strictly proper
scoring rule, that is, we use RLC(p, y) := log py, and therefore our objective function is

RLC(f (x, z), y) := log f (x, z)y.

(1)

By strict propriety, if our objective is maximized, then the user’s forecast f (x, z) equals the ground-
truth p(y | x) and we obtain linguistic distribution calibration.
Proper scoring rules do not guarantee that weaker notions of calibration hold near their optima.
However, it is well-established empirically that recalibrating classifiers by optimizing proper scoring
rules such as the log loss approximately achieves weaker notions of calibration such as confidence
calibration (Guo et al., 2017, Minderer et al., 2021, Ovadia et al., 2019).

3 Method

In this section, we describe our training framework for linguistically calibrating an LM. We begin by
constructing a surrogate distribution to approximate the distribution of question-answer pairs that
users encounter during real-world decision-making. Then, we apply a two-step training framework
(Fig. 2). First, we obtain an LM with some ability to express confidences in a long-form generation.
Second, we use it as an RL policy and optimize our proper scoring rule objective end-to-end, with
supervision from the surrogate question-answer distribution.

6

Summary DistillationLM-Assisted User Forecasting 0.50.30.2RL with Policy and Surrogate ReaderInitialize a policy capable of expressing conﬁdence statements in natural languageWrite a paragraph about the band that  sang “No Distance Left to Run”.Base LMDraw samples {zi}Mi=1Blur are an English rock band formed…Alt-J are an English indie rock band…… I estimate the band was Alt-J with 90% conﬁdence, Blur with 5% conﬁdence…Summarize into sLC SFT Initial Policy Finetune on  pairs(q,s)LC RL Final Model Surrogate Reader (x,z)↦˜f(x,z)∈Δ|𝒴|Alt-J are one of the most commonly…Decision-Based RL…Open-ended query  q… I believe there is a 50% chance the Atomium is in Brussels, a 30%……I am unsure of which book inspired the ﬁlm Slumdog…BrusselsParisViennaReward generations that enable a user to provide calibrated answers to a related questionGiven  and question , user provides answer forecast  zxf(x,z)∈Δ|𝒴|Approximate forecasts with LMsSample paragraphs  from SFTzWith ground-truth answer  and proper scoring rule , use reward   yRR(˜f(x,z),y)3.1 Generating Synthetic Supervision for Long-Form Calibration

In our setup (§2.1), the process of LM-assisted user forecasting involves a tuple (q, z, x, y), where q
are user-written queries to the LM, z ∼ π(z | q) are long-form generations sampled from the LM,
and (x, y) is a related question-answer pair.
Our training framework will closely follow this process and therefore requires access to a dataset of
tuples (q, z, x, y). We can synthetically generate this dataset in a manner agnostic to the downstream
task, using arbitrary question-answer pairs (x, y). In this work, we make a particular choice to use
(x, y) pairs from off-the-shelf question-answering datasets.
Specifically, we first sample a question-answer pair (x, y) ∼ p(x, y) from a question-answering
dataset, which is implicitly associated with some actual decision-making tasks. Next, we need
an LM query q such that z ∼ π(z | q) is a long-form generation salient to (x, y). We obtain one by
converting the question x into an open-ended query (q := “Write a paragraph about {x}”) using
an API-based LLM. Altogether, this gives us a tuple (q, z, x, y) where (q, x, y) ∼ p(q, x, y) and
z ∼ π(z | q).
Next, we describe our two-step training framework.

3.2 Summary Distillation

Summary distillation (Fig. 2 Upper) bootstraps a base LM πBase to have some ability to express
its confidence in long-form natural language generations. We follow a simple approach inspired
by Self-Consistency (Wang et al., 2023), which obtains calibrated LM confidences for short answer
questions by computing a statistic of many output samples. Summary distillation generalizes this
idea to longer generations, and then finetunes on our equivalent of the statistics.
First, we provide the base LM with an open-ended query q and sample many long-form responses:
{zi}M
i=1 ∼ πBase(z | q). To obtain statements of confidence that are faithful to the base model’s
internal confidence levels, we prompt an API-based LLM to summarize these samples into a
single consensus paragraph s with statements of confidence based on the frequency of claims:
s ∼ Summarize(s | z1, . . . , zM ). For example, we would expect the summary shown in Fig. 2
(Upper) if 90% of the M samples answer the question with Alt-J and 5% with Blur. We perform
frequency-based summarization at the claim level, meaning that each summary paragraph s contains
multiple claims with various confidence levels and styles (e.g., numerical and linguistic).
Finally, to distill these extracted confidences back into the base model, we finetune πBase on the
dataset of open-ended query and summary pairs {(q(i), s(i))}N
i=1 to obtain the supervised finetuned
(SFT) model πSFT. πSFT serves as a strong initial policy for the second RL-based step.

3.3 Decision-Based RL

Decision-based RL (Fig. 2 Lower) linguistically calibrates a policy πRL (initialized at πSFT) by
finetuning it to emit long-form generations z that improve the calibration of the user forecast f (x, z).

RL objective. We use our log likelihood proper scoring rule RLC (cf. §2.3) as our reward function.
Our RL objective optimizes RLC over our semi-synthetic distribution:

max
πRL

E
(q,x,y)∼p(q,x,y), z∼πRL(z|q)

[RLC (f (x, z), y)] .

(2)

7

Algorithm 1: Decision-Based RL with a Surrogate Reader
Given: SFT LM πSFT, LLM simulated reader fLLM : X × Z → ∆|Y|, joint distribution over

open-ended queries and question-answer pairs p(q, x, y).

Result: linguistically calibrated LM πLC.
/* Construct datasets for training surrogate reader functions: ExtractAnswers (EA) and ForecastProbs (FP).
DEA ← {}, DFP ← {}.
for i = 1, . . . , N do

/* LM-assisted user forecasting (§2.1).
Sample open-ended query and question-answer pair (q, x, y) ∼ p(q, x, y).
Sample long-form generation z ∼ πSFT(z | q).
Apply LLM reader to obtain forecast fLLM(x, z) ∈ ∆|Y|.
/* Add ExtractAnswers example.
DEA ← DEA ∪
/* Add ForecastProbs example for ground-truth answer.
x, z, y, fLLM(x, z)y
DFP ← DFP ∪
/* Add ForecastProbs example for all extracted answers.
for extracted answer y′ ∈ Support(fLLM(x, z)) do

x, z, Support(fLLM(x, z))

(cid:17)

(cid:16)

(cid:16)

(cid:17)

.

.

DFP ← DFP ∪

end

(cid:16)

x, z, y′, fLLM(x, z)y′

(cid:17)

.

end
/* Finetune surrogate LMs.

Finetune ExtractAnswers on DEA and ForecastProbs on DFP.

/* Optimize RL objective.
πLC ← PPO with initial policy πSFT, surrogate reader (cid:101)f , and Objective 3.
return πLC

*/

*/

*/

*/

*/

*/

*/

3.4

Implementation

We next describe our instantiation of decision-based RL, which we used in our experiments. How-
ever, we note that the notion of linguistic calibration defined in §2 is agnostic to these design
decisions. Algorithm 1 presents the pseudocode.

Surrogate reader. For our training framework to be as scalable as possible, we would ideally avoid
the use of a human or LLM-simulated reader f in the RL loop. We find that we can train a neural net
surrogate reader (cid:101)f : X × Z → ∆|Y| which produces reasonable forecasts, because “reading” is not a
fundamentally hard task. For example, if z provides a clear list of possible answers to the question x
and associated percentage likelihoods, reading is a simple extractive task.
Using the surrogate, we optimize approximate reward RLC( (cid:101)f (x, z), y) ≈ RLC(f (x, z), y). In our
evaluation, we will test if our LM calibrated on this approximate reward generalizes to produce
long-form generations z which improve simulated LLM and human forecasts f (x, z).
We cannot simply train a neural net to directly predict a softmax output (cid:101)f (x, z) ∈ ∆|Y|, because Y is
the vast space of all answers expressible in a finite-length string. Instead, we decompose forecasting
into two operations:

8

1. ExtractAnswers: (x, z) (cid:55)→ ExtractAnswers(x, z) ⊂ Y extracts all possible answers to the ques-
tion x from the paragraph z. We implement this by finetuning a pretrained LM (RedPajama 3B,
together.ai (2023)).

2. ForecastProbs: (x, z, y′) (cid:55)→ [0, 1] assigns a probability to an answer y′ to question x based on

the paragraph z. We finetune πSFT with a cross-entropy loss.

We define the surrogate reader’s forecast (cid:101)f (x, z) as a categorical distribution with probability
ForecastProbs(x, z, y′) on each answer y′ ∈ ExtractAnswers(x, z), and probability 0 on all others. In
this particular construction, we are not guaranteed that the surrogate forecast (cid:101)f (x, z) will be normal-
ized, but in practice we find that adding a regularization term is sufficient to enforce normalization:

RLC( (cid:101)f (x, z), y) = log (cid:101)f (x, z)y −λ

(cid:12)
(cid:12)1−(cid:80)
(cid:12)

y′∈Y (cid:101)f (x, z)y′

(cid:12)
(cid:12)
(cid:12) ,

where λ > 1 restores strict propriety (cf. C.1 for proof). Lastly, we use a standard KL penalty
from πSFT to mitigate over-optimization of the surrogate reader (Ouyang et al., 2022), giving us the
following objective (with KL coefficient β):

max
πRL

E
(q,x,y)∼p(q,x,y), z∼πRL(z|q)

(cid:20)
RLC( (cid:101)f (x, z), y) − β log

πRL(z | q)
πSFT(z | q)

(cid:21)

.

(3)

See Appendix C for further implementation details.

4 Experiments

This section empirically validates our training and evaluation framework for linguistic calibration1,
demonstrating that it fulfills the following three goals:

(1) LC provides better calibration with comparable or better accuracy. We show that our linguis-
tically calibrated LM πLC emits long-form generations z which improve the calibration of user
forecasts with accuracy comparable to or better than strong baselines finetuned for factuality
with RL.

(2) LC is computationally tractable. We show that πLC—which avoids the need to obtain many
costly human forecasts by training with cheap surrogates—improves the calibration of human
forecasts at evaluation time. Moreover, we develop an automated framework to evaluate
linguistic calibration with simulated forecasts and validate its agreement with crowdworkers.
(3) LC generalizes well out-of-distribution. We demonstrate that the improvement in forecast
calibration due to adopting LM πLC generalizes under significant domain shift to scientific
question-answer distributions pOOD(x, y). We also evaluate πLC on an entirely held-out task
of person biography generation without any re-training, finding that πLC produces calibrated
claims throughout the long-form generation z according to a fine-grained simulated evaluation.

4.1 Setup

We use our training framework to linguistically calibrate Llama 2 7B, sourcing question-answer
(QA) pairs (x, y) (cf. §3.1) from TriviaQA (Joshi et al., 2017). We emphasize that our LMs produce
long-form generations z on the question’s topic, unlike previous works which calibrate models that
directly predict a class distribution or short answer (cf. §5). We refer the reader to Appendix C for
further details on the training framework.

1We release all parts of our training and evaluation framework at github.com/tatsu-lab/linguistic_calibration.

9

Question-answering evaluation framework. Following our generative process during training
(§3.1), we use off-the-shelf QA datasets as a proxy for questions encountered during real-world
decision-making, and evaluate the linguistic calibration of generations z through the performance of
downstream forecasts. Specifically, for a held-out QA pair (x, y), we convert x into an open-ended
query q, sample a long-form generation z ∼ π(z | q) from various LMs π, and evaluate the calibration
and accuracy of forecast f (x, z).
Naturally, this framework depends on which users are providing forecasts and how, i.e., the choice
of reader f . We are interested in the case where users strongly rely on the knowledge of the
LM. Therefore, we include instructions to the user (either simulated or human) to ignore their
background knowledge about the correct answer when providing a forecast (cf. Appendix D for
further evaluation details).

Reader expected calibration error. We measure the calibration of the reader f over the joint
distribution p(x, y, z) with expected calibration error (ECE) (Guo et al., 2017). Intuitively, reader
ECE is a proxy for decision-making performance through the equivalence of confidence calibration
and optimal decision-making: see Theorem B.4 in Appendix B for details. Given N question-answer
pairs {(x(i), y(i))}N
i=1, we partition them into M
bins by max forecast probability maxy f (x(i), z(i))y. ECE is then expressed as

i=1 and corresponding long-form generations {z(i)}N

ECE =

M
(cid:88)

j=1

|Bj|
N

|Acc(Bj) − Conf(Bj)| , where

Conf(Bj) =

1
|Bj|

(cid:88)

i∈Bj

max
y

f (x(i), z(i))y, Acc(Bj) =

1
|Bj|

(cid:88)

i∈Bj

1[y(i) = arg max

y

f (x(i), z(i))y].

We set the number of bins as M = 20 on simulated QA evaluations, and M = 10 on all others. Note
that log loss is not a reasonable evaluation metric in our setting because simulated and human
readers can assign zero probability to the ground-truth class label resulting in infinite log loss.

In our main evaluations, we compare LC RL (πLC) with two types of baselines, all
Baselines.
derived from Llama 2 7B: non-confidence and confidence. We provide a strong data-matched
comparison to LC by finetuning directly for factuality using RL. This baseline is similar to Tian et al.
(2024), but instead of using self-supervised or automated factuality scores as the RL reward, we
use correctness determined with ground-truth question-answer pairs from TriviaQA. In-context
learning (ICL) baselines use TriviaQA examples from a prompt development split, and SFT/RL-
based baselines use the same splits as πSFT and πLC. Each example in our splits is a (q, x, y) tuple,
where q is an open-ended query obtained from question x (cf. §3.1).
We include the following non-confidence baselines:

• ICL. We randomly sample 8 open-ended queries, generate long-form responses with GPT-4,
manually fact-check those responses using Wikipedia, and use these fact-checked (query, response)
pairs as ICL examples for Llama 2 7B.

• Claude Distill. We generate long-form responses with Claude 2 over all queries in the SFT split,

and finetune Llama 2 7B on these (query, response) pairs.

• Factuality SFT. We use the above ICL baseline to generate long-form responses over all queries in
the SFT split, and finetune Llama 2 7B on these (query, response) pairs. We found Factuality SFT
to outperform Claude Distill on a TriviaQA validation split, so we use it as the starting point for
the following baseline, Factuality RL.

10

(a) TriviaQA (in-distribution),
simulated reader.

(b) TriviaQA (in-distribution),
human readers.

(c) SciQ (out-of-distribution),
simulated reader.

Figure 3: Accuracy-ECE Frontier for Question-Answering (upper left is better). LC RL pareto-
dominates Factuality RL and SFT, with significantly better reader ECE while matching or exceeding
their accuracy.

• Factuality RL. To provide a strong RL-based baseline, we train a reward model which scores
the correctness of long-form outputs and use it in RL. Our approach to obtain this baseline is
analogous to the decision-based RL algorithm (Algorithm 1), except instead of training a surrogate
reader, we train a single reward model that, given a generation z and question-answer pair (x, y),
predicts a binary indicator whether z provides the correct answer to the question. This serves as
the RL reward. We use Factuality SFT as the initial policy for PPO.

When training our confidence methods, we use the ICL baseline above to generate the responses
which are summarized in summary distillation. Our confidence baselines include the LC SFT model
(πSFT) and the following baseline:
• Summary ICL. We use the summary distillation algorithm (§3.2) on 8 queries sampled from the
i=1, which we use in (query,

prompt development split to produce 8 Claude 2 summaries {s(i)}8
summary) ICL examples.

Other baselines including GPT-4. We include results for several other methods in Appendix
A including Llama 2 Chat (which underperformed Factuality SFT), the oracle baseline of direct
evaluation of summaries s, and GPT-4–based methods including GPT-4 0-Shot, GPT-4 ICL 8-Shot,
asking GPT-4 for confidence statements zero-shot, and Summary ICL using GPT-4. Unsurprisingly,
GPT-4–based methods are far more factual than all Llama 2 7B–based methods. However, we find
that LC RL has reader ECE comparable to GPT-4 baselines (cf. Figs. 8, 10, 12, 14, 16), despite signifi-
cantly worse factuality. This demonstrates that even small LLMs with relatively weak factuality can
be well-calibrated with the right objective.

4.2 Linguistic Calibration using Question-Answering Datasets

To begin, we evaluate our methods using held-out (x, y) pairs from the TriviaQA, Jeopardy (Kaggle,
2020), SciQ (Welbl et al., 2017), and BioASQ Task B (Krithara et al., 2023) question-answering
datasets. The latter two datasets are a significant domain shift; SciQ consists of crowdsourced
physics, chemistry, and biology exam questions, and BioASQ Task B consists of biomedical questions
annotated by experts (cf. Appendix D for dataset details).
These QA evaluations validate two of our three goals: we find that LC improves calibration with
comparable or better accuracy, and that our training and evaluation framework are computationally

11

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)Summary ICL 8-ShotNon-Confidence BaselinesFactuality RLFactuality SFTICL 8-Shot0.100.150.200.250.300.350.40ECE0.590.600.610.620.630.640.650.660.67Accuracy0.100.150.200.250.300.350.400.45ECE0.560.580.600.620.200.250.300.350.400.45ECE0.5400.5450.5500.5550.5600.5650.5700.575(a) LC SFT, human
readers. ECE: 0.163

(b) LC RL, human
readers. ECE: 0.116

(c) LC SFT, simulated
reader. ECE: 0.166

(d) LC RL, simulated
reader. ECE: 0.108

Figure 4: TriviaQA Reliability Diagrams. LC models display a wide range of confidences and good
calibration in their long-form generations, with LC RL improving calibration further. Human and
simulated results closely match.

tractable. Our strong results on the Jeopardy, SciQ, and BioASQ datasets also provide a partial
validation of our final goal of strong generalization under distribution shift. We report TriviaQA
and SciQ results here, and refer the reader to Appendix A for results with all datasets and baselines.

Better ECE with comparable or better accuracy in long-form generation. Our main result in Fig. 3
is that LC RL has significantly better ECE than non-confidence baselines, including Factuality RL,
while matching or exceeding their accuracy. This result holds in-distribution on TriviaQA with both
simulated (Fig. 3a) and human (Fig. 3b) readers, and out-of-distribution (OOD) on SciQ (Fig. 3c),
Jeopardy (Fig. 10), and BioASQ (Fig. 14) with a simulated reader, demonstrating that LC generalizes
across distribution shifts in the question-answer distribution p(x, y).
Our results also support the effectiveness of decision-based RL. LC RL significantly improves over
LC SFT in both ECE and accuracy, with a greater absolute improvement in ECE/accuracy than
Factuality SFT to Factuality RL. This supports our claim that optimizing proper scoring rules of
downstream forecasts is an effective way to induce calibration in long-form generations.

Reliability diagrams demonstrate meaningful confidences. A natural question is whether the
confidences learned by our LC models are meaningful. For example, if all confidences of a model
collapsed to its average accuracy, it would obtain perfect ECE despite having confidences that are
useless for tasks such as conveying the likelihood of hallucination. To obtain a more fine-grained
understanding of a model’s calibration, we use reliability diagrams (DeGroot and Fienberg, 1983,
Niculescu-Mizil and Caruana, 2005), which visualize the average confidence and accuracy of each
ECE bin. The plot of the pathological model above would be a single point. A perfectly calibrated
model with meaningful confidences would have an identity plot y = x. In Fig. 4, we observe
that LC model confidences are both meaningful, covering a wide range of confidence values, and
consistently close to the identity across confidence values. This validates that LC is effective in
linguistically conveying the likelihood of hallucination in a long-form generation.

4.3 Zero-Shot Generalization to a Biography Generation Task

The QA evaluation validated two of our three goals: (1) LC RL pareto-dominates baselines on
the accuracy-ECE frontier. Its significant improvement over LC SFT validates the effectiveness of
decision-based RL. (2) We demonstrated the computational efficiency of our training and evaluation
framework, because LC RL is trained with cheap surrogates but performs well in evaluations with
human readers, and our simulated readers have high agreement with human readers (see Appendix
D.3 for full forecast agreement statistics). Lastly, the QA evaluation partially validates (3) OOD

12

Perfectly CalibratedLM0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00AccuracyPrompt: Write a paragraph bio about Rory Byrne.

Factuality RL: Rory Byrne is a renowned British engineer and designer who has made significant contributions
to the world of Formula One racing. Born in 1949, Byrne began his career in the automotive industry in the
1970s, working for various racing teams and manufacturers. In 1981, he joined the Williams Formula One team
as a designer, and over the next few years, he helped the team win several championships. In 1994, Byrne left
Williams to join Ferrari, where he served as the team’s technical director for several years. During his time at
Ferrari, Byrne helped the team win several championships, including the 1999 constructor’s championship. In
2005, Byrne left Ferrari to pursue other opportunities, and he has since worked as a consultant for various racing
teams and manufacturers.

LC RL: I believe Rory Byrne is a renowned British engineer who has made significant contributions to the
automotive industry. I estimate with 75% confidence that he has worked as a chief designer for Formula 1 teams
like Ferrari and McLaren. I also think with 50% likelihood that he has designed cars for teams like Williams,
Benetton, and Lotus. I am uncertain whether he was born in Ireland or Scotland. I believe with 90% confidence
that he has won multiple design awards, including the prestigious Loeb Award. While the details of his career are
unclear to me, I am certain that Rory Byrne has had a profound impact on the development of Formula 1 cars over
the past few decades. His innovative designs have helped teams achieve success on the track.

Figure 5: Qualitative example from Factuality and LC RL when evaluated under task distribution
shift on biography generation. LC RL produces numerical and linguistic confidence statements
throughout the paragraph, highlighted in blue. False statements are highlighted in red. We include
additional examples in Appendix A.13.

generalization: LC RL performs well on the Jeopardy, SciQ, and BioASQ datasets with simulated
readers.
To conclusively validate this final goal, we evaluate LC on a significant distribution shift in the task.
Our models were trained to perform long-form generation about trivia-style questions, and we now
test their ability to write factual biographies on people sampled from Wikipedia. Specifically, we
source 500 people from the unlabeled split of FactScore (Min et al., 2023) and use prompt “Write a
paragraph bio about {person}” (cf. Fig. 5 for a qualitative example, and additional ones in Appendix
A.13). We emphasize that our models were not trained on biography generation.

FactScore-based metric. We also use a more fine-grained simulated evaluation than the QA
tasks, testing the accuracy and calibration of generated biographies z at the per-claim level. More
specifically, we split generations into a list of claims, filter out subjective claims, and then compute
accuracy and ECE over all claims pooled across biographies, following FactScore (Min et al., 2023).
We use Claude 2 for splitting and filtering, and an identical fact checking pipeline to Min et al.
other than using Claude 2 instead of ChatGPT for fact-checking conditioned on retrieved Wikipedia
context paragraphs. In order to compute ECE, we need to assign confidence values to each atomic
claim. For numerical uncertainties such as percentages, this is a simple extractive task which
API-based LLMs perform well. For linguistic uncertainties, we provide Claude 2 with a short list
of mappings between linguistic phrases and consensus probabilities collected in a human study
(Wallsten, 1990), and allow the LLM to generalize from this to assign probabilities for other linguistic
phrases not present in the mapping (cf. Appendix D.2 for details).

LC generalizes to biography generation and claim-level evaluation. Both LC methods demon-
strate significant improvements in ECE and accuracy compared to non-confidence baselines, gener-

13

Figure 6: Accuracy-ECE Frontier for Person Biography Generation, simulated reader (upper left
is better). Accuracy and ECE are computed at the claim level, using a finetuned retrieval model
and API-based LLM for fact checking, following Min et al. (2023). LC RL significantly outperforms
non-confidence methods in accuracy and ECE.

(a) LC SFT. ECE: 0.301

(b) LC RL. ECE: 0.266

Figure 7: Person Biography Generation Reliability Diagrams for LC SFT (Left) and LC RL (Right);
simulated reader. LC methods demonstrate non-trivial calibration in an entirely out-of-distribution
task which requires calibrated statements of confidence throughout the paragraph.

alizing well to an entirely held-out task (Fig. 6). Because we here compute ECE at the per-claim level
and the LC methods obtain reasonable ECE values and reliability diagrams (Fig. 7), we confirm that
they incorporate calibrated statements of confidence throughout their long-form generations. Addition-
ally, decision-based RL significantly improves both accuracy and ECE over LC SFT even under a
significant task distribution shift, further validating our linguistic calibration objective.
Lastly, we note the surprising finding that LC SFT improves in accuracy compared to Factuality
RL. We attribute this to the tendency of LC models to generate a higher proportion of less “precise”
claims which are still objective and correct and therefore count towards the accuracy metric.
Altogether, our strong results in a challenging distribution shift setting validate our final goal.

14

Our MethodLC RL (LC)Confidence BaselineLC SFT (SFT)Non-Confidence BaselinesFactuality RLFactuality SFTClaude DistillLlama 2 Chat 7B0.250.300.350.400.450.500.550.600.65ECE over Claims0.360.380.400.420.440.460.48Accuracy over ClaimsPerfectly CalibratedLM0.00.20.40.60.81.0Confidence0.00.20.40.60.81.0Accuracy0.00.20.40.60.81.0Confidence0.00.20.40.60.81.0Accuracy5 Related Work

Calibration. The calibration of probabilistic forecasts is an extensively studied topic (Brier, 1950,
Dawid, 1984, Gneiting and Raftery, 2007, Hebert-Johnson et al., 2018, Kull and Flach, 2015, Murphy,
1973, Savage, 1971, Zadrozny and Elkan, 2001). In particular, isotonic regression (Niculescu-Mizil and
Caruana, 2005), Platt scaling (Platt et al., 1999), temperature scaling (Guo et al., 2017), and histogram
binning (Kumar et al., 2019) are effective approaches for improving the calibration of probability
estimates. Other established methods improve calibration through ensembling (Lakshminarayanan
et al., 2017) and Bayesian model averaging using an approximate posterior over model parameters
(Band et al., 2021, Blundell et al., 2015, Gal and Ghahramani, 2016, Malinin et al., 2021, Nado et al.,
2022, Tran et al., 2022).
Recent works also studied the probabilistic calibration of LMs, showing that LMs can be well-
calibrated (Kadavath et al., 2022). LMs that went through RL from human feedback (RLHF) tend to
be ill-calibrated on multiple-choice questions (OpenAI et al., 2023), but simple temperature scaling
fixes the issue. Chain-of-thought prompting leads to ill-calibration (Bai et al., 2022). The present
focus is the linguistic calibration of LMs that produce long-form natural language, a setting to which
established recalibration methods do not apply. This is because long-form generations may contain
multiple claims, each of which has its confidence expressed in language.

Calibration and decision-making. The particular decision-making framework we adopted was
originally used to convey the confidence of individual predictions to decision-makers (Zhao and
Ermon, 2021), and later used to draw an equivalence between different notions of calibration and
optimal decision-making (Zhao et al., 2021) (cf. §2.2 and Appendix B). In seminal work, Foster and
Vohra (1999, 1998) showed that the existence of certain no regret schemes in an online decision-
making setting imply the existence of calibrated probabilistic forecasts. More recently, Cresswell et al.
(2024) explore using conformal prediction sets in classification tasks to improve human decision-
making, but do not consider calibrating long-form LM generations.

LMs producing uncertainty. The literature has studied methods to let LMs produce confidence
scores. Works have benchmarked the uncertainty quantification capabilities of LMs on multiple
choice tasks (Ye et al., 2024) and studied how sampling variance could be used to estimate un-
certainty (Kuhn et al., 2023, Malinin and Gales, 2021, Wang et al., 2022, Xiong et al., 2024). Our
sampling-based Summary ICL and Direct Summary Eval baselines (cf. Appendix A) can be seen as
extensions of the black-box confidence elicitation framework in Xiong et al. (2024) to long-form text,
because they do not require finetuning or access to logits.
Other works analyze how LMs express linguistic uncertainty (Mielke et al., 2022, Zhou et al., 2023).
Considering single-claim utterances, Mielke et al. (2022) define an LM to be linguistically calibrated
if it emits a verbalized confidence statement matching the likelihood of its response’s correctness.
However, the long-form, multi-claim generations that users encounter in practice have neither a
single closed-form confidence nor a correctness; each generation contains information that answers
many possible downstream questions. Defn. 2.1 recovers the notion of linguistic calibration in
Mielke et al. when specialized to single-claim outputs z.2
A related line of work enables LMs to directly express uncertainty, focusing on short utter-
ances (Mielke et al., 2022, Xiong et al., 2024), arithmetic problems (Lin et al., 2022), and question-
answering (Jiang et al., 2021, Shrivastava et al., 2023, Tian et al., 2023). All these prior works consider

2Let q := x, and suppose the LM generates a single-claim utterance z ∼ π(z | q) with a confidence statement and
answer from Y, e.g., z = “I’m not sure, but my guess is Los Angeles.” Then Mielke et al. (2022) considers linguistic
confidence calibration.

15

settings where the set of responses is a small closed set and the notion of calibration is well-defined.
Other works finetune LMs to abstain (Cheng et al., 2024) or to output templated uncertainty phrases
(Yang et al., 2023) on question-answering tasks. Lastly, concurrent work (Huang et al., 2024) evalu-
ates methods such as self-consistency (Wang et al., 2022) and supervised finetuning in calibrating
long-form generations. To the best of our knowledge, our method is the first to simultaneously
provide calibrated text-based statements of confidence, which are important for interpretability to
users (Mielke et al., 2022), while working in the setting of long-form, multi-claim generations. We
overcome the challenge of defining calibration in this setting by drawing connections between
decision-making and uncertainty quantification, enabling us to build a single end-to-end objective that
can calibrate long-form generations.

Other LM finetuning works.
Improving the factuality of LMs is a complementary approach
to calibration in mitigating LM hallucinations. Previous works have improved the factuality of
LMs by finetuning on self-supervised or automated factuality scores (Akyürek et al., 2024, Tian
et al., 2024). A related line of work uses supervised finetuning and RL to improve the honesty of
LLMs (Askell et al., 2021, Cui et al., 2023, Evans et al., 2021, Ouyang et al., 2022, Park et al., 2023),
hypothesizing that the pretraining objective alone is insufficient to encourage honest responses.
Because improving factuality alone can improve calibration metrics such as ECE, we include a
strong baseline finetuned with RL on ground-truth factuality labels and find that our approach to
linguistic calibration significantly improves ECE beyond this baseline while matching or exceeding
its accuracy.

6 Limitations, Future Work, and Conclusions

Limitations and future work. Our linguistically calibrated LM generalizes well from surrogate
to crowdworker forecasts. However, many of the confidence statements it emits are fairly unam-
biguous, e.g., percentages. Therefore, future work could investigate how closely LM and human
interpretations of ambiguous linguistic confidence statements match, which could enable training
LMs with linguistic confidence statements that are tailored to user populations. Additionally, we use
off-the-shelf question-answering datasets as a proxy for questions encountered during real-world
decision-making. To improve LC’s generalization to decision-making scenarios in-the-wild, future
work could curate a more representative QA dataset. Lastly, we work in a white-box setting where
finetuning LMs is possible; our training framework could not be used to calibrate API-based LLMs
that only provide access to completions.

Conclusions. We defined linguistic calibration of long-form generations: calibrating the long-form
generations of an LM in a way that leads to calibrated probabilistic forecasts by its downstream
users. By constructing an objective in the space of these forecasts, we were able to apply the standard
calibration machinery of proper scoring rules for end-to-end linguistic calibration. Instantiating
this objective in a training framework and linguistically calibrating Llama 2 7B enables it to emit
calibrated confidence statements, significantly improving the calibration of downstream human and
simulated forecasts while matching or exceeding strong RL-tuned baselines in accuracy.

7 Acknowledgements

We thank Michael Y. Li, Yann Dubois, Yu Sun, Zitong Yang, and members of the Tatsu Lab, Ma
Lab, Stanford Machine Learning Group, and Stanford NLP Group for their helpful feedback. This
work was supported by Open Philanthropy, IBM, NSF IIS 2211780, the Stanford HAI–Google Cloud

16

Credits Program, and the Anthropic Researcher Access Program. NB acknowledges funding from
an NSF Graduate Research Fellowship and a Quad Fellowship. XL acknowledges funding from a
Stanford Graduate Fellowship and a Meta PhD Fellowship.

17

References

Afra Feyza Akyürek, Ekin Akyürek, Leshem Choshen, Derry Wijaya, and Jacob Andreas. Deductive

closure training of language models for coherence, accuracy, and updatability, 2024.

Anthropic. Model card and evaluations for claude models, 2023.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones,
Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernan-
dez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark,
Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for
alignment, 2021.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna
Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness
from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

Neil Band, Tim G. J. Rudner, Qixuan Feng, Angelos Filos, Zachary Nado, Michael W Dusenberry,
Ghassen Jerfel, Dustin Tran, and Yarin Gal. Benchmarking bayesian deep learning on diabetic
retinopathy detection tasks. In NeurIPS Datasets and Benchmarks Track, 2021.

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight Uncertainty
in Neural Networks. In Francis Bach and David Blei, editors, PMLR, volume 37 of Proceedings of
Machine Learning Research, pages 1613–1622, Lille, France, 07–09 Jul 2015. PMLR.

Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge University Press, 2004.

Glenn W. Brier.

Verification of forecasts expressed in terms of probability. Monthly
Weather Review, 78(1):1–3, 1950.
doi: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;
2. URL https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_
1950_078_0001_vofeit_2_0_co_2.xml.

Jochen Bröcker. Reliability, sufficiency, and the decomposition of proper scores. Quarterly Journal
of the Royal Meteorological Society, 135(643):1512–1519, 2009. doi: https://doi.org/10.1002/qj.456.
URL https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.456.

Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang
Li, Zhengfu He, Kai Chen, and Xipeng Qiu. Can ai assistants know what they don’t know?, 2024.

Thomas M. Cover and Joy A. Thomas. Elements of Information Theory. Wiley, New York, 1991.

Jesse C. Cresswell, Yi Sui, Bhargava Kumar, and Noël Vouitsis. Conformal prediction sets improve

human decision making, 2024.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu,
and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E. Ho. Large legal fictions: Profiling legal

hallucinations in large language models, 2024.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and

memory-efficient exact attention with io-awareness, 2022.

18

A Philip Dawid. Present position and potential developments: Some personal views statistical
theory the prequential approach. Journal of the Royal Statistical Society: Series A (General), 147(2):
278–290, 1984.

Morris H. DeGroot and Stephen E. Fienberg. The comparison and evaluation of forecasters, 1983.

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise

quantization, 2022.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin,
Percy Liang, and Tatsunori Hashimoto. Alpacafarm: A simulation framework for methods that
learn from human feedback. In Thirty-seventh Conference on Neural Information Processing Systems,
2023. URL https://openreview.net/forum?id=4hturzLcKX.

Owain Evans, Owen Cotton-Barratt, Lukas Finnveden, Adam Bales, Avital Balwit, Peter Wills, Luca
Righetti, and William Saunders. Truthful ai: Developing and governing ai that does not lie, 2021.

Dean P. Foster and Rakesh Vohra. Regret in the on-line decision problem. Games and Economic
Behavior, 29(1):7–35, 1999. ISSN 0899-8256. doi: https://doi.org/10.1006/game.1999.0740. URL
https://www.sciencedirect.com/science/article/pii/S0899825699907406.

Dean P. Foster and Rakesh V. Vohra. Asymptotic calibration. Biometrika, 85(2):379–390, 1998. ISSN

00063444. URL http://www.jstor.org/stable/2337364.

Yarin Gal and Zoubin Ghahramani. Dropout As a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning. In Proceedings of the 33rd International Conference on International
Conference on Machine Learning - Volume 48, Icml 2016, pages 1050–1059, 2016.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Jour-
nal of the American Statistical Association, 102(477):359–378, 2007. doi: 10.1198/016214506000001437.
URL https://doi.org/10.1198/016214506000001437.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural
In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International
networks.
Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–
1330. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

Ursula Hebert-Johnson, Michael Kim, Omer Reingold, and Guy Rothblum. Multicalibration: Cal-
ibration for the (Computationally-identifiable) masses.
In Jennifer Dy and Andreas Krause,
editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Pro-
ceedings of Machine Learning Research, pages 1939–1948. PMLR, 10–15 Jul 2018. URL https:
//proceedings.mlr.press/v80/hebert-johnson18a.html.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions, 2023.

Yukun Huang, Yixin Liu, Raghuveer Thirukovalluru, Arman Cohan, and Bhuwan Dhingra. Cali-

brating long-form generations from large language models, 2024.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput.
Surv., 55(12), mar 2023. ISSN 0360-0300. doi: 10.1145/3571730. URL https://doi.org/10.
1145/3571730.

19

Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. How Can We Know When Language
Models Know? On the Calibration of Language Models for Question Answering. Transactions of
the Association for Computational Linguistics, 9:962–977, 09 2021. ISSN 2307-387X. doi: 10.1162/tacl_
a_00407. URL https://doi.org/10.1162/tacl_a_00407.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting
of the Association for Computational Linguistics, Vancouver, Canada, July 2017. Association for
Computational Linguistics.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk,
Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort,
Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt,
Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas
Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly)
know what they know, 2022.

Kaggle. 200,000+ jeopardy! questions, 2020. URL https://www.kaggle.com/datasets/

tunguz/200000-jeopardy-questions/data.

Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, and Georgios Paliouras. Bioasq-
qa: A manually curated corpus for biomedical question answering. Scientific Data, 10(1):170, Mar
2023. ISSN 2052-4463. doi: 10.1038/s41597-023-02068-4. URL https://doi.org/10.1038/
s41597-023-02068-4.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for

uncertainty estimation in natural language generation, 2023.

Meelis Kull and Peter Flach. Novel decompositions of proper scoring rules for classification: Score
adjustment as precursor to calibration.
In Annalisa Appice, Pedro Pereira Rodrigues, Vítor
Santos Costa, Carlos Soares, João Gama, and Alípio Jorge, editors, Machine Learning and Knowledge
Discovery in Databases, pages 68–85, Cham, 2015. Springer International Publishing. ISBN 978-3-
319-23528-8.

Meelis Kull, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter
Flach. Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with
dirichlet calibration. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and
R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates,
Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/
8ca01ea920679a0fe3728441494041b9-Paper.pdf.

Ananya Kumar, Percy S Liang, and Tengyu Ma. Verified uncertainty calibration. Advances in Neural

Information Processing Systems, 32, 2019.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles. In Isabelle Guyon, Ulrike von Luxburg, Samy
Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors,
Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information
Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 6402–6413, 2017.

20

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen,
Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario
Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen
Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid, Sylvain Gugger, Clément Delangue,
Théo Matussière, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar,
François Lagunas, Alexander Rush, and Thomas Wolf. Datasets: A community library for natural
language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language
Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic,
November 2021. Association for Computational Linguistics. URL https://aclanthology.
org/2021.emnlp-demo.21.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby
Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas,
Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu
Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun,
Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan
Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard,
Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta
Koreeda. Holistic evaluation of language models, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in

words, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference

on Learning Representations, 2019.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction.
In International Conference on Learning Representations, 2021. URL https://openreview.net/
forum?id=jN5y-zb5Q7m.

Andrey Malinin, Neil Band, Yarin Gal, Mark Gales, Alexander Ganshin, German Chesnokov, Alexey
Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, Vatsal Raina, Vyas Raina,
Denis Roginskiy, Mariya Shmatova, Panagiotis Tigas, and Boris Yangel. Shifts: A Dataset of
Real Distributional Shift Across Multiple Large-Scale Tasks. In Thirty-fifth Conference on Neural
Information Processing Systems Datasets and Benchmarks Track, 2021.

Sabrina J. Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. Reducing conversational agents’
overconfidence through linguistic calibration. Transactions of the Association for Computational
Linguistics, 10:857–872, 2022. doi: 10.1162/tacl_a_00494. URL https://aclanthology.org/
2022.tacl-1.50.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke
Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual
precision in long form text generation. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–
12100, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/
2023.emnlp-main.741. URL https://aclanthology.org/2023.emnlp-main.741.

Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby,
In

Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks.

21

M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Ad-
vances in Neural Information Processing Systems, volume 34, pages 15682–15694. Curran Associates,
Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/
8420d359404024567b5aefda1231af24-Paper.pdf.

Allan H Murphy. A new vector partition of the probability score. Journal of Applied Meteorology and

Climatology, 12(4):595–600, 1973.

Zachary Nado, Neil Band, Mark Collier, Josip Djolonga, Michael W. Dusenberry, Sebastian Farquhar,
Qixuan Feng, Angelos Filos, Marton Havasi, Rodolphe Jenatton, Ghassen Jerfel, Jeremiah Liu,
Zelda Mariet, Jeremy Nixon, Shreyas Padhy, Jie Ren, Tim G. J. Rudner, Faris Sbahi, Yeming Wen,
Florian Wenzel, Kevin Murphy, D. Sculley, Balaji Lakshminarayanan, Jasper Snoek, Yarin Gal, and
Dustin Tran. Uncertainty baselines: Benchmarks for uncertainty & robustness in deep learning,
2022.

Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised
learning. In Proceedings of the 22nd International Conference on Machine Learning, ICML ’05, page
625–632, New York, NY, USA, 2005. Association for Computing Machinery. ISBN 1595931805. doi:
10.1145/1102351.1102430. URL https://doi.org/10.1145/1102351.1102430.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni
Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor
Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum,
Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg
Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage,
Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory
Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason
Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings,
Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville,
Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna
Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella
Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha
Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross,
Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton,
Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton,
Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela
Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan,
Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan
Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt
Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic,
Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel
Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan
Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski,
Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine
McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz,
Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong
Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind
Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex
Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita,

22

Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael
Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong,
Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae,
Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri
Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather
Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica
Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin,
Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski
Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet,
Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón
Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang,
Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder,
Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah
Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin
Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao,
Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2023.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton,
Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and
Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL
https://arxiv.org/abs/2203.02155.

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon,
Balaji Lakshminarayanan, and Jasper Snoek. Can You Trust Your Model’s Uncertainty? Evaluating
Predictive Uncertainty Under Dataset Shift. In Advances in Neural Information Processing Systems 32.
2019.

Peter S. Park, Simon Goldstein, Aidan O’Gara, Michael Chen, and Dan Hendrycks. Ai deception: A

survey of examples, risks, and potential solutions, 2023.

John Platt et al. Probabilistic outputs for support vector machines and comparisons to regularized

likelihood methods. Advances in large margin classifiers, 10(3):61–74, 1999.

Leonard J. Savage. Elicitation of personal probabilities and expectations. Journal of the American
Statistical Association, 66(336):783–801, 1971. ISSN 01621459. URL http://www.jstor.org/
stable/2284229.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms, 2017.

Vaishnavi Shrivastava, Percy Liang, and Ananya Kumar. Llamas know what gpts don’t show:

Surrogate models for confidence estimation, 2023.

Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang
Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature Medicine, 29(8):1930–
1940, Aug 2023. ISSN 1546-170X. doi: 10.1038/s41591-023-02448-8. URL https://doi.org/10.
1038/s41591-023-02448-8.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea
Finn, and Christopher Manning. Just ask for calibration: Strategies for eliciting calibrated confi-
dence scores from language models fine-tuned with human feedback. In Houda Bouamor, Juan

23

Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing, pages 5433–5442, Singapore, December 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.emnlp-main.330. URL https://aclanthology.org/
2023.emnlp-main.330.

Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D Manning, and Chelsea Finn. Fine-tuning
language models for factuality. In The Twelfth International Conference on Learning Representations,
2024. URL https://openreview.net/forum?id=WPZ2yPag4K.

together.ai. Releasing 3b and 7b redpajama-incite family of models including base, instruction-tuned
& chat models, 2023. URL https://www.together.ai/blog/redpajama-models-v1.

Dustin Tran, Andreas Kirsch, Balaji Lakshminarayanan, Huiyi Hu, Du Phan, D. Sculley, Jasper Snoek,
Jeremiah Zhe Liu, Jie Ren, Joost van Amersfoort, Kehang Han, E. Kelly Buchanan, Kevin Patrick
Murphy, Mark Collier, Michael W Dusenberry, Neil Band, Nithum Thain, Rodolphe Jenatton,
Tim G. J. Rudner, Yarin Gal, Zachary Nado, Zelda E Mariet, Zi Wang, and Zoubin Ghahramani.
Plex: Towards reliability using pretrained large model extensions. In ICML 2022 Workshop on
Pre-training, 2022.

Thomas Wallsten. Measuring Vague Uncertainties and Understanding Their Use in Decision Making,
pages 377–398. Measuring Vague Uncertainties and Understanding Their Use in Decision Making,
01 1990. ISBN 978-90-481-5785-3. doi: 10.1007/978-94-015-7873-8_15.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models,
2022. URL https://arxiv.org/abs/2203.11171.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models. In The Eleventh International Conference on Learning Representations, 2023. URL https:
//openreview.net/forum?id=1PL1NIMMrw.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions,

2017.

Mark E. Whiting, Grant Hugh, and Michael S. Bernstein. Fair work: Crowd work minimum wage
with one line of code. Proceedings of the AAAI Conference on Human Computation and Crowdsourcing,
7(1):197–206, Oct. 2019. doi: 10.1609/hcomp.v7i1.5283. URL https://ojs.aaai.org/index.
php/HCOMP/article/view/5283.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,
Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von
Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama
Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language
processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Process-
ing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational
Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs
express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The
Twelfth International Conference on Learning Representations, 2024. URL https://openreview.
net/forum?id=gjeQKFxFpZ.

24

Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, and Pengfei Liu. Alignment for honesty,

2023.

Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue Wang, Derek F. Wong, Emine Yilmaz, Shuming

Shi, and Zhaopeng Tu. Benchmarking llms via uncertainty quantification, 2024.

Bianca Zadrozny and Charles Elkan. Obtaining calibrated probability estimates from decision trees

and naive bayesian classifiers. In Icml, volume 1, pages 609–616, 2001.

Shengjia Zhao and Stefano Ermon. Right decisions from wrong predictions: A mechanism design
alternative to individual calibration. In Arindam Banerjee and Kenji Fukumizu, editors, Pro-
ceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130
of Proceedings of Machine Learning Research, pages 2683–2691. PMLR, 13–15 Apr 2021. URL
https://proceedings.mlr.press/v130/zhao21a.html.

Shengjia Zhao, Michael Kim, Roshni Sahoo, Tengyu Ma, and Stefano Ermon. Calibrat-
ing predictions to decisions: A novel approach to multi-class calibration.
In M. Ran-
zato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances
in Neural Information Processing Systems, volume 34, pages 22313–22324. Curran Associates,
Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/
bbc92a647199b832ec90d7cf57074e9e-Paper.pdf.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid
Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard
Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on
scaling fully sharded data parallel, 2023.

Kaitlyn Zhou, Dan Jurafsky, and Tatsunori Hashimoto. Navigating the grey area: How expressions of
uncertainty and overconfidence affect language models. In Houda Bouamor, Juan Pino, and Kalika
Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 5506–5524, Singapore, December 2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.emnlp-main.335. URL https://aclanthology.org/2023.emnlp-main.
335.

25

Supplementary Material

Table of Contents

A Additional Results

A.1 Codebase . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.2 Additional Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.3 TriviaQA: Full Accuracy-ECE Frontier . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.4 TriviaQA: Additional Reliability Diagrams . . . . . . . . . . . . . . . . . . . . . . . .
A.5 Jeopardy: Full Accuracy-ECE Frontier . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.6 Jeopardy: All Reliability Diagrams . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.7 SciQ: Full Accuracy-ECE Frontier . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.8 SciQ: All Reliability Diagrams . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.9 BioASQ Task B: Full Accuracy-ECE Frontier . . . . . . . . . . . . . . . . . . . . . . . .
A.10 BioASQ Task B: All Reliability Diagrams . . . . . . . . . . . . . . . . . . . . . . . . . .
A.11 Person Biography Generation Frontier . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
A.12 Tabular Results
A.13 Qualitative Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

B Benefits of Linguistic Calibration for Decision-making

B.1 Review of the LC Objective
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
B.2 Decision Calibration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
B.3 Linguistic Calibration and Optimal Decision-making . . . . . . . . . . . . . . . . . .

C Training Framework

C.1 Regularized Linguistic Calibration Objective . . . . . . . . . . . . . . . . . . . . . . .
C.2 Proof: Regularized Objective is Strictly Proper . . . . . . . . . . . . . . . . . . . . . .
C.3 Additional Details on Training Framework . . . . . . . . . . . . . . . . . . . . . . . .

D Evaluation Framework

D.1 Simulated Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
D.2 FactScore-Based Evaluation Metric . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
D.3 Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

27
27
27
28
29
30
31
32
33
34
35
36
37
39

44
44
44
46

48
48
48
50

53
53
55
56

26

A Additional Results

A.1 Codebase

For prompts, checkpoints, and implementations of methods and datasets, please refer to our
codebase:

https://github.com/tatsu-lab/linguistic_calibration

A.2 Additional Baselines

In addition to the baselines described in §4.1, we below provide full results including several
other baselines. All references to GPT-4 in this paper use version gpt-4-1106-preview. Unless
otherwise specified, all references to Claude or Claude 2 use version claude-2.0. Due to compute
constraints, all GPT-4 baselines are evaluated on a smaller number of question-answering test
samples: 1000 samples for TriviaQA, Jeopardy, and SciQ, and the entire 1515 sample test set for
BioASQ Task B (cf. Appendix D for further dataset and evaluation details).
We include the following additional non-confidence baselines:

• Llama 2 Chat. We zero-shot prompt Llama 2 Chat to generate long-form responses to evaluation

queries.

• GPT-4 0-Shot. We zero-shot prompt GPT-4 to generate long-form responses to evaluation

queries.

• GPT-4 ICL 8-Shot. Analogous to the Llama 2 7B ICL baseline. We randomly sample 8
queries, generate long-form responses with GPT-4, manually fact-check those responses using
Wikipedia, and use these fact-checked (query, response) pairs as ICL examples for GPT-4.

We also include the following additional confidence baselines:

• GPT-4 Just Ask for Uncertainty (JAFU) 0-Shot. We zero-shot prompt GPT-4 to generate long-
form responses to evaluation queries, and include an instruction in the prompt directing GPT-4
to indicate any uncertainty in its claims using probabilities.

• GPT-4 Summary ICL 8-Shot. Analogous to the Llama 2 7B Summary ICL baseline. We
sample 8 queries from the prompt development split {q(i)}8
i=1 (i.e., questions x which have been
converted to open-ended queries). For each query q(i), we generate 8 long-form responses from
GPT-4: {z(i)
j=1 ∼ πGPT-4(z | q(i)). We then summarize these responses into a single consensus
response: s(i) ∼ Summarize(s | z(i)
8 ). Finally, we use the queries and summaries as ICL
examples {(q(i), s(i))}8

1 , . . . , z(i)

j }8

i=1.

• Direct Summary Eval. For a given evaluation query q, we use the Llama 2 7B ICL baseline
to generate 8 long-form responses {zi}8
i=1, and use Claude 2 to summarize these responses
into a single consensus response: s ∼ Summarize(s | z1, . . . , z8). Then, we directly evaluate
these summaries s. This is an oracle baseline because it requires sampling several long-
form generations from the base LM at evaluation time, followed by summarization with an
API-based LLM.

27

A.3 TriviaQA: Full Accuracy-ECE Frontier

Figure 8: Accuracy-ECE Frontier for All Methods on TriviaQA, evaluated with simulated reader
(upper left is better). GPT-4 Summary ICL works well despite its simplicity. LC RL outperforms
all methods in ECE, including GPT-4 Summary ICL and Direct Summary Eval. Llama 2 Chat 7B
performs significantly worse than all other methods and is not displayed for clarity (cf. Table 2 for
numerical results).

28

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)Summary ICL 8-ShotGPT-4 JAFU 0-ShotGPT-4 Summary ICL 8-ShotDirect Summary EvalNon-Confidence BaselinesFactuality RLFactuality SFTClaude DistillICL 8-ShotLlama 2 Chat 7BGPT-4 0-ShotGPT-4 ICL 8-Shot0.100.150.200.250.300.350.40ECE0.600.650.700.750.800.85AccuracyA.4 TriviaQA: Additional Reliability Diagrams

(a) GPT-4 JAFU 0-Shot. ECE: 0.212

(b) GPT-4 Summary ICL 8-Shot. ECE: 0.119

(c) Summary ICL 8-Shot. ECE: 0.257

(d) Direct Summary Eval. ECE: 0.112

Figure 9: TriviaQA Reliability Diagrams for additional baselines and oracle methods, simulated
reader. Notably, Summary ICL works well for both Llama 2 7B (Summary ICL 8-Shot) and GPT-4.
The oracle baseline of Direct Summary Eval has strong ECE.

29

0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00AccuracyA.5

Jeopardy: Full Accuracy-ECE Frontier

Figure 10: Accuracy-ECE Frontier for All Methods on Jeopardy, evaluated with simulated reader
(upper left is better). Results are similar to TriviaQA. GPT-4 Summary ICL works well despite
its simplicity. LC RL outperforms all methods in ECE, including GPT-4 Summary ICL and Direct
Summary Eval. Llama 2 Chat 7B performs significantly worse than all other methods and is not
displayed for clarity (cf. Table 2 for numerical results).

30

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)Summary ICL 8-ShotGPT-4 JAFU 0-ShotGPT-4 Summary ICL 8-ShotDirect Summary EvalNon-Confidence BaselinesFactuality RLFactuality SFTClaude DistillICL 8-ShotLlama 2 Chat 7BGPT-4 0-ShotGPT-4 ICL 8-Shot0.100.150.200.250.300.350.40ECE0.600.650.700.750.800.85AccuracyA.6

Jeopardy: All Reliability Diagrams

(a) LC SFT (πSFT). ECE: 0.162

(b) LC RL (πRL). ECE: 0.088

(c) GPT-4 JAFU 0-Shot. ECE: 0.207

(d) GPT-4 Summary ICL 8-Shot. ECE: 0.130

(e) Summary ICL 8-Shot. ECE: 0.254

(f) Direct Summary Eval. ECE: 0.104

Figure 11: Jeopardy Reliability Diagrams for all methods, simulated reader. LC RL has better ECE
than all other methods while predicting a variety of confidence levels. Notably, Summary ICL works
well for both Llama 2 7B (Summary ICL 8-Shot) and GPT-4. The oracle baseline of Direct Summary
Eval has strong ECE.

31

0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00AccuracyA.7 SciQ: Full Accuracy-ECE Frontier

Figure 12: Accuracy-ECE Frontier for All Methods on SciQ, evaluated with simulated reader
(upper left is better). GPT-4 Summary ICL works well despite its simplicity. LC RL outperforms
all methods in ECE, including GPT-4 Summary ICL and Direct Summary Eval. Llama 2 Chat 7B
performs significantly worse than all other methods and is not displayed for clarity (cf. Table 2 for
numerical results).

32

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)Summary ICL 8-ShotGPT-4 JAFU 0-ShotGPT-4 Summary ICL 8-ShotDirect Summary EvalNon-Confidence BaselinesFactuality RLFactuality SFTClaude DistillICL 8-ShotLlama 2 Chat 7BGPT-4 0-ShotGPT-4 ICL 8-Shot0.200.250.300.350.400.45ECE0.550.600.650.700.75A.8 SciQ: All Reliability Diagrams

(a) LC SFT (πSFT). ECE: 0.313

(b) LC RL (πRL). ECE: 0.213

(c) GPT-4 JAFU 0-Shot. ECE: 0.304

(d) GPT-4 Summary ICL 8-Shot. ECE: 0.225

(e) Summary ICL 8-Shot. ECE: 0.324

(f) Direct Summary Eval. ECE: 0.295

Figure 13: SciQ Reliability Diagrams for all methods, simulated reader. LC RL has better ECE than
all other methods while predicting a variety of confidence levels.

33

0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00AccuracyA.9 BioASQ Task B: Full Accuracy-ECE Frontier

Figure 14: Accuracy-ECE Frontier for All Methods on BioASQ Task B, evaluated with simulated
reader (upper left is better). Under a significant distribution shift, LC RL ECE is competitive with
GPT-4–based methods and outperforms all Llama-based baselines. BioASQ is a small expert-
annotated dataset, so we report Student’s t 0.95 CIs over 10 evaluation seeds for non-GPT-4 methods.
Llama 2 Chat 7B performs significantly worse than all other methods and is not displayed for clarity
(cf. Table 2 for all numerical results).

34

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)Summary ICL 8-ShotGPT-4 JAFU 0-ShotGPT-4 Summary ICL 8-ShotDirect Summary EvalNon-Confidence BaselinesFactuality RLFactuality SFTClaude DistillICL 8-ShotLlama 2 Chat 7BGPT-4 0-ShotGPT-4 ICL 8-Shot0.300.350.400.450.500.550.60ECE0.400.450.500.550.60A.10 BioASQ Task B: All Reliability Diagrams

(a) LC SFT (πSFT). ECE: 0.390

(b) LC RL (πRL). ECE: 0.342

(c) GPT-4 JAFU 0-Shot. ECE: 0.356

(d) GPT-4 Summary ICL 8-Shot. ECE: 0.281

(e) Summary ICL 8-Shot. ECE: 0.477

(f) Direct Summary Eval. ECE: 0.375

Figure 15: BioASQ Task B Reliability Diagrams for all methods, simulated reader. Under significant
distribution shift, LC RL has ECE competitive with GPT-4 methods while predicting a variety of
confidence levels. BioASQ is a small expert-annotated dataset, so for non-GPT-4 methods, we pool
examples across 10 evaluation seeds when plotting the reliability diagram (per-seed plots in gray).

35

0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00Accuracy0.000.250.500.751.00Confidence0.000.250.500.751.00AccuracyA.11 Person Biography Generation Frontier

Figure 16: Accuracy-ECE Frontier for All Methods on Person Biography Generation, evaluated
with simulated reader (upper left is better). GPT-4 is a strong baseline, and GPT-4 Just Ask for
Uncertainty (JAFU) does improve both accuracy and ECE. Notably, LC RL has better ECE than
GPT-4 0-Shot, despite being trained from a Llama 2 7B base model.

36

Our MethodLC RL (LC)Confidence BaselinesLC SFT (SFT)GPT-4 JAFU 0-ShotNon-Confidence BaselinesFactuality RLFactuality SFTClaude DistillLlama 2 Chat 7BGPT-4 0-Shot0.20.30.40.50.6ECE over Claims0.40.50.60.7Accuracy over ClaimsA.12 Tabular Results

Table 2: Simulated Question-Answering Results. Accuracy and expected calibration error (ECE) of Llama 2 7B and GPT-4–based
methods on the in-distribution (ID) TriviaQA and out-of-distribution (OOD) Jeopardy, SciQ, and BioASQ Task B question-
answering datasets, with simulated reader evaluation. “API-Based LLM Methods” are those that use either GPT-4 or Claude 2
at test time. CIs are reported as (lower, upper). For all datasets other than BioASQ, these are 95% bootstrap CIs; for the smaller
BioASQ dataset, these are Student’s t 0.95 CIs over 10 evaluation seeds. Due to compute constraints, for GPT-4–based methods on
BioASQ, we report accuracy and ECE for a single evaluation seed (cf. Appendix D for all evaluation details).

TriviaQA (ID)

Jeopardy (OOD)

SciQ (OOD)

BioASQ Task B (OOD)

Method

Accuracy (%) ↑

ECE ↓

Accuracy (%) ↑

ECE ↓

Accuracy (%) ↑

ECE ↓

Accuracy (%) ↑

ECE ↓

Llama 2 7B–Based Methods

Llama 2 Chat 7B
Claude Distill
ICL 8-Shot
Factuality SFT
Factuality RL
Summary ICL 8-Shot
LC SFT (πSFT)
LC RL (πLC)

52.32 (51.42, 53.27)
60.89 (59.98, 61.78)
61.91 (61.02, 62.81)
61.17 (60.26, 62.07)
63.33 (62.44, 64.21)
61.36 (60.47, 62.25)
60.98 (60.10, 61.88)
64.74 (63.86, 65.63)

0.477 (0.467, 0.486)
0.391 (0.382, 0.400)
0.381 (0.372, 0.390)
0.388 (0.379, 0.397)
0.367 (0.358, 0.376)
0.257 (0.248, 0.267)
0.166 (0.158, 0.176)
0.108 (0.101, 0.117)

50.86 (49.91, 51.82)
60.54 (59.64, 61.44)
61.63 (60.69, 62.57)
62.13 (61.21, 63.05)
64.05 (63.14, 64.97)
60.90 (59.98, 61.83)
62.46 (61.53, 63.36)
65.73 (64.83, 66.61)

0.491 (0.482, 0.501)
0.395 (0.386, 0.404)
0.384 (0.374, 0.393)
0.379 (0.370, 0.388)
0.359 (0.350, 0.369)
0.254 (0.245, 0.263)
0.162 (0.154, 0.172)
0.088 (0.082, 0.098)

51.52 (50.68, 52.34)
57.00 (56.17, 57.83)
55.10 (54.27, 55.92)
55.08 (54.26, 55.92)
56.11 (55.27, 56.94)
54.76 (53.91, 55.61)
54.87 (54.04, 55.69)
56.85 (56.02, 57.67)

0.485 (0.477, 0.493)
0.430 (0.422, 0.438)
0.449 (0.441, 0.457)
0.449 (0.441, 0.457)
0.439 (0.431, 0.447)
0.324 (0.316, 0.333)
0.313 (0.306, 0.323)
0.213 (0.205, 0.222)

35.31 (34.79, 35.82)
39.78 (39.30, 40.27)
37.72 (37.09, 38.35)
37.54 (37.22, 37.87)
38.04 (37.66, 38.42)
38.15 (37.77, 38.53)
38.53 (38.19, 38.87)
38.89 (38.39, 39.39)

0.647 (0.642, 0.652)
0.602 (0.597, 0.607)
0.623 (0.617, 0.629)
0.625 (0.621, 0.628)
0.620 (0.616, 0.623)
0.476 (0.471, 0.480)
0.389 (0.384, 0.394)
0.342 (0.335, 0.350)

API-Based LLM Methods

GPT-4 0-Shot
GPT-4 ICL 8-Shot
GPT-4 JAFU 0-Shot
GPT-4 Summary ICL 8-Shot
Direct Summary Eval

78.00 (75.30, 80.50)
80.80 (78.40, 83.20)
77.70 (75.10, 80.20)
80.40 (77.90, 82.80)
63.67 (62.78, 64.55)

0.220 (0.195, 0.247)
0.192 (0.168, 0.216)
0.212 (0.187, 0.238)
0.119 (0.104, 0.148)
0.112 (0.105, 0.120)

77.60 (75.00, 80.10)
82.30 (80.00, 84.70)
78.00 (75.40, 80.50)
80.30 (77.80, 82.70)
64.50 (63.59, 65.43)

0.224 (0.199, 0.250)
0.177 (0.153, 0.200)
0.207 (0.182, 0.234)
0.130 (0.111, 0.156)
0.104 (0.098, 0.114)

66.20 (63.20, 69.10)
71.60 (68.80, 74.50)
68.50 (65.50, 71.40)
70.50 (67.70, 73.30)
57.89 (57.04, 58.73)

0.338 (0.309, 0.368)
0.284 (0.255, 0.312)
0.304 (0.274, 0.334)
0.225 (0.197, 0.253)
0.295 (0.286, 0.303)

62.11
63.43
62.31
61.39
40.42 (39.91, 40.93)

0.379
0.366
0.356
0.281
0.374 (0.367, 0.380)

Table 3: Human Question-Answering Results on TriviaQA. Accuracy and expected calibration
error (ECE) of core Llama 2 7B–based methods on the TriviaQA question-answering dataset, with
human readers (cf. Appendix D for evaluation framework details). 95% bootstrap CIs are reported
as (lower, upper).

Method

Accuracy (%) ↑

ECE ↓

Factuality RL
LC SFT (πSFT)
LC RL (πLC)

59.62 (56.65, 62.60)
57.44 (54.37, 60.52)
60.12 (57.14, 63.19)

0.404 (0.374, 0.434)
0.163 (0.135, 0.192)
0.116 (0.091, 0.145)

Table 4: Person Biography Generation Results. Accuracy and expected calibration error (ECE)
of Llama 2 7B and GPT-4–based methods on the Person Biography Generation dataset, using a
FactScore-based automated metric (Min et al., 2023). Metrics are computed over atomic claims
pooled across all biography generations. 95% bootstrap CIs are reported as (lower, upper).

Method

Accuracy (%) ↑

ECE ↓

Llama 2 7B–Based Methods

Llama 2 Chat 7B
Claude Distill
Factuality SFT
Factuality RL
LC SFT (πSFT)
LC RL (πLC)

36.27 (35.18, 37.36)
39.24 (38.13, 40.34)
38.07 (37.01, 39.13)
39.86 (38.72, 40.98)
44.49 (43.42, 45.57)
46.77 (45.50, 48.08)

0.637 (0.626, 0.648)
0.608 (0.597, 0.619)
0.619 (0.609, 0.630)
0.601 (0.590, 0.613)
0.301 (0.292, 0.314)
0.266 (0.253, 0.280)

GPT-4–Based Methods

GPT-4 0-Shot
GPT-4 JAFU 0-Shot

71.84 (70.88, 72.81)
75.95 (74.93, 76.93)

0.282 (0.272, 0.291)
0.210 (0.200, 0.220)

38

A.13 Qualitative Examples

We randomly sample two examples from each evaluation dataset and compare Factuality RL with LC
RL. We manually fact-check each generation using Wikipedia and other Google-accessible sources
and highlight incorrect statements in red. Numerical and linguistic confidence statements are
highlighted in blue.

Dataset: TriviaQA
Query q: Write a paragraph about the classic book in which a boy hides in a ship’s barrel of apples.
Ground-Truth Answer y: Treasure Island

Factuality RL: The classic book in which a boy hides in a ship’s barrel of apples is "Treasure Island," written by
Robert Louis Stevenson. The story follows the adventures of Jim Hawkins, a young boy who lives in a inn on
an island off the coast of England. One day, a mysterious sailor named Long John Silver arrives at the inn, and
Jim becomes involved in a treasure hunt for a buried treasure on the island. Jim hides in a barrel of apples on the
ship that is carrying the treasure, and he is able to escape with the treasure when the ship is attacked by pirates.
"Treasure Island" is a classic tale of adventure and intrigue, and it has been adapted into numerous films and stage
productions.

LC RL: Based on my understanding, the classic book in which a boy hides in a ship’s barrel of apples is most
likely "Treasure Island" by Robert Louis Stevenson. I estimate there is a 75% chance it is "Treasure Island", a 15%
chance it is "The Wonderful Wizard of Oz" by L. Frank Baum, a 5% chance it is "The Adventures of Tom Sawyer"
by Mark Twain, and a 5% chance it is another book. I believe the story involves a young boy named Jim Hawkins
who hides in a barrel of apples to avoid being discovered by pirates. Though I do not know all the details, I am
confident the boy’s adventures involve searching for buried treasure. Regardless of the specific book, I am certain
this classic tale has become beloved by generations of readers.

Dataset: TriviaQA
Query q: Write a paragraph about the type of creature that a tanager is.
Ground-Truth Answer y: Bird

Factuality RL: A tanager is a type of bird that belongs to the Thraupidae family. Tanagers are known for their
brightly colored plumage, with many species featuring vibrant red, yellow, and blue feathers. They are found
in a wide range of habitats, including forests, savannas, and grasslands, and are known for their acrobatic flight
patterns. Tanagers are social birds that often form flocks, and they are known for their loud and distinctive calls.

LC RL: I believe a tanager is a type of bird that belongs to the Thraupidae family. I am confident they are found in
the Americas, though I am uncertain of their exact range. I estimate with 75% likelihood that tanagers are brightly
colored birds with distinctive plumage. I am 50% sure they are known for their songbird abilities. I am uncertain
whether they are found in forests, grasslands, or other habitats. I believe with 75% likelihood that tanagers feed on
insects and fruit. While the details of their appearance and behavior are unclear to me, I am certain that tanagers
are colorful, songbirds native to the Americas.

39

Dataset: SciQ
Query q: Write a paragraph about the term used to describe elements that specifically are noted to have properties of both
metals and nonmetals.
Ground-Truth Answer y: Metalloids

Factuality RL: The term used to describe elements that specifically are noted to have properties of both metals and
nonmetals is "metalloids." Metalloids are elements that have some of the properties of metals, such as conductivity
and malleability, and some of the properties of nonmetals, such as a tendency to form covalent bonds. Examples
of metalloids include silicon, germanium, and arsenic.

LC RL: Based on my understanding, the term used to describe elements that specifically are noted to have
properties of both metals and nonmetals is either "metalloids", "semimetals", or "semiconductors". I estimate
there is a 70% likelihood that the term is "metalloids", a 20% chance it is "semimetals", and a 10% chance it is
"semiconductors". I believe these elements have properties that are somewhere between metals and nonmetals,
though I do not know all the details. I am confident that they are used in various applications like electronics,
batteries, and semiconductors. However, I am uncertain about the specific elements that fall into this category.
Overall, despite my lack of complete knowledge, I am sure that there is a term used to describe elements with
properties of both metals and nonmetals.

Dataset: SciQ
Query q: Write a paragraph about the periods when most plants do not grow continuously.
Ground-Truth Answer y: Dormant

Factuality RL: The periods when most plants do not grow continuously are known as dormancy periods. During
these periods, plants enter a state of inactivity, where they stop growing and may even shed leaves or go into
a state of hibernation. Dormancy periods can be caused by various factors, such as cold temperatures, lack of
sunlight, or insufficient water. Plants may also enter dormancy periods to conserve energy and resources during
times of stress or drought. Dormancy periods are an important part of the plant’s life cycle, allowing them to
adapt to changing environmental conditions and survive through challenging times.

LC RL: Based on my understanding, the periods when most plants do not grow continuously are known as
dormant periods. I am confident that dormancy occurs in response to environmental factors like cold temperatures,
lack of sunlight, or limited water availability. However, I am uncertain about the exact duration of dormancy for
different plant species. I estimate there is a 50% chance that dormancy lasts for several months, and a 50% chance
it lasts for several years. I believe dormancy allows plants to conserve energy and resources, though I do not
know all the details. While dormancy may seem like a state of inactivity, I am certain it is an important adaptive
strategy for plants to survive harsh conditions. Overall, despite my lack of complete knowledge, I am confident
that dormancy is a natural part of plant growth cycles.

40

Dataset: BioASQ Task B
Query q: Write a paragraph about the gene that is most commonly mutated in Tay-Sachs disease.
Ground-Truth Answer y: HEXA

Factuality RL: The gene that is most commonly mutated in Tay-Sachs disease is the HEXA gene, which encodes
the alpha subunit of the hexosaminidase A enzyme. This enzyme is responsible for breaking down the sugar
hexosamine, which is found in the membranes of cells. When the HEXA gene is mutated, the alpha subunit of
the hexosaminidase A enzyme is not produced, leading to a buildup of hexosamine in the cells. This buildup
can cause neurological damage and eventually lead to death. Tay-Sachs disease is a rare genetic disorder that
primarily affects Ashkenazi Jews and is caused by a mutation in the HEXA gene.

LC RL: Based on my understanding, the gene that is most commonly mutated in Tay-Sachs disease is either the
HEXA gene or the GAA gene. I estimate there is a 70% likelihood that the HEXA gene is mutated, and a 30%
chance that the GAA gene is mutated. I believe the HEXA gene encodes an enzyme called beta-hexosaminidase A,
while the GAA gene encodes an enzyme called beta-glucuronidase. I am confident that mutations in these genes
lead to a deficiency of these enzymes, which causes accumulation of toxic substances in the nervous system and
other tissues. This results in neurological symptoms like abnormal eye movements, seizures, and mental decline.
While I do not know all the details, I believe Tay-Sachs disease is a rare genetic disorder that primarily affects
Ashkenazi Jews.

Dataset: BioASQ Task B
Query q: Write a paragraph about the type of GTPases that are required for amino acid-dependent activation of mTORC1.
Ground-Truth Answer y: Heterodimeric Rag GTPases

Factuality RL: The type of GTPases that are required for amino acid-dependent activation of mTORC1 are
rapamycin-sensitive GTPases. These GTPases are involved in the regulation of cell growth and proliferation, and
are activated by amino acids, such as leucine, isoleucine, and valine. The activation of these GTPases leads to the
phosphorylation of the mTORC1 complex, which in turn activates downstream signaling pathways that regulate
cell growth and proliferation. The activation of mTORC1 by amino acids is a crucial step in the regulation of cell
growth and proliferation, and is a key target for the development of therapeutics for various diseases, including
cancer and diabetes.

LC RL: Based on my understanding, the type of GTPases required for amino acid-dependent activation of mTORC1
are either mTORC1 itself, Rheb, or Rag GTPases. I estimate there is a 70% likelihood that mTORC1 itself is the
GTPase required, a 20% chance that Rheb is the GTPase required, and a 10% chance that Rag GTPases are the
GTPases required. I believe the activation of mTORC1 by amino acids involves the binding of amino acids to Rag
GTPases, which then activates mTORC1. However, I am uncertain about the exact details of how this process
works. Regardless of the specific GTPase involved, I am confident that amino acid activation of mTORC1 plays a
crucial role in regulating cell growth, metabolism, and protein synthesis.

41

Dataset: Jeopardy
Query q: Write a paragraph about what Hiram Percy Maxim invented for guns, in addition to inventing the muffler for
cars.
Ground-Truth Answer y: A silencer

Factuality RL: Hiram Percy Maxim, an American inventor and entrepreneur, is best known for his contributions
to the firearms industry. In addition to inventing the muffler for cars, Maxim also invented the Maxim machine
gun, a rapid-fire weapon that revolutionized warfare in the early 20th century. The Maxim machine gun was the
first fully automatic weapon to be widely used in combat, and it played a significant role in the Allied victory in
World War I. Maxim’s invention of the Maxim machine gun earned him the nickname "the father of the machine
gun."

LC RL: I believe that Hiram Percy Maxim, an American inventor, invented at least one significant invention related
to guns in addition to inventing the muffler for cars. Based on my understanding, I estimate there is a 50% chance
he invented the Maxim gun, a 25% chance he invented the silencer, and a 25% chance he invented the machine
gun. I am confident that regardless of the specific invention, Maxim’s work revolutionized the firearms industry
and had a significant impact on warfare. Though I do not know all the details, I believe his inventions were known
for their efficiency, accuracy, and durability. While the specifics are unclear to me, I am certain that Hiram Percy
Maxim was an influential American inventor who made important contributions to the development of firearms
technology.

Dataset: Jeopardy
Query q: Write a paragraph about the planet that a probe from the Galileo spacecraft plunged into the atmosphere of on
December 7, 1995.
Ground-Truth Answer y: Jupiter

Factuality RL: The planet that a probe from the Galileo spacecraft plunged into the atmosphere of on December
7, 1995, was Jupiter. The Galileo probe was launched by NASA in 1989 and arrived at Jupiter in 1995, where
it conducted numerous scientific studies of the planet and its moons. On December 7, 1995, the probe was
intentionally crashed into Jupiter’s atmosphere, where it collected data on the planet’s atmosphere and magnetic
field. The Galileo probe’s mission was a significant achievement in the study of Jupiter and the solar system, and
its impact on the field of planetary science is still felt today.

LC RL: I am certain that on December 7, 1995, a probe from the Galileo spacecraft plunged into the atmosphere of
Jupiter. I believe this was a major milestone in space exploration, as it was the first time a spacecraft had been
intentionally sent to study Jupiter up close. I estimate with 90% confidence that the probe was named Galileo after
the Italian astronomer Galileo Galilei. I am fairly confident that the probe carried instruments to study Jupiter’s
atmosphere, magnetic field, and moons. Though I do not know all the details, I believe the data collected by
the probe helped advance our understanding of Jupiter and its moons. While the specifics are unclear to me, I
am certain that the Galileo probe’s plunge into Jupiter’s atmosphere marked an important achievement in space
exploration.

42

Dataset: Person Biography Generation
Query q: Write a paragraph bio about Griselda Blanco.

Factuality RL: Griselda Blanco, also known as "The Black Widow," was a Colombian drug lord and crime boss
who operated in the 1970s and 80s. She was known for her ruthlessness and her involvement in the cocaine
trade, which made her one of the most powerful drug lords in the world. Blanco was responsible for the murder
of numerous rivals and enemies, and her criminal activities led to her being listed as one of the FBI’s Ten Most
Wanted Fugitives in 1985. Despite her criminal activities, Blanco was also known for her philanthropy, and she
was involved in various charitable organizations in Colombia.

LC RL: I believe Griselda Blanco was a Colombian drug lord who operated in the 1970s and 1980s. I estimate
with 75% confidence that she was known as the "Black Widow" due to her ruthlessness and alleged involvement
in numerous murders. I am fairly certain she was involved in cocaine trafficking and organized crime, though
I am uncertain about the specifics of her operations. I believe with 50% likelihood that she was born in 1943 in
Colombia. I am confident she was arrested multiple times and imprisoned, though I do not know the details.
While the specifics are unclear to me, I am certain that Griselda Blanco was a notorious drug lord who left a lasting
legacy of violence and crime in Colombia and the United States.

Dataset: Person Biography Generation
Query q: Write a paragraph bio about Uee.

Factuality RL: Uee, also known as Lee Hye-young, is a South Korean singer and actress. She is best known for
her work in the K-pop girl group After School, where she served as the main vocalist and lead dancer. Uee has
also appeared in several television dramas, including "The Greatest Love," "The Greatest Love," and "The Greatest
Love." In addition to her musical and acting career, Uee is also known for her philanthropic work, including her
participation in the "2014 Samsung Hope for Children" charity event.

LC RL: I am confident that Uee is a South Korean actress and singer. I estimate with 75% likelihood that she was
born in 1988 in Seoul, South Korea. I believe with 50% probability that she made her acting debut in 2007 in the
television drama "Love Needs a Miracle." I am uncertain whether she has appeared in films like "The Concubine"
or "The King’s Case Note." I am confident that she has been a member of the girl groups After School and Orange
Caramel. I believe with 75% likelihood that she has won awards for her acting and singing, though I do not know
the specifics. Overall, despite my lack of knowledge on some details, I am certain that Uee is a talented South
Korean actress and singer who has had a successful career in the entertainment industry.

43

B Benefits of Linguistic Calibration for Decision-making

In §2, we describe the connection between calibration and decision-making (Zhao and Ermon,
2021, Zhao et al., 2021) and how it motivates our objective for linguistic calibration: namely, that
maximizing our objective implies zero expected regret for a downstream Bayes-optimal decision-
maker. Here, we expand on this guarantee and provide additional ones using the results of Zhao
et al. (2021), which apply even if we cannot perfectly maximize the linguistic calibration objective.
In particular, we will see that forecasts fulfilling weaker notions of calibration still guarantee that
decisions are optimal in a weaker sense.
Following the generative process described in §2.1, we assume that the linguistic calibration objective
is optimized over the real-world distribution of LM queries and related question-answer pairs
p(q, x, y). In practice, we found that using off-the-shelf question-answering datasets to generate a
surrogate distribution (cf. §3.1) robustly improves an LM’s ability to express confidence levels in
text, including under distribution shift. Therefore, the generalization properties of LC may justify
this assumption. Moreover, we believe an exciting avenue for future work is to curate a more
representative question-answer distribution (cf. §6). Lastly, recall that our generative process makes
the following technical conditional independence assumption:

Assumption B.1. Under the ground-truth distribution, the answer Y is independent of the open-ended LM
query and LM response (Q, Z), given the question X: Y ⊥ (Q, Z) | X.

B.1 Review of the LC Objective

Our decision-based RL algorithm optimizes a strictly proper scoring rule RLC of user forecasts
f (x, z) ∈ ∆|Y| with respect to an LM πRL producing long-form generations z ∼ πRL(z | q) (restate-
ment of Eq. 2 for convenience):
max
πRL

E
(q,x,y)∼p(q,x,y), z∼πRL(z|q)

[RLC (f (x, z), y)] .

(B.1)

Because RLC is strictly proper (cf. §2.3) and assuming a sufficiently flexible LM πRL and reader
function f : X × Z → ∆|Y|, our training procedure maximizes Eq. B.1 if and only if the user’s
forecast exactly equals the ground-truth conditional distribution: f (x, z) = p(y | x). Then, intuitively,
making Bayes-optimal decisions according to the user forecast should be optimal in some sense.
Zhao et al. (2021) make this notion of optimal decision-making concrete as decision calibration.
To be self-contained, we will provide the definition of decision calibration from Zhao et al. (2021)
below, and then use it to precisely describe the guarantees that linguistic calibration provides for
decision-making.

B.2 Decision Calibration

Setup and notation. To define decision calibration, we first introduce some notation closely
following Zhao et al. (2021), §2. In the process of decision-making based on probabilistic forecasts
(ignoring LMs and their generations z for now), users receive a question x′ ∈ X ′, forecast a possible
answer to the question with a probabilistic forecaster f : X ′ → ∆|Y|, and finally choose from a set of
available actions A based on their loss function L : A × Y → R and their forecast f (x′) ∈ ∆|Y|.
The process of choosing an action can be described by a decision rule which maps forecasts to
actions δ : ∆|Y| → A, where ∆all is the set of all decision rules. Additionally, define the set of all loss
functions as Lall = {L : A × Y → R}.
We are interested in Bayes decision rules, i.e., the set of rules that are optimal for some loss function.
More specifically, consider some particular loss function L ∈ Lall. Then its corresponding Bayes

44

decision rule is

δL(f (x′)) = arg min

a∈A

E

(cid:98)y∼f (x′)[L(a, (cid:98)y)].

(B.2)

For some subset L ⊂ Lall, we denote the corresponding Bayes decision rules as ∆L := {δL | L ∈ L}.

Decision calibration. Zhao et al. (2021) defines decision calibration to formalize the following
intuition: ideally, a decision-maker with loss function L should be able to consider an arbitrary deci-
sion rule δ ∈ ∆all and compute the expected loss of using δ in decision-making, given a probabilistic
forecaster f .

Definition B.2 (Decision Calibration, Definition 2 in Zhao et al. (2021)). For any set of loss functions
L ⊂ Lall, we say that a forecaster f is L-decision calibrated (with respect to the ground-truth conditional
distribution p(y | ·)) if for each L ∈ L and δ ∈ ∆L,
L(cid:0)δ(f (x′)), (cid:98)y(cid:1)(cid:105)

L(cid:0)δ(f (x′)), y(cid:1)(cid:105)

= Ex′∼p(x′)Ey∼p(y|x′)

Ex′∼p(x′)E

(cid:98)y∼f (x′)

(B.3)

(cid:104)

(cid:104)

.

Following the analysis of Zhao et al. (2021) (§3.1), the left-hand side of Eq. B.3 simulates the loss of
taking decisions according to the loss L and rule δ using data drawn from the forecast (cid:98)y ∼ f (x′).
This simulated loss can be computed by a user without observing any ground-truth outcome
y ∼ p(y | x′). The right-hand side is the true loss for using decision rule δ with loss L. Therefore,
decision calibration means that a forecaster f can be used to accurately estimate the expected loss of
a decision rule under the true data distribution.

Decision calibration with LM assistance. With a few particular choices, we can apply this general
definition to the setting of LM-assisted decision-making. In our context, the relevant variables are
the open-ended query to the LM q, the LM response z, and the related question-answer pair (x, y).
These variables are distributed by the joint p(q, x, y, z) := p(q, x, y)πRL(z | q). Write the marginal over
questions and related LM responses as p(x, z) = (cid:80)
q′,y′ p(q′, x, y′, z). In the notation of Definition B.2,
we define the input x′ := (x, z).
To be concrete, we reiterate decision calibration in our context, where users also condition their
forecasts on LM outputs.

Definition B.3 (Decision Calibration with LM Assistance). For any set of loss functions L ⊂ Lall, we
say that a reader function f : X × Z → ∆|Y| is L-decision calibrated (with respect to the ground-truth
conditional distribution p(y | ·)) if for each L ∈ L and δ ∈ ∆L,

E(x,z)∼p(x,z)E

(cid:98)y∼f (x,z)

(cid:104)

L(cid:0)δ(f (x, z)), (cid:98)y(cid:1)(cid:105)

= E(x,z)∼p(x,z)Ey∼p(y|x)

(cid:104)

L(cid:0)δ(f (x, z)), y(cid:1)(cid:105)

.

Zhao et al. (2021) proves that if a forecaster f is decision calibrated, it provides the downstream
decision-maker with two key guarantees. First, a decision-maker with loss L is assured that the
Bayes decision rule δL outperforms alternative decision rules. Second, the decision-maker is able
to accurately estimate the loss that they will incur by using the Bayes decision rule δL, despite not
having access to ground-truth outcomes y ∼ p(y | x′). In particular, we will show that decision
calibration provides the following guarantees in our setting.
Lemma 1 (instantiation of Proposition 1 in Zhao et al. (2021)). If a reader f : X × Z → ∆|Y| is
L-decision calibrated, then it satisfies:

1. No regret: for each δ′ ∈ ∆L,

E(x,z)∼p(x,z)Ey∼p(y|x)

(cid:104)

L(cid:0)δL(f (x, z)), y(cid:1)(cid:105)

≤ E(x,z)∼p(x,z)Ey∼p(y|x)

(cid:104)

L(cid:0)δ′(f (x, z)), y(cid:1)(cid:105)

.

45

2. Accurate loss estimation:

E(x,z)∼p(x,z)E

(cid:98)y∼f (x,z)

(cid:104)

L(cid:0)δL(f (x, z)), (cid:98)y(cid:1)(cid:105)

= E(x,z)∼p(x,z)Ey∼p(y|x)

(cid:104)

L(cid:0)δL(f (x, z)), y(cid:1)(cid:105)

.

To summarize, no regret means that the Bayes decision rule δL performs no worse than any other
decision rule δ′ ∈ ∆L. Accurate loss estimation means that the decision-maker can determine before
observing any ground-truth outcomes what their expected loss will be, allowing them to prepare for
the future.

B.3 Linguistic Calibration and Optimal Decision-making

Now we can show that linguistic calibration implies the decision-making guarantees of Lemma 1.
We do so using a key result from Zhao et al. (2021) (Theorem 1): standard notions of classifier
calibration are equivalent to L-decision calibration for particular choices of the loss family L. We
highlight two examples of this equivalence here and direct the interested reader to Zhao et al. (2021)
for others.

Example 1: linguistic distribution calibration implies Lall-decision calibration. Recall distri-
bution calibration (cf. §2.1 for definition), the strongest standard notion of classifier calibration
(Bröcker, 2009). Zhao et al. (2021) (Theorem 1) proves that distribution calibration is equivalent
to Lall-decision calibration. Suppose that we optimize the linguistic calibration objective RLC in
Eq. B.1 and obtain linguistic distribution calibration (for example, by maximizing RLC). By definition,
linguistic distribution calibration means that the reader f is distribution calibrated over p(x, y, z).
Then, applying Zhao et al. (2021) (Theorem 1), the reader is Lall-decision calibrated, and therefore a
Bayes-optimal decision-maker is guaranteed to perform no worse than any decision rule δ′ ∈ ∆all.
Unsurprisingly, this is very difficult to achieve in practice. However, even if linguistic calibration
only manages to calibrate the reader in a weaker sense, we still have L-decision calibration with
respect to a smaller family of losses L.

Example 2: linguistic confidence calibration implies Lr-decision calibration. To illustrate this
decision-making guarantee for weaker notions of calibration, recall confidence calibration (cf. §2.1).
Confidence calibration is a relaxation of the distribution calibration condition, only requiring it to
hold for the most likely label. Zhao et al. (2021) (Theorem 1) demonstrates that confidence calibration
is equivalent to Lr-decision calibration, where

Lr :=

(cid:110)
L(a, y) = 1[y ̸= a ∩ a ̸=⊥] + β · 1[a =⊥] (cid:12)

(cid:12) a ∈ Y ∪ {⊥}, ∀β ∈ [0, 1]

(B.4)

(cid:111)
.

Lr is the class of loss functions for the refrained prediction task, where the decision-maker can either
take an action corresponding to an answer y ∈ Y or report “I don’t know” with the action ⊥. Then,
the decision-maker incurs a loss of 0 for correctly predicting the ground-truth answer y, a loss of 1
for predicting an incorrect answer, or a loss of β < 1 for reporting ⊥. Similarly, Zhao et al. (2021)
(Theorem 1) proves that classwise calibration (cf. §2.1) corresponds to another notion of L-decision
calibration.

Linguistic calibration implies no regret and accurate loss estimation guarantees. We conclude this
section by applying the correspondence between classifier calibration and L-decision calibration to
prove decision-making guarantees for linguistic calibration. For each notion of classifier calibration
ϕ ∈ {distribution, classwise, confidence}, identify the corresponding class of decision calibration
loss functions as Lϕ (e.g., distribution induces Lall, and confidence induces Lr). Then we have the
following result:

46

Theorem B.4 (Linguistic ϕ-calibration implies no regret and accurate loss estimation guarantees).
Suppose that

• An LM πRL is linguistically ϕ-calibrated with respect to a user with reader function f : X × Z → ∆|Y|,

where ϕ ∈ {distribution, classwise, confidence} (cf. Definition 2.1).

• The user’s loss function L : A × Y → R is in the set of losses Lϕ induced by ϕ.

• The conditional independence assumption B.1 holds.

Then f is Lϕ-decision calibrated, and moreover, Lemma 1 (no regret and accurate loss estimation guarantees)
holds for the user’s decision-making.

Proof. Linguistic ϕ-calibration is equivalent to ϕ-calibration of the reader f (cf. Definition 2.1). Recall
that p(x, z) = (cid:80)
q′,y′ p(q′, x, y′, z) is the marginal over questions and related LM responses, and
let x′ := (x, z) and p(x′) := p(x, z). Applying Theorem 1 from Zhao et al. (2021) (equivalence of
ϕ-calibration and Lϕ-decision calibration), we observe that the reader f is Lϕ-decision calibrated
according to Definition B.2, Eq. B.3, and therefore the assumption of Lemma 1 holds.
We conclude by proving Lemma 1. Because the reader f is Lϕ-decision calibrated, we may apply
Proposition 1 from Zhao et al. (2021), which provides the no regret and accurate loss estimation
guarantees of decision calibration. Lastly, we invoke the conditional independence statement
p(Y | X, Z) = p(Y | X) (Assumption B.1) to simplify these guarantees, obtaining the guarantees in
Lemma 1.

As an example of this result, consider ϕ = confidence calibration. Suppose that our optimization of
RLC leads to linguistic confidence calibration, i.e., a confidence calibrated reader f . Because of the
equivalence of confidence calibration and Lr-decision calibration, f is Lr-decision calibrated and
we obtain the corresponding guarantees for decision-making. For example, the no regret guarantee
states that the Bayes decision rule δL outperforms decision rules δ′ ∈ ∆Lr, where ∆Lr is a set of Bayes
decision rules induced by Lr. Intuitively, because confidence calibration is weaker than distribution
calibration, ∆Lr is a more restricted set than ∆all.

Summary: linguistic calibration provides optimal decision-making guarantees.
If we maximize
the linguistic calibration objective, our forecasts are distribution calibrated and therefore we obtain
Lall-decision calibration, i.e., the Bayes optimal decision rule outperforms all other decision rules.
Even if our optimization only achieves a weaker notion of calibration such as confidence calibration,
we still obtain decision calibration over a more restricted class of loss functions. This fact also
motivates our use of reader ECE as an evaluation metric, because it estimates confidence calibration
and can therefore be thought of as a proxy for the quality of downstream user decision-making. For
further discussion, we direct the reader to Zhao et al. (2021) and Zhao and Ermon (2021).

47

C Training Framework

Linguistic calibration is a general-purpose definition (Definition 2.1) that is agnostic to the particular
design decisions we made in our proof-of-concept training framework. Nevertheless, this section
details the design choices we made, including the regularized objective function we used to train
with our neural net–based surrogate reader, and our implementation of PPO (Schulman et al., 2017).

C.1 Regularized Linguistic Calibration Objective

Objective regularized to encourage normalized surrogate forecasts.
In our instantiation of
linguistic calibration, we use a surrogate forecast (cid:101)f (x, z) which is not guaranteed to be normalized:
we may have (cid:101)f (x, z) ∈ [0, 1]|Y| \ ∆|Y|. We remedy this by adding a regularization term in the reward
function which encourages (cid:101)f (x, z) to be normalized, restoring strict propriety (cf. §2.3) even with
forecasts in [0, 1]|Y|. We indeed find in practice that training with this objective results in normalized
forecasts.
Our regularized reward function RLC : [0, 1]|Y| × Y → R is written as:

RLC( (cid:101)f (x, z), y) = log (cid:101)f (x, z)y − λ

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

1 −

(cid:88)

y′∈Y

(cid:101)f (x, z)y′

(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)
(cid:12)

+ C,

(C.5)

where λ > 0, C > 0 are hyperparameters. C is a small positive constant which we find to reduce
reward hacking during PPO.
Below, we prove that with λ > 1, we indeed have a strictly proper objective.

C.2 Proof: Regularized Objective is Strictly Proper

Setup.
In decision-based RL, we wish to optimize an objective that encourages the surrogate
forecast (cid:101)f (x, z) to exactly match the ground-truth distribution p(y | x), which is supported over all
possible answers Y. (cid:101)f (x, z) is not necessarily in ∆|Y| because it is obtained by several independent
neural network predictions falling in the range [0, 1], i.e., (cid:101)f (x, z) ∈ [0, 1]|Y|. Therefore, in addition
to a log loss term, we also add a regularizer which encourages (cid:101)f (x, z) to fall in the simplex, i.e., be
normalized.
We would like our reward function to be strictly proper—its unique maximizer should be the
ground-truth answer distribution. We will prove that our reward function is indeed strictly proper
below. First, we define (strictly) proper scoring rules in our setting, which is slightly more general
than the standard definition (e.g., in Gneiting and Raftery (2007)) in that the first argument need not
be a probability distribution: it may fall in [0, 1]|Y| \ ∆|Y|. As shorthand throughout this proof, we
will refer to the ground-truth answer distribution as p∗ := p(y | x) and the forecast as (cid:101)p := (cid:101)f (x, z),
because this proof is done pointwise for a given (question, LM generation) tuple (x, z).
Definition C.1 (based on Gneiting and Raftery (2007)). Let ¯R be the extended real line [−∞, ∞]. A
scoring rule R : [0, 1]|Y| × Y → ¯R is proper if

EY ∼p∗[R(p∗, Y )] ≥ EY ∼p∗[R((cid:101)p, Y )],

∀p∗ ∈ ∆|Y|, (cid:101)p ∈ [0, 1]|Y|.

It is strictly proper if Eq. C.6 holds with equality if and only if p∗ = (cid:101)p.
Equivalently, we may express the strictly proper condition as

EY ∼p∗[R(p∗, Y )] > EY ∼p∗[R((cid:101)p, Y )],

∀p∗ ∈ ∆|Y|, (cid:101)p ∈ [0, 1]|Y|, (cid:101)p ̸= p∗.

48

(C.6)

(C.7)

In proving that RLC is strictly proper, we will also make use of the following standard result:
Lemma 2 (e.g., Gneiting and Raftery (2007)). The logarithmic scoring rule Rlog : ∆|Y| × Y → {−∞} ∪ R
is strictly proper.

Lastly, we will also use the log-sum inequality:
Lemma 3 (e.g., p. 29, Cover and Thomas (1991)). Let a1, . . . , an and b1, . . . , bn be nonnegative numbers.
Denote a := (cid:80)n

i=1 ai and b := (cid:80)n

i=1 bi. Then

n
(cid:88)

i=1

ai log

ai
bi

≥ a log

a
b

.

We will now prove that our regularized training framework objective RLC is a strictly proper scoring
rule using the condition for strict propriety in Eq. C.7.
Theorem C.2. For any C ∈ R and with λ > 1, the reward function RLC((cid:101)p, y) from Eq. C.5 is strictly proper.
Proof. Take an arbitrary p∗ ∈ ∆|Y| and (cid:101)p ∈ [0, 1]|Y| such that (cid:101)p ̸= p∗. We need to show that

EY ∼p∗[RLC(p∗, Y )] − EY ∼p∗[RLC((cid:101)p, Y )] > 0.
(C.8)
All expectations for the rest of the proof are taken with respect to p∗, so we omit the subscript Y ∼ p∗.
As in the main text, we identify the space of answers Y with the index set [|Y|], and therefore use
notation py to refer to the y-th index of the vector p. Lastly, we write (cid:80)
Plugging in with our reward function, Inequality C.8 is equivalent to
log (cid:101)pY − λ|1 − 1⊤
Y − λ|1 − 1⊤p∗| + C

y′∈Y (cid:101)py′ as 1⊤

(cid:101)p for brevity.

(cid:101)p| + C

log p∗

> 0.

− E

E

(cid:104)

(cid:105)

(cid:104)

(cid:105)

We can simplify the LHS by applying linearity of expectation and simplifying expectations of
constants:

Y ] − λ|1 − 1⊤p∗| − E [log (cid:101)pY ] + λ|1 − 1⊤
Note that we assume p∗ ∈ ∆|Y| so 1⊤p∗ = 1, and we can simplify further to

LHS = E [log p∗

(cid:101)p|.

LHS = E [log p∗

(C.9)
Now we proceed by cases on whether or not (cid:101)p ∈ ∆|Y|. Suppose that indeed (cid:101)p ∈ ∆|Y|. Then its
normalization term λ|1 − 1⊤
(cid:101)p| is also 0, and therefore this inequality condition is precisely the strict
propriety condition for the logarithmic scoring rule (including the assumption that both p∗ and (cid:101)p are
in the simplex):

Y ] − E [log (cid:101)pY ] + λ|1 − 1⊤

(cid:101)p|.

E [log p∗

Y ] − E [log (cid:101)pY ] > 0,

which is true according to Lemma 2, and therefore this case is done.
So now assume that (cid:101)p ∈ [0, 1]|Y| \ ∆|Y|. This implies that its normalization term is greater than 0, i.e.,
|1 − 1⊤
(cid:101)p| > 0. In this case, we can prove using the properties of log and the log-sum inequality that
the LHS expression in Eq. C.9 is greater than 0, in order to complete the proof.
First, we use the properties of log to rewrite the expectations in Eq. C.9 into the entropy of p∗ plus a
relative entropy between p∗ and (cid:101)p:

(definition of entropy functional H[·])

(cid:101)p|

(decomposition of cross-entropy)

LHS = E [log p∗

Y ] − E [log (cid:101)pY ] + λ|1 − 1⊤
(cid:101)p|
+ λ|1 − 1⊤

= −H[p∗] − E [log (cid:101)pY ] + λ|1 − 1⊤
p∗
y
= −H[p∗] + H[p∗] +
(cid:101)py

p∗
y log

(cid:88)

(cid:101)p|

y∈Y

=

(cid:88)

y∈Y

p∗
y log

p∗
y
(cid:101)py

+ λ|1 − 1⊤

(cid:101)p|,

49

where the penultimate step used the decomposition of cross-entropy = entropy + relative entropy.
We cannot immediately apply the information inequality to show that the relative entropy is
nonnegative, because (cid:101)p is not a probability distribution (and therefore the KL Divergence and
relative entropy are not the same; see, e.g., p. 90 in Boyd and Vandenberghe (2004)). However,
we may use the log-sum inequality to finish the proof. Applying Lemma 3 using the fact that all
elements of p∗ and (cid:101)p are nonnegative, we can lower bound the relative entropy term as

p∗
y log

(cid:88)

y∈Y

p∗
y
(cid:101)py

≥ 1⊤p∗ log

1⊤p∗
1⊤
(cid:101)p

= log

1
1⊤

.
(cid:101)p

Altogether, substituting normalizer Z := 1⊤
Eq. C.9 of

(cid:101)p, we currently have a lower bound of the LHS in

+ λ|1 − Z|.

LHS ≥ log

1
Z
Inspecting this function for λ > 1, we can see that it is positive for all values of Z other than Z = 1.
To be rigorous, we complete the proof with cases on the values of Z. Because (cid:101)p ∈ [0, 1]|Y|, Z is
nonnegative, and recall that we already covered the case when Z = 1⊤
(cid:101)p = 1 by reduction to strict
propriety of the logarithmic scoring rule (Lemma 2).
When Z ∈ [0, 1), we immediately see that log 1
implying that the lower bound in Eq. C.10 is positive as we needed to show.
Lastly, suppose that Z ∈ [1, ∞). In that case, λ|1 − Z| = λ(Z − 1). Now, note that the derivative of
the lower bound in Eq. C.10 is positive for all Z ≥ 1 (recalling that λ > 1):

Z > 0 and λ|1 − Z| > 0 (since λ > 1 by assumption),

(C.10)

(cid:20)

(cid:21)

log

1
Z

+ λ(Z − 1)

d
dZ
e.g., at Z = 1, we have that the derivative is greater than − 1
1 + 1 = 0. And further, at precisely Z = 1,
the lower bound equals log 1
1 + λ|1 − 1| = 0. This proves that the lower bound is positive for all
Z ∈ (1, ∞), which implies that the LHS is also positive for Z ∈ (1, ∞). We have covered all values
of Z, concluding that when λ > 1, the reward function in Eq. C.5 is strictly proper.

+ λ > −

= −

+ 1,

1
Z

1
Z

C.3 Additional Details on Training Framework

We provide additional details on the training framework below. For other details, we direct the
reader to the codebase.3

Training and Validation Splits. Our training framework and all baselines use examples from the
TriviaQA (Joshi et al., 2017) unfiltered.nocontext subset on HuggingFace Datasets (Lhoest
et al., 2021). These examples are randomly assigned to the following splits:

• SFT (10000 examples): used for summary distillation and the other SFT baselines (Factuality

SFT, Claude Distill).

• Prompt Validation (1000 examples): used for all ICL-based baselines and to construct ICL

examples for the simulated reader, which uses an API-based LLM.

• Reward Model (20000 examples): used to train surrogate reader for LC, and binary reward

model for Factuality RL baseline.

• PPO (40000 examples): used for PPO with LC RL and Factuality RL methods.

3https://github.com/tatsu-lab/linguistic_calibration

50

• PPO Validation (1000 examples): during PPO, we evaluate reward model rewards on this split

and store checkpoints every 20 steps.

• Validation (1000 examples): used for tuning evaluation temperature and model selection for

RL methods (described below).

Summary distillation details. For each SFT dataset query q, we use the (non-confidence) ICL
baseline to sample eight long-form generations {zi}8
i=1 ∼ πICL(z | q) (temperature 0.7) that are then
summarized using a Claude few-shot prompt s ∼ Summarize(s | z1, . . . , z8) (temperature 0.3). We
use the dataset of (q, s) pairs to finetune the base Llama 2 7B model, obtaining πSFT. The same
temperature settings are used for obtaining Summary ICL context examples, and for the Direct
Summary Eval oracle baseline.

Surrogate and reward model details. Following our evaluation protocol for user forecasts (§4.1),
we specify prompt instructions so that LLM simulated forecasts fLLM(x, z) used in Algorithm 1
avoid the use of background knowledge. Dataset construction for the surrogate reader models
ForecastProbs and ExtractAnswers is described in Algorithm 1.
For LC RL, we use the following training details:

• ForecastProbs: a Llama 2 7B–based model initialized with the LC SFT parameters and trained

using a binary cross-entropy loss on prompts from the Reward Model split.

• ExtractAnswers: a RedPajama 3B–based model initialized from the pretrained model weights
and trained with causal language modeling on targets formatted as a delimited list of possible
answers, e.g., “Sushi; Sashimi; Nigiri”. We used a 3B model instead of Llama 2 7B due
to computational constraints, i.e., to fit all standard PPO models such as the policy, reference
policy, value model, and reward models into GPU memory on an 8x80GB A100 node.

For Factuality RL, we use the following training details:

• Reward Model: a Llama 2 7B–based model initialized with the Factuality SFT parameters and
trained using binary cross-entropy loss on prompts from the Reward Model split, with binary
correctness labels from Claude 2.0. In producing binary correctness labels for a given (question,
ground-truth answer, generation) tuple (x, y, z), Claude is given access to the ground-truth
answer. Therefore, determining binary correctness is a simple extractive task that we find
Claude performs with very high accuracy.

We tune all reward models over a wide range of learning rates; see codebase for details on hyperpa-
rameters.

PPO implementation and training. We use a standard implementation of PPO from Dubois et al.
(2023) and train with 8 80GB A100 GPUs using Flash Attention 2 (Dao, 2023, Dao et al., 2022) and
PyTorch FSDP (Zhao et al., 2023). For both PPO-based methods (LC RL and Factuality RL), we use
default hyperparameters other than the following:

• We use a larger step batch size (512) with one optimization epoch per step for better training

stability.

• We shorten query_len to 128 tokens, because our PPO inputs were essentially all under this

length.

• We use a slightly lower temperature during the rollout phase (0.7 instead of 1.0).

51

• We use the paged_adamw_8bit optimizer (Dettmers et al., 2022) due to computational con-
straints. We also use this optimizer for SFT and reward model training for consistency, and find
it had no negative impact on validation performance compared to AdamW (Loshchilov and
Hutter, 2019).

• We train for 1500 PPO steps.

We tune the KL penalty and learning rate of both PPO methods across a wide range, ultimately
finding that kl_coef of 0.1 and learning rate of 1e-5 works best for both methods.
For the LC RL objective, we find that λ = 5, C = 5 works well to enforce normalization of down-
stream forecasts and prevent reward hacking. In the log-loss term, we clip the probability of the
ground-truth answer at ϵ =1e-4 for numerical stability.

Model selection for RL methods. We perform model selection for PPO-based methods (LC RL,
Factuality RL) by taking the 5 PPO checkpoints with the highest reward model rewards on the PPO
Validation split, evaluating them on the Validation split with the API-based LLM simulated reader,
and selecting the checkpoint with the best ECE.

52

D Evaluation Framework

Test Dataset Splits. We use the following test dataset splits.

• TriviaQA (Joshi et al., 2017) (11313 examples):

take the validation split from the
unfiltered.nocontext subset from HuggingFace Datasets (Lhoest et al., 2021, Wolf et al.,
2020).

• Jeopardy (Kaggle, 2020) (10638 examples): take a random sample of the train subset from
HuggingFace Datasets. These questions are initially written in the indirect Jeopardy-style
format (“For the last 8 years of his life, Galileo was under house arrest for espousing this man’s
theory”), so we use Claude 2.0 to convert them into direct questions x.

• SciQ (Welbl et al., 2017) (13679 QA examples): combine train, validation, and test splits from
allenai/sciq on HuggingFace Datasets. SciQ is a dataset of crowdsourced science exam
questions in biology, chemistry, earth science, and physics, spanning elementary level to college
introductory material.

• BioASQ (Krithara et al., 2023) (1515 QA examples): we use the 2024 BioASQ12 Task B dataset,
which is a biomedical QA dataset annotated by a team of experts. Specifically, we use their 1515
“factoid” examples, which have a short-answer gold label (e.g., “proprotein convertase subtilis-
in/kex in type 9”). The dataset is available at http://participants-area.bioasq.org/datasets.
• Person Biography Generation (Min et al., 2023) (500 entities): use the unlabeled split of

Wikipedia entities from the FactScore dataset.

TriviaQA, Jeopardy, SciQ, and BioASQ are question-answering datasets which we repurpose for
long-form evaluation. Specifically, for these question-answering datasets, we use an API-based LLM
(Claude 2.0) to convert questions x into open-ended queries q that allow us to sample long-form
generations z ∼ π(z | q) on the topic of the question x (as we do at training time, cf. §3.1). For
Person Biography Generation, we construct open-ended queries using the fixed template “Write a
paragraph bio about {entity}.”.

Confidence intervals. We report bootstrap 0.95 confidence intervals in all frontier plots and tabular
results, except for the BioASQ dataset. BioASQ is a relatively small dataset (1515 examples; it is
expert-written) and therefore we report Student’s t distribution 0.95 confidence intervals over 10
evaluation seeds. In reliability diagrams for BioASQ, we report the reliability diagram computed
over samples pooled across all seeds in blue, and per-seed reliability diagrams in gray.

D.1 Simulated Evaluation

Simulated reader. Simulated forecasts f (x, z) are a distribution over possible answers to the
question x, based on long-form LM generations z. All simulated forecasts in this paper are formed
using an API-based LLM simulated reader fLLM. We use Claude due to compute constraints, but
other LLMs such as GPT-4 are well-capable of producing high-quality forecasts. These simulated
forecasts are used in two parts of our training and evaluation frameworks:

• Training framework: simulated forecasts are used as ground-truth targets for training the
surrogate reader (cid:101)f (cf. Algorithm 1). Then the surrogate reader is used during PPO, enabling
efficient training by avoiding calls to API-based LLMs.

• Evaluation framework: in our question-answering evaluations, we sample a held-out question-
answer pair (x, y) ∼ p(x, y), convert x into open-ended query q, and evaluate the accuracy and

53

ECE of the simulated forecast fLLM(x, z) for z ∼ π(z | q) sampled from a variety of LMs such as
LC RL and Factuality RL.

In both cases, we obtain simulated forecasts by decomposing reading into two prompts, one for
ExtractAnswers and one for ForecastProbs, following the description in §3. This is just one way of
implementing a reader, and there are likely other reasonable ways to obtain high-quality probabilistic
predictions conditioned on long-form text.
In order to compute accuracy and ECE for the evaluation framework, we need to obtain the
confidence and correctness of each forecast. We direct the reader to our codebase4 for all prompts.

Accuracy and ECE for confidence-based methods. The confidence of a forecast is obtained by
simply taking its argmax probability maxy∈Y f (x, z)y. Recall that for question-answering evaluation
datasets, we prompt an API-based LLM to perform the ForecastProbs operation. We find a zero-
shot prompt is effective because confidence statements in z associated with the question x are
usually numerical (and even when they are occasionally linguistic, Claude 2.0 produces reasonable
confidence estimations). The correctness is obtained by determining whether the top answer in the
forecast is correct, i.e., 1[y = arg maxy′∈Y f (x, z)y′]. In practice, we perform this check by prompting
Claude with the question x, ground-truth answer choice y, and the argmax answer choice, and asking
the LLM to decide whether the two answers are semantically equivalent, using a few-shot prompt.
For TriviaQA and Jeopardy, which are general-domain knowledge tasks, claude-2.0 can perform
this semantic equivalence check reliably. For the domain-specific tasks SciQ and BioASQ, we use the
more powerful claude-3-opus-20240229 to perform this semantic equivalence check, because
the ground-truth and argmax answers are more complicated; e.g., y = “hormone receptor-positive,
human epidermal growth factor receptor 2-negative”, argmax = “hormone-receptor-positive, HER2-
negative breast cancer patients”. This semantic equivalence check avoids the false negative issues
that string match–based metrics encounter due to paraphrases, which are very common in our
open-ended, long-form generation setting.

Accuracy and ECE for non-confidence methods. Our evaluation for non-confidence methods
closely follows the pipeline for confidence-based methods. We extract answers using the Extrac-
tAnswers prompt, which accounts for the possibility that models may provide multiple mutually
exclusive answers to the question x. We select the “argmax” answer by taking the answer which
appears first in the paragraph z. Because non-confidence methods are not prompted or trained
to produce confidence statements, we fix their confidence values to 1. Then, we compute ECE
for non-confidence methods in closed form as (1 − Accuracy). This approach is validated by
our human evaluations, in which the correlation between a fixed confidence value of 1 and the
human-interpreted argmax probability of Factuality RL is 0.993, over 1000+ samples (cf. Table 7).

Temperature tuning. We tuned the decoding temperature for generating paragraphs across the
range {0.1, 0.3, 0.5, 0.7} for the core methods (LC RL, LC SFT, Factuality RL, Factuality SFT) on
the TriviaQA Validation split with simulated reader evaluation. We found that temperature 0.3
uniformly performed the best across methods. Therefore we defaulted to paragraph decoding
temperature 0.3 for evaluation throughout. We found that temperature 0.2 consistently worked
well for simulated reader operations ExtractAnswers, ForecastProbs, and for checking semantic
equivalence.

4https://github.com/tatsu-lab/linguistic_calibration

54

Other evaluation details. Because we prompt our LMs with open-ended queries (q := “Write a
paragraph about {x}”) it is rarely possible that a paragraph z will not provide information that
answers question x. We account for this during evaluation by allowing the ExtractAnswers prompt
to return an empty list if z does not provide any plausible answers to the question. For any example
with an empty extracted list, we consider the accuracy to be 0 and the confidence to be 1, which is the
most conservative possible penalty for both accuracy and ECE. Because we significantly outperform
baselines finetuned with RL for factuality, we account for the possibility of better responsiveness to
the prompt (and therefore fewer empty ExtractAnswers lists) explaining our improvements.

D.2 FactScore-Based Evaluation Metric

We use an automated FactScore-based (Min et al., 2023) metric using an API-based LLM (Claude 2.0
in our case) to test the accuracy and calibration of generated biographies z at the per-claim level,
probing for the quality of confidences throughout long-form generations. §4.3 describes this metric
in detail. Prompts and code for this metric are available in our codebase.

Forecasting confidences. As part of this metric, for confidence-based methods, we need to forecast
probabilities which are used to compute ECE. As compared to the question-answering evaluations,
this evaluation considers all claims in the generated paragraph and tends to have a higher proportion
of claims with linguistic confidence statements. Therefore, we use the following procedure to assign
probabilities to each claim. When a claim has a clear numerical uncertainty (e.g., a percentage) this is
a simple extractive task which API-based LLMs perform well by default. For interpreting linguistic
uncertainties, we provide the LLM with a mapping between linguistic phrases and probabilities
collected in a linguistics human study (Wallsten, 1990), and allow it to generalize from this mapping
to phrases that are not present in it (Table 5). We believe that developing better strategies to interpret

Linguistic Confidence Phrase Probability

Almost Impossible
Doubtful
Improbable
Unlikely
Possible
Tossup
Good Chance
Likely
Probable
Almost Certain

0.05
0.1
0.1
0.15
0.3
0.5
0.65
0.75
0.75
0.95

Table 5: Ground-truth linguistic confidence mapping used in our FactScore probability interpretation
prompt, from Wallsten (1990).

linguistic confidence statements in a manner tailored to a given user population is exciting future
work (discussed in §6).

55

D.3 Human Evaluation

Task.
In order to compute our accuracy and ECE metrics using forecasts from human crowdwork-
ers, we asked crowdworkers to provide the following for a given question, ground-truth answer,
and long-form generation tuple (x, y, z):

• Confidence: the max probability of their forecast distribution maxy∈Y f (x, z)y.
• Correctness: a binary indicator of whether or not their top answer was correct, 1[y =

arg maxy′∈Y f (x, z)y′].

We asked human crowdworkers to interpret the generated paragraphs literally, and provided a
couple examples. These examples were intentionally selected to be unambiguous, i.e., numerical
confidences, to ensure we were not biasing towards any particular manner of interpreting linguistic
confidences.

Qualification. We conducted two qualification rounds with our human annotators for the question-
answering TriviaQA experiment. In the first round, we sent a single question-answer pair (x, y)
sampled from TriviaQA, along with 3 model outputs (from LC RL, LC SFT, and Factuality RL)
to 117 workers, out of which 32 matched the expected output. This example was selected to be
unambiguous. In the second round, we sent the 32 workers who passed the previous test a roughly
hour-long test (examples also selected to be unambiguous; the student authors of this paper agreed
on all confidence and correctness labels of the hour-long test). Among these annotators, we took the
top 15 performers for the main annotation process.

Main annotation round. During the main annotation process, we simply selected examples uni-
formly at random from the TriviaQA test set, including examples which were potentially ambiguous
due to linguistic confidence statements. Overall, we collected annotations for 1008 examples (each
with 3 model outputs) from human crowdworkers.

Quality assurance. We compared crowdworker agreement with Claude 2.0 as a quality assurance
measure. Five of the annotators had agreement rates which dropped significantly after the qual-
ification round, from perfect performance on the hour-long test to less than 0.40 Cohen’s kappa
agreement with Claude. Therefore, we discontinued working with these annotators and removed
their labels.

Payment. To determine payment, the authors of this paper timed our own completion time of the
task, which was an average of approximately 1.3 minutes. Following best practices for crowdworker
payment (Liang et al., 2023, Whiting et al., 2019), we used a conservative estimate of annotation time
of 3 minutes per task and set the per-task payment to $0.85 amounting to a compensation of $17 per
hour, greater than the California minimum wage of $16 per hour.

Agreement statistics.
crowdworkers) computed over 5% of the collected samples, for each of the tested models.
In Table 7, we report human agreement with Claude 2.0 over all examples.

In Table 6 we report interannotator agreement (i.e., agreement between

56

Model

Correlation on Confidence Cohen’s Kappa on Correctness

LC RL
LC SFT
Factuality RL

0.886
0.719
1.000

0.850
0.842
0.758

Table 6: Interannotator Agreement.

Model

Correlation on Confidence Cohen’s Kappa on Correctness

LC RL
LC SFT
Factuality RL

0.626
0.618
0.993

0.739
0.748
0.741

Table 7: Human-Claude 2.0 Agreement.

57

