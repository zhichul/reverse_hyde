SFT Memorizes, RL Generalizes:
A Comparative Study of Foundation Model Post-training

Tianzhe Chu ♠ * Yuexiang Zhai ♥ ♣ * Jihan Yang ♦ Shengbang Tong ♦

Saining Xie ♣ ♦ Dale Schuurmans ♣

Quoc V. Le ♣ Sergey Levine ♥ Yi Ma ♠ ♥

5
2
0
2

y
a
M
6
2

]
I

A
.
s
c
[

2
v
1
6
1
7
1
.
1
0
5
2
:
v
i
X
r
a

Abstract

1. Introduction

Supervised fine-tuning (SFT) and reinforcement
learning (RL) are widely used post-training tech-
niques for foundation models. However, their
respective role in enhancing model generaliza-
tion in rule-based reasoning tasks remains un-
clear. This paper studies the comparative effect
of SFT and RL on generalization and memoriza-
tion, focusing on text-based and visual reason-
ing tasks. We introduce GeneralPoints, an
arithmetic reasoning card game, and also con-
sider V-IRL, a real-world navigation environ-
ment, to assess how models trained with SFT and
RL generalize to unseen variants in both novel
textual rules and visual domains. We show that
RL, especially when trained with an outcome-
based reward, generalizes in both the rule-based
textual and visual environments. SFT, in con-
trast, tends to memorize the training data and
struggles to generalize out-of-distribution in ei-
ther scenario. Further analysis reveals that RL
improves the model’s underlying visual recog-
nition capabilities, contributing to its enhanced
generalization in visual domains. Despite RL’s
superior generalization, we show that SFT is still
helpful for effective RL training: SFT stabilizes
the model’s output format, enabling subsequent
RL to achieve its performance gains. These find-
ings demonstrate the advantage of RL for acquir-
ing generalizable knowledge in complex, multi-
modal tasks.

♦NYU,

*Equal contribution .

♠HKU, ♥UC Berkeley, ♣Google
All experi-
DeepMind,
Project page:
ments are conducted outside of Google.
https://tianzhechu.com/SFTvsRL.
Correspondence
to: Tianzhe Chu <tianzhechu@gmail.com>, Yuexiang Zhai <si-
monzhai@berkeley.edu>.

University of Alberta.

Proceedings of the 42 nd International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).

1

Although SFT and RL are both widely used for foundation
model training (OpenAI, 2023b; Google, 2023; Jaech et al.,
2024; DeepSeekAI et al., 2025), their distinct effects on
generalization (Bousquet & Elisseeff, 2000; Zhang et al.,
2021) remain unclear, making it challenging to build reli-
able and robust AI systems. A key challenge in analyz-
ing the generalizability of foundation models (Bommasani
et al., 2021; Brown et al., 2020) is to separate data mem-
orization1 from the acquisition of transferable principles.
Thus, we investigate the key question whether SFT or RL
primarily memorize training data (Allen-Zhu & Li, 2023a;
Ye et al., 2024; Kang et al., 2024), or whether they learn
generalizable rules that can adapt to novel task variants.

To address this question, we focus on two aspects of gener-
alization: textual rule-based generalization and visual gen-
eralization. For textual rules, we study the ability of a
model to apply learned rules (given text instructions) to
variants of these rules (Zhu et al., 2023; Yao et al., 2024;
Ye et al., 2024). For vision-language models (VLMs),
visual generalization measures the consistency of perfor-
mance with variations in visual input, such as color and
spatial layout, within a given task. For studying text-based
and visual generalization, we investigate two different tasks
that embody rule-based and visual variants. Our first task
is GeneralPoints, an original card game task similar
to Points24 of RL4VLM (Zhai et al., 2024a), which is
designed to evaluate a model’s arithmetic reasoning ca-
pabilities. The model receives four cards (presented as a
text description or an image), and is required to compute
a target number (24 by default) using each card’s numer-
ical value exactly once. Second, we adopt V-IRL (Yang
et al., 2024a), a real-world navigation task that focuses on
the model’s spatial reasoning capabilities.

We adopt a multi-step RL framework similar to Zhai
et al. (2024a), by instantiating RL after running SFT on

1We use “memorization” the refer a model’s capacity to gener-
ate near-exact copies of training examples when prompted based
on information present in the training dataset. This definition ex-
plicitly excludes bitwise or codewise replication of training data
within the model itself.

SFT Memorizes, RL Generalizes

2024a). SFT adapts pre-trained models to downstream
tasks by training them on task-specific, often instruction-
formatted datasets. Previous work, such as FLAN (Wei
et al., 2022a), demonstrates that fine-tuning on diverse
instruction-tuning datasets significantly enhances zero-shot
performance on unseen tasks. Furthermore, LIMA (Zhou
et al., 2024a) shows that supervised fine-tuning acts as
a “format teacher” effectively adapting the model’s re-
sponses to a desired format while leveraging the capabil-
ities of pre-trained LLMs. In contrast, RL (Ziegler et al.,
2019; Ouyang et al., 2022; Sun et al., 2024; Ramamurthy
et al., 2023; Abdulhai et al., 2023; Zhou et al., 2024b;
Zhai et al., 2024a) has been primarily used to align mod-
els with human preferences or training the foundational
model to solve a specific task (Abdulhai et al., 2023; Zhou
et al., 2024b; Zhai et al., 2024a; Chen et al., 2024b). Our
work differs from prior studies, as we aim to comparatively
analyze the generalization and memorization of SFT and
RL on both LLM and VLM, while previous studies have
focused primarily on only one of these two post-training
methods (or only study LLM or VLM) or on only one post-
training method.

Memorization and generalization in LLM/VLM. Sev-
eral studies have examined the interplay between memo-
rization and generalization in neural networks (Han et al.,
2022; Carlini et al., 2022; Yang et al., 2023).
In LLMs,
memorization can manifest as the model memorizing the
training data (Carlini et al., 2022; Jiang et al., 2024;
Kang et al., 2024), while generalization reflects the di-
vergence between the model’s output distribution and the
pre-training data distribution (Zhang et al., 2023). Prior
studies suggest that LLMs exhibit more overfitting on sim-
pler, knowledge-intensive tasks and greater generalization
on more complex, reasoning-intensive ones (Wang et al.,
2024; Qi et al., 2024). For example, recent studies (Ye
et al., 2024; Allen-Zhu, 2024; Allen-Zhu & Li, 2023a;b;
2024; Tong et al., 2024b) have demonstrated that LLMs
develop reasoning skill sets beyond their training data by
pre-computing reasoning graphs before autoregressive gen-
eration, which provides compelling evidence of generaliza-
tion. Our study takes a different approach by investigating
the role of different post-training paradigms on memoriza-
tion versus generalization in the context of textual ruled-
based and visual variants. We conduct comparative studies
in both unimodal (LLM) and multimodal (VLM) settings,
and demonstrate that RL leads to better generalization per-
formance than SFT.

Scaling up inference-time compute. Recent research
has increasingly focused on scaling up inference-time com-
putation to improve model performance (Wei et al., 2022b;
Yao et al., 2024; Snell et al., 2024; Jaech et al., 2024).
Early studies (Wei et al., 2022b; Yao et al., 2024) prompted

Figure 1: A comparative study of RL and SFT on the vi-
sual navigation environment V-IRL (Yang et al., 2024a)
for OOD generalization. OOD curves represent perfor-
mance on the same task, using a different textual action
space. See detailed descriptions of the task in Section 5.1.

the backbone model (Dubey et al., 2024), using the se-
quential revision formulation (Snell et al., 2024).
In
both GeneralPoints and V-IRL, we observe that
RL learns generalizable rules (expressed in text), where
in-distribution performance gains also transfer to unseen
rules. In contrast, SFT appears to memorize the training
rules and does not generalize (see Figure 1 for an example).
Beyond textual rule-based generalization, we further inves-
tigate generalization in the visual domain and observe that
RL also generalizes to visual OOD tasks, whereas SFT con-
tinues to struggle. As a by-product of the visual OOD gen-
eralization capability, our multi-turn RL approach achieves
state-of-the-art performance on the V-IRL mini bench-
mark, by +33.8% (44.0%→77.8%) (Yang et al., 2024a),
highlighting the generalization capability of RL. To un-
derstand how RL affects the visual abilities of a model,
we conducted additional analysis on GeneralPoints,
revealing that training RL with an outcome-based reward
function (Cobbe et al., 2021) improves visual recognition
capabilities. Although RL exhibits superior generalization
compared to SFT, we show that SFT is still necessary to
stabilize the model’s output format, enabling RL to achieve
its performance gains. Last but not least, we observe that
scaling up the inference time compute by increasing the
number of maximal steps leads to better generalization.

2. Related Works

Post-training. Post-training is crucial
for enhancing
model performance (Zhang et al., 2022; Hoffmann et al.,
2023; OpenAI, 2023b; Google, 2023; Touvron et al., 2023).
This stage commonly utilizes large-scale supervised fine-
tuning (SFT) (Radford et al., 2018; Brown et al., 2020;
Radford et al., 2021; Wei et al., 2022a; Chung et al.,
2022; Zhou et al., 2024a) and/or reinforcement learning
(RL) (Ziegler et al., 2019; Ouyang et al., 2022; Sun et al.,
2024; Abdulhai et al., 2023; Zhou et al., 2024b; Zhai et al.,

2

In-DistributionOut-of-DistributionSFTRLSFT Memorizes, RL Generalizes

models to generate intermediate reasoning steps and extend
the responses before producing a final answer. Subsequent
work (Zelikman et al., 2022; Feng et al., 2023; Tian et al.,
2024; Chen et al., 2024a; Snell et al., 2024) has demon-
strated that fine-tuning verifiers during inference improves
model accuracy, effectively utilizing test-time computation.
Notably, recent findings (Jaech et al., 2024; DeepSeekAI
et al., 2025) reveal “scaling laws” for inference-time com-
pute, highlighting significant performance gains with in-
creased computational resources. Our work builds upon
these findings in two ways. First, we integrate insights from
inference-time verification into a multi-turn RL formula-
tion that allows the model to identify and correct its errors.
Second, we examine the impact of inference-time verifica-
tion on RL generalization, demonstrating that scaling up
inference-time verification (in terms of the maximum num-
ber of verification steps) is a key for RL to generalize.

Improving visual capability in VLMs. While VLMs
have demonstrated remarkable skill across a wide range of
challenging tasks, such as solving advanced college exam
questions (Lu et al., 2023; Yue et al., 2024a;b) and spatial
understanding tasks (Yang et al., 2024a;b), they also ex-
hibit limitations in visual perception (Zhai et al., 2024a;b;
Tong et al., 2024c;d; Rahmanzadehgervi et al., 2024). Prior
efforts to enhance VLMs’ visual perception include com-
bining multiple visual encoders (Tong et al., 2024d; Kar
et al., 2025; Tong et al., 2024a), curating high-quality SFT
data (Chen et al., 2023; Liu et al., 2024; Tong et al., 2024a),
and improving the SFT training recipe by unfreezing the vi-
sual backbone (Liu et al., 2023; Tong et al., 2024a). While
these prior works primarily focus on experiments during
the SFT stage, our work demonstrates that RL can also im-
prove visual perception.

3. Preliminaries

Standard RL terminology. We consider finite horizon
decision making, and adopt standard notation from the
classical RL literature (Sutton & Barto, 2018; Agarwal
et al., 2019), where S denotes the state space, A denotes the
action space, r : S × A → R denotes the reward function,
and T denotes the maximum number of steps per episode.
The goal is to learn a policy π : S → A that maximizes
(cid:105)
the overall return maxπ∈Π Eπ
, where rt denotes
r(st, at). Without loss of generality, we use π(a|s) ∈ [0, 1]
to denote probability of π choosing a at s.

t=0 rt

(cid:104)(cid:80)T

Adapting RL terminology to LLM/VLM with a verifier.
We adopt a multi-turn RL setting for foundation model
training (Zhai et al., 2024a). Let V represent the discrete
and finite vocabulary (token) space. The input and output
text spaces are denoted by V m and V n respectively, where

3

m and n are the maximum token length of the input se-
quence vin and output sequence vout. For models requiring
visual inputs (VLM), we define O as the space of all RGB
images. The state space, denoted by S, is defined as S :=
V m×O for VLM, and S := V m for LLM. The action space
A is defined as A := V n. We use VER : V n → R × V k to
denote a verifier, which evaluates the outcome of vout and
generates an outcome-based reward function (Cobbe et al.,
2021; Hosseini et al., 2024; Snell et al., 2024; Setlur et al.,
2024) r along with textual information vver. Mathemati-
cally, at time t, VER(vout
t ). Similar to Zhai
et al. (2024a), we treat the model with parameter θ as our
policy network πθ : S → V n, and adopt PPO (Schulman
et al., 2017) as the backbone RL algorithm for updating πθ.

t ) (cid:55)→ (rt, vver

Sequential revision. For modeling the state-action tran-
sition, we adopt the sequential revision formulation (Snell
et al., 2024). Specifically, at time step t = 0 the initial
input vin
0 consists of the system prompt. For subsequent
time steps (t ≥ 1), the input prompt vin
t comprises the sys-
tem prompt concatenated with all prior model and verifier
outputs, denoted by [vout
k=0. An illustration of the
sequential revision is provided in Figure 2 (also see Figure
5 of Snell et al. (2024)), and an example of the state-action
transition is shown in Figure 3.

k , vver

k ]t−1

4. Evaluation Tasks

To evaluate the generalization of different post-training
methods, we select two tasks that each offer rule and vi-
sual variations. The first task, GeneralPoints, is a
new environment we have designed that allows assessment
of arithmetic reasoning abilities (Section 4.1). The second
task, V-IRL (Yang et al., 2024a), is chosen to examine the
model’s reasoning capabilities in an open-world visual nav-
igation domain (Section 4.2).

4.1. The General Points Environment

Our original GeneralPoints environment, instantiated
on top of the Points24 environment (Zhai et al., 2024a),
is designed to evaluate generalization of arithmetic reason-
ing. Each state s of the environment contains 4 cards, de-
scribed as text (in the GP-L variant) or presented as an im-
age (in the GP-VL variant); see Figure 2 left for a visual
example of GeneralPoints. The goal is to produce an
equation that equals a target number (24 by default) us-
ing all 4 numbers from the cards exactly once. Detailed
examples of the state-action transitions are provided in Ap-
pendix A.2. Note that when input from GeneralPoints
is presented in an image (GP-VL), it naturally introduces
additional visual challenges requiring the VLM to recog-
nize all cards before solving the equation.

SFT Memorizes, RL Generalizes

Figure 2: An example of the sequential revision formulation with a verifier. The model generate the next answer vout
on all previous answers and information (vout

t , 0 ≤ i ≤ t) from the verifier.

, vver

i

t+1 conditioned

System Prompt (vin
0 )

[Task Description] You are an expert in {task name}, you are observing {purely language/vision-language
inputs + <image>}. You are currently at {state related info}. Please follow {tasks rules}.

[Output] Your response should be a valid json file in the following format:
{task related information and answer}

Appending previous model and verifier outputs to obtain vin
t

t = [vout
vin

0 , vver

0 , vout

1 , vver

1 , . . . , vout

t−1, vver

t−1]

Model output (vout

t

) and Verifier Output (vver

t

)

{Task related json outputs}, {You success/fail}.

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1

k=0

(cid:1)

▷ vin

t+1 = concat(vin

t , vout
t

, vver
t )

Figure 3: An template of our prompt update for constructing vin
the purple parts denote the state (st) specific info. The blue and red describe the output from the model and verifier, respectively.

t+1. The brown parts marks the task and related information, and

study whether

variations. To

Rule
the model
learns arithmetic operations or simply memorizes the
post-training data, we introduce rule variations
in
GeneralPoints. These variations consist of interpret-
ing the symbols 'J', 'Q', and 'K' either as '11', '12', and
'13', respectively, or all as the same number '10'. These
variations ensure a rigorous evaluation of the model’s
ability to generalize arithmetic reasoning across diverse
settings. Each rule is specified as text in the input prompt,
see the {tasks rules} part in Figure 3. For studying ruled
based generalization, we post-train the model using one
rule, then evaluate using a different rule.

Visual variations. The GeneralPoints environment
can also be naturally customized to evaluate generalization
across visual variants. Since the major visual challenge is
to recognize the number of each card, agnostic to the the
color of the cards, we consider the cards with different col-
ors as visual variants of the task. In the visual generaliza-
tion setting, we train the model using cards of one color,
then test OOD performance using the other color.

4.2. The V-IRL Environment

While the GeneralPoints environment is designed to
assess arithmetic reasoning abilities, we further utilize the
V-IRL environment (Yang et al., 2024a) to study spatial
reasoning ability in an open-world navigation domain that

4

uses realistic visual input. As in GeneralPoints we
consider two versions of the environment, one (V-IRL-L)
that consists of pure language descriptions,2 and another
(V-IRL-VL) that includes vision-language input. The ma-
jor visual challenge in V-IRL involves recognizing differ-
ent landmarks from the visual observation3 before taking
an action. The goal is to navigate to a target location by
following a set of instructions that contain spatial informa-
tion. A detailed example of one environment step is shown
in Appendix B.2.

Rule variations. To evaluate whether the model pos-
sesses spatial knowledge or simply memorizes post-
training data, we consider two distinct action space con-
figurations. The first variant utilizes an absolute orienta-
tion action space, which includes {'north', 'northeast', 'east',
'southeast', 'south', 'southwest', 'west', 'northwest'}. The
second variant employs a relative orientation action space,
containing {'left', 'right', 'slightly left', 'slightly right'}. This
relative configuration adjusts the current orientation by 90
degrees or 45 degrees to the left or right, respectively. An

2The visual input can be parsed into pure text description, see
more details in Yang et al. (2024a) and an illustration of pure text
the version in Figure 14.

3See Figure 4, the model needs to recognize landmarks like
The Dutch, Lola Taverna, and Shuka from the visual observa-
tion, and relate these landmarks with the textual instructions for
taking the right action.

Q: Compute 24 using these four cards: [5, 4, 10, 7](V)LM10+7+4+5(7-4)*10-6(7-5)*10+4wrong calculation

Reward: -1illegal number used

Reward: -5correct answer

Reward: +10Verifier Info:SFT Memorizes, RL Generalizes

Figure 4: Demonstration of one navigation task in V-IRL. Agent navigates from place to place following the given linguistic
navigation instructions in V-IRL. The navigation procedure is shown at the top, with the navigation instructions displayed below.
Visual observation-related information is highlighted in green, while action-related information is marked in orange.

overview of a navigation task in V-IRL is provided in Fig-
ure 4, and a detailed state-action transition in V-IRL is
provided in Figure 13 (in Appendix B.2).

Visual variations. The key visual challenge in V-IRL is
to recognize landmarks from the visual observations (e.g.,
the green parts in Figure 4). Since the V-IRL environ-
ment contains visual observations from different cities, we
can assess visual generalization in V-IRL by training the
model to navigate in one location and then evaluate its per-
formance in different locations.

5. Results

In this section, we present experiments that investigate
the generalization abilities induced by post-training with
RL and SFT. We adopt Llama-3.2-Vision-11B (Dubey
Following the
et al., 2024) as the backbone model.
standard pipelines of RLHF (Ouyang et al., 2022) and
RL4VLM (Zhai et al., 2024a), we initialize the model with
SFT before running RL. We specifically study the follow-

ing questions. Section 5.1: how does SFT or RL affect
the model’s generalization to different rules? Section 5.2:
when the model contains a visual component, how does
RL/SFT affect its generalization to different visual vari-
ants? Section 5.3: how does RL/SFT affect visual recogni-
tion capability in a VLM? Section 5.4: what role does SFT
play in RL training? Section 5.5: how does the number of
verification iterations affect generalization?

5.1. Generalization across Rules

We evaluate the performance of different post-training
methods on GeneralPoints and V-IRL, each of which
has a pure language (-L) and a vision-language (-VL) vari-
ant, and each encompassing rule variations. For each task,
we separately scale the training compute for RL and SFT
on a single rule. We consider the results on the trained rule
as in-distribution (ID) performance, whereas results on the
unseen rules measures out-of-distribution (OOD) general-
ization. In GeneralPoints, the ID case treats all 'J', 'Q',
'K' as 10, and the OOD cases interprets them as 11, 12, and
13. As for V-IRL, the ID case adopts the absolute orienta-

5

ShukaMediterranean⭐First, turn slightly right towards the northeast and walk a short distance until you reach the next intersection, where you‘ll see The Dutch on your right.Next, make a sharp left turn to head northwest. Continue for a while until you reach the next intersection, where Lola Taverna will be on your right.Finally, turn slightly right to face northeast and walk a short distance until you reach your destination, Shuka, which will be on your right.TheDutchAmericanrestaurantLolaTavernaGreek[OBSERVATION]“Start!”[ACTION]“Turntonortheast.”[OBSERVATION]“SeeLola Taverna onmyright.”[ACTION]“Leftturntonorthwest.”[OBSERVATION]“SeeShuka onmyright.”[ACTION]“Stop.”[OBSERVATION]“SeeTheDutchonmyright.”[ACTION]“Leftturntonorthwest.”SFT Memorizes, RL Generalizes

Figure 5: Success rate (%) - GFLOPs trendlines for RL and SFT on GeneralPoints and V-IRL. The top row
shows in-distribution performance, while the bottom row shows out-of-distribution performance. Results are presented
for both pure language (-L) and vision-language (-VL) variants of each task. For GeneralPoints, we report the
episode success rate, while for V-IRL, we report per-step accuracy with overall success rate in Figures 1 and 18. Detailed
evaluation setups (and curve smoothing) are provided in Appendix C.3.

tion coordinate system and the OOD case uses the relative
orientation action space. Other details and additional ex-
perimental setup can be found in Appendix C.

tasks,

RL generalizes, SFT memorizes. As illustrated in Fig-
ure 5, RL consistently improves OOD performance on
all
including both unimodal (LLM) and multi-
modal (VLM). Specifically, Figure 6 demonstrates that
RL achieves an increase of +3.5% on GP-L (11.5% →
15.0%) and +11.0% on V-IRL-L (80.8% → 91.8%).
Even with the additional challenge of visual recognition in
the VLM, RL maintains consistent performance improve-
ments of +3.0% (11.2% → 14.2%) on GP-VL and +9.3%
(35.7% → 45.0%) on V-IRL-VL, respectively.
In con-
trast, SFT consistently exhibits performance degradation
across all OOD evaluations on all tasks: -8.1% on GP-L
(11.5% → 3.4%), -79.5% on V-IRL-L (80.8% → 1.3%),
-5.6% (11.2% → 5.6%) on GP-VL, and -33.2% (35.7% →
2.5%) on V-IRL-VL.

5.2. Generalization in Visual Out-of-Distribution Tasks

Section 5.1 demonstrates that RL yields generalization
across rule variations, whereas SFT exhibits the opposite
trend. Since VLMs also incorporate a visual modality, we
next study the effects of visual variation in OOD general-
ization. For GeneralPoints, we train the VLM using
the black suits (♠, ♣) and test out-of-distribution perfor-

mance on the red suits (♥, ♦). For V-IRL, we train the
model on routes collected in New York City and evalu-
ate it on the original V-IRL VLN mini benchmark (Yang
et al., 2024a) containing routes from various cities world-
wide (see Appendix B.1 for details). Note that the rules
remain consistent across experiments in this section.

Specifically,

RL generalizes in visual OOD tasks. As shown in Fig-
ure 7, we observe that RL still generalizes in visual OOD
tasks, while SFT continues to suffer.
in
GP-VL and VIRL-VL, RL achieves performance improve-
ments of +17.6% (23.6% → 41.2%), +61.1% (16.7% →
77.8%), whereas SFT suffers from performance decreases
of -9.9% (23.6% → 13.7%) and -5.6% (16.7% → 11.1%).
As a byproduct of this visual OOD study, we also show
that our multi-turn RL formulation improves the state-of-
the-art results (see Table 5 of Yang et al. (2024a)) on the
V-IRL mini benchmark by +33.8% (44.0% → 77.8%).
Notably, unlike the previous state-of-the-art approach re-
ported in V-IRL, which relies on a two stage VLM-LLM
collaboration technique and tailored prompt engineering
on closed-sourced model (OpenAI, 2023a), our end-to-end
RL approach enables an open-sourced model (Dubey et al.,
2024) to reach superior performance.

5.3. RL Improves Visual Capabilities

Building upon the above observation that VLMs trained
with RL generalize to visual OOD tasks (Section 5.2), we

6

0.00.51.01e10285276100GP-L0241e9708090100V-IRL-L2461e1014284155GP-VL241e1069778593V-IRL-VL0.00.51.01e101612170241e903365982461e10381419241e100193756In-distributionOut-of-distributionTraining Computation (GFLOPs)pure languagewith visionInitializationSFTRLSFT Memorizes, RL Generalizes

Figure 6: Comparison of out-of-distribution performance under rule variants. We report the success rate for
GeneralPoints and per-step-accuracy for V-IRL. For each subplot, RL and SFT are trained with equal computa-
tion, and their shared initial checkpoint (marked as Init) is set as baseline. Detailed setups are provided in Appendix C.3.

Figure 7: Comparison of out-of-distribution performance under visual variants. Similar to Figures 5 and 6, we present
both the performance dynamics (shown as lines) and final performance (shown as bars) for visual out-of-distribution
evaluations. The previous state-of-the-art on V-IRL VLN mini benchmark (Yang et al., 2024a) is marked in orange.
Detailed evaluation setups (and curve smoothing) are provided in Appendix C.3.

consider a natural follow-up question: How does RL af-
fect VLMs’ visual capabilities? To study this question, we
conducted additional ablation studies in the GP-VL envi-
ronment to investigate the OOD performance of RL and
SFT, along with the model’s visual recognition accuracy,
in terms of recognizing the 4 cards from the input image.
In particular, we study how scaling post-training compute
via RL/SFT both affects generalization in rule-based OOD
(Figure 8 left), and visual recognition accuracy and visual
OOD (Figure 8 right).

5.4. The Role of SFT for RL Training

Despite the superiority of RL in generalizing the model’s
reasoning and visual capabilities, as discussed previously,
the experimental pipeline still instantiates RL after SFT.
In this subsection, we focus on another key question: Is
SFT necessary for RL training? To answer this question,
we conduct additional experiments that directly apply end-
to-end RL to post-train the base model Llama3.2 using
GeneralPoints in the purely language case (Figure 9).

Scaling RL improves visual recognition accuracy in
VLM training. As shown in Figure 8, we observe that
the VLM’s visual recognition accuracy largely affects
the overall performance, which was similarly observed
in Zhong et al. (2024). In addition, scaling up RL compute
also improves visual recognition accuracy, as a byproduct
of its generalization capability, while scaling SFT deteri-
orates both visual recognition accuracy and overall per-
formance. Additional experimental results are provided
in Figures 16 and 17 of Appendix D.1.

7

SFT is necessary for RL training when the backbone
model does not follow instructions. Figure 9 shows
that without SFT, all end-to-end RL runs fail to improve.
More specifically, we observe that without SFT, the base
model suffers from poor instruction following capability.
A detailed failure case is provided in Figure 20 (in Ap-
pendix D.3), revealing that the base Llama-3.2-Vision-11B
model tends to generate long, tangential, and unstructured
responses. This issue makes it impossible to retrieve task-
related information and rewards for RL training. Note that
due to the difference in backbone model, our results do not
contradict with DeepSeekAI et al. (2025), which suggests
that SFT is unnecessary for downstream RL training.

03691215SFTInitRL3.4%11.5%15.0%GP-L0204060801001.3%80.8%91.8%VIRL-L0481216205.6%11.2%14.2%GP-VL010203040502.5%35.7%45.0%V-IRL-VLOOD success rate (%)24GFLOPs1e1010203040SFTInitRL1020304013.7%23.6%41.2%2.55.07.5GFLOPs1e1020406080SFTInitRL2040608011.1%16.7%77.8%V-OOD Success Rate (%)GP-VLV-IRL-VLInitializationSFTRLPrevious SOTASFT Memorizes, RL Generalizes

Figure 8: Recognition vs. success rate for RL and SFT under different variants in GP-VL. We report both in-
distribution (red) and OOD (blue) performance of recognition (y-axis) and episode success rate (x-axis). We denote the
training compute of each data point via transparency (color bar) while connected (⋆-◦) pairs are evaluated using same
checkpoints. As scaling up post-training compute, RL improves both recognition and overall accuracy, while SFT shows
opposite effect.

Figure 9: RL experiments on GP-L without SFT initial-
ization. All trials fail due to poor instruction following
capability of the base model.

5.5. Role of Verification Iterations

Verification serves as another crucial component in our
multi-step training and evaluation pipeline (see Figures 2
and 3). To validate its necessity and better understand its
effect, we conduct RL experiments with different verifica-
tion iterations {1, 3, 5, 10} using GP-L (Figure 10).

Scaling up verification improves
generalization.
In Figure 10, we observe that RL generalizes better with
more verification steps. More specifically, under the same
computational budget across all experiments, we observe
improvements of +2.15% (3 steps), +2.99% (5 steps),
+5.99% (10 steps).
in the case with one
In contrast,
verification step, we only observe a marginal improvement
of +0.48% in OOD performance improvement.

6. Conclusion, Discussion, and Limitations

In this paper, we present a comprehensive analysis of the
generalization effects of foundation model post-training
techniques, specifically RL and SFT. Through extensive

Figure 10: In-distribution vs. OOD performance growth
on GP-L. We record RL experiments with different num-
ber of verification iterations (VIter) as scaling up training
compute (color transparency).

experiments on the GeneralPoints and V-IRL tasks,
we demonstrated that RL exhibits superior performance
in learning generalizable knowledge, while SFT tends to
merely memorize the training data, across both the rule
and visual variations. This phenomenon consistently oc-
curs across multimodal arithmetic and spatial reasoning ca-
pabilities. In addition, we studied the effect of RL on vi-
sual recognition, the role of SFT, and the role of verification
steps. During our study, two challenges were not resolved.

Failure of SFT on GP-VL.
In Figure 5 for GP-VL,
we observe that SFT fails to achieve a comparable in-
distribution performance with RL. To mitigate the vari-
ance introduced by hyperparameter choices, we addition-
ally conduct 10 more experiments with different learning
rates and tunable components (Figure 16), none of which
exhibits a strong increasing trend like RL (Figure 17).

8

010203040506030507090Rule Variants010203040506030507090Visual Variants4e105e106e107e108e10Computation (GFLOPs)GP-VL Success Rate (%)Recognition Accuracy (%)Out-of-distributionIn-distributionSFTRLInit0.00.20.40.60.81.01.2Computation (GFLOPs)1e100.00.10.20.30.4Success Rate (%)1e-62e-65e-70123456Out-of-distribution Growth (%)4681012In-distribution Growth (%)VIter 1VIter 3VIter 5VIter 103e96e9Computation (GFLOPs)SFT Memorizes, RL Generalizes

Given our observation that scaling up SFT degrades visual
recognition capabilities (Figure 8), we hypothesize that
SFT locally overfits to reasoning tokens while neglecting
recognition tokens, possibly due to the higher frequency of
reasoning tokens (see Figure 11 as example). We leave fur-
ther investigation to future work.

Limits of RL in corner cases. As discussed in Sec-
tion 5.4, SFT is necessary for effective RL training on
Llama-3.2. We investigate applying RL to an overly-tuned
SFT checkpoint. As demonstrated in Figure 19, RL is un-
able to recover out-of-distribution performance when start-
ing from such a checkpoint. Example failure cases are il-
lustrated in Figure 21, where the model collapses to the
training rule. These results, together with findings in Sec-
tion 5.4, indicate that RL has limited effectiveness when
applied to extremely underfit or overfit initial checkpoints.
Further research is needed to delineate the conditions under
which SFT facilitates effective RL.

Impact Statement

This paper presents work aimed at advancing the field of
Machine Learning. While the study includes tasks such as
GeneralPoints, which is a synthetic environment, and
V-IRL, a real-world map simulator, our work is confined
to controlled research settings. The V-IRL environment is
designed as a simulated proxy for real-world tasks, but no
deployment or interaction with actual real-world systems or
data was involved. The methods, environments, and tasks
investigated in this study were constructed to advance our
understanding of model generalization without introducing
any foreseeable societal or ethical implications.

Acknowledgements

YZ would like to thank Xiaoxuan Feng for beautifying Fig-
ure 4. We would like to thank Jincheng Mei and Doina Pre-
cup for feedbacks on earlier manuscripts. Yi Ma would like
to acknowledge support from the joint Simons Foundation-
NSF DMS grant #2031899, the ONR grant N00014-22-1-
2102, the NSF grant #2402951, and also support from and
the HKU startup, the Hong Kong Center for Construction
Robotics Limited (HKCRC) Award 052245, and JC Club
of Hong Kong.

References

Abdulhai, M., White, I., Snell, C., Sun, C., Hong, J., Zhai,
Y., Xu, K., and Levine, S. LMRL Gym: Benchmarks for
multi-turn reinforcement learning with language models.
arXiv preprint arXiv:2311.18232, 2023. 2

Agarwal, A., Jiang, N., Kakade, S. M., and Sun, W. Re-

inforcement learning: Theory and algorithms. CS Dept.,
UW Seattle, Seattle, WA, USA, Tech. Rep, 32, 2019. 3

Allen-Zhu, Z.

ICML 2024 Tutorial: Physics of Lan-
guage Models, July 2024. Project page: https://
physics.allen-zhu.com/. 2

Allen-Zhu, Z. and Li, Y. Physics of language models: Part
3.1, knowledge storage and extraction. arXiv preprint
arXiv:2309.14316, 2023a. 1, 2

Allen-Zhu, Z. and Li, Y.

Physics of language mod-
els: Part 3.2, knowledge manipulation. arXiv preprint
arXiv:2309.14402, 2023b. 2

Allen-Zhu, Z. and Li, Y. Physics of language models: Part
3.3, knowledge capacity scaling laws. arXiv preprint
arXiv:2404.05405, 2024. 2

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R.,
Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosse-
lut, A., Brunskill, E., et al. On the opportunities and risks
of foundation models. arXiv preprint arXiv:2108.07258,
2021. 1

Bousquet, O. and Elisseeff, A. Algorithmic stability and

generalization performance. volume 13, 2000. 1

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan,
J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sas-
try, G., Askell, A., et al. Language models are few-shot
learners. Advances in neural information processing sys-
tems, 33:1877–1901, 2020. 1, 2

Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F.,
and Zhang, C. Quantifying memorization across neu-
ral language models. arXiv preprint arXiv:2202.07646,
2022. 2

Chen, G., Liao, M., Li, C., and Fan, K. AlphaMath al-
most zero: Process supervision without process. arXiv
preprint arXiv:2405.03553, 2024a. 3

Chen, J., Han, X., Ma, Y., Zhou, X., and Xiang, L. Un-
lock the correlation between supervised fine-tuning and
reinforcement learning in training code large language
models. arXiv preprint arXiv:2406.10305, 2024b. 2

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J.,
Zhao, F., and Lin, D. ShareGPT4V: Improving large
multi-modal models with better captions. arXiv preprint
arXiv:2311.12793, 2023. 3

Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y.,
Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma,
S., et al. Scaling instruction-finetuned language models.
arXiv preprint arXiv:2210.11416, 2022. 2

9

SFT Memorizes, RL Generalizes

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., et al. Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168, 2021. 2, 3

DeepSeekAI et al. DeepSeek-R1:

Incentivizing rea-
soning capability in LLMs via reinforcement learning,
URL https://arxiv.org/abs/2501.
2025.
12948. 1, 3, 7

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,
A., et al. The Llama 3 Herd of models. arXiv preprint
arXiv:2407.21783, 2024. 2, 5, 6

Feng, X., Wan, Z., Wen, M., McAleer, S. M., Wen, Y.,
Zhang, W., and Wang, J. AlphaZero-like tree-search can
guide large language model decoding and training. arXiv
preprint arXiv:2309.17179, 2023. 3

Introducing Gemini:

Google, D.
and most
capable AI model,
https://blog.google/technology/ai/
google-gemini-ai/. 1, 2

2023.

Our

largest
URL

Han, J., Zhan, H., Hong, J., Fang, P., Li, H., Petersson,
L., and Reid, I. What images are more memorable to
machines? arXiv preprint arXiv:2211.07625, 2022. 2

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E.,
Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A.,
Welbl, J., Clark, A., et al. Training compute-optimal
large language models. NeurIPS, 2023. 2, 18

Hosseini, A., Yuan, X., Malkin, N., Courville, A., Sordoni,
A., and Agarwal, R. V-STar: Training verifiers for self-
taught reasoners. In First Conference on Language Mod-
eling, 2024. URL https://openreview.net/
forum?id=stmqBSW2dV. 3

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky,
A., Low, A., Helyar, A., Madry, A., Beutel, A., Car-
ney, A., et al. OpenAI o1 system card. arXiv preprint
arXiv:2412.16720, 2024. 1, 2, 3

Jiang, M., Liu, K. Z., Zhong, M., Schaeffer, R., Ouyang,
S., Han, J., and Koyejo, S. Investigating data contami-
nation for pre-training language models. arXiv preprint
arXiv:2401.06059, 2024. 2

Kang, K., Setlur, A., Ghosh, D., Steinhardt, J., Tomlin, C.,
Levine, S., and Kumar, A. What do learning dynamics
reveal about generalization in LLM reasoning? arXiv
preprint arXiv:2411.07681, 2024. 1, 2

Kar, O. F., Tonioni, A., Poklukar, P., Kulshrestha, A., Za-
mir, A., and Tombari, F. Brave: Broadening the vi-
sual encoding of vision-language models. In European

10

Conference on Computer Vision, pp. 113–132. Springer,
2025. 3

Liu, H., Li, C., Li, Y., and Lee, Y. J.
lines with visual instruction tuning.
arXiv:2310.03744, 2023. 3

Improved base-
arXiv preprint

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S.,
Improved rea-
URL

and Lee, Y.
soning, ocr, and world knowledge, 2024.
https://llava-vl.github.io/blog/
2024-01-30-llava-next/. 3

LLaVA-NeXT:

J.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H.,
Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Math-
Vista: Evaluating mathematical reasoning of foundation
models in visual contexts. ICLR, 2023. 3

OpenAI. GPT-4, 2023a. URL https://openai.com/

research/gpt-4. 6

OpenAI. GPT-4 technical report. arXiv, pp. 2303–08774,

2023b. 1, 2

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright,
C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray,
A., et al. Training language models to follow instructions
with human feedback. In NeurIPS, 2022. 2, 5

Qi, Z., Luo, H., Huang, X., Zhao, Z., Jiang, Y., Fan, X.,
Lakkaraju, H., and Glass, J. Quantifying generalization
complexity for large language models. arXiv preprint
arXiv:2410.01769, 2024. 2

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I.,
et al. Improving language understanding by generative
pre-training. 2018. 2

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark,
J., et al. Learning transferable visual models from natu-
ral language supervision. In International conference on
machine learning, pp. 8748–8763. PMLR, 2021. 2

Rahmanzadehgervi, P., Bolton, L., Taesiri, M. R., and
In
Nguyen, A. T. Vision language models are blind.
Proceedings of the Asian Conference on Computer Vi-
sion, pp. 18–34, 2024. 3

Ramamurthy, R., Ammanabrolu, P., Brantley, K., Hessel,
J., Sifa, R., Bauckhage, C., Hajishirzi, H., and Choi, Y.
Is reinforcement learning (not) for natural language pro-
cessing: Benchmarks, baselines, and building blocks for
In The Eleventh
natural language policy optimization.
International Conference on Learning Representations,
2023. URL https://openreview.net/forum?
id=8aHzds2uUyB. 2

SFT Memorizes, RL Generalizes

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347, 2017. 3, 18

Setlur, A., Nagpal, C., Fisch, A., Geng, X., Eisenstein, J.,
Agarwal, R., Agarwal, A., Berant, J., and Kumar, A. Re-
warding progress: Scaling automated process verifiers
for LLM reasoning. arXiv preprint arXiv:2410.08146,
2024. 3

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling LLM test-
time compute optimally can be more effective than scal-
ing model parameters. arXiv preprint arXiv:2408.03314,
2024. 2, 3, 18

Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan,
C., Gui, L., Wang, Y.-X., Yang, Y., Keutzer, K., and
Darrell, T. Aligning large multimodal models with fac-
In Ku, L.-W., Martins, A.,
tually augmented RLHF.
and Srikumar, V. (eds.), Findings of the Association
for Computational Linguistics: ACL 2024, pp. 13088–
13110, Bangkok, Thailand, August 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.
findings-acl.775. URL https://aclanthology.
org/2024.findings-acl.775. 2

Sutton, R. S. and Barto, A. G. Reinforcement Learning: An

Introduction. MIT press, 2018. 3

Tian, Y., Peng, B., Song, L., Jin, L., Yu, D., Mi, H.,
and Yu, D. Toward self-improvement of LLMs via
imagination, searching, and criticizing. arXiv preprint
arXiv:2404.12253, 2024. 3

Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M.,
Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., et al.
Cambrian-1: A fully open, vision-centric exploration of
multimodal LLMs. In NeurIPS, 2024a. 3

Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K.,
Rabbat, M., LeCun, Y., Xie, S., and Liu, Z. Metamorph:
Multimodal understanding and generation via instruc-
tion tuning. arXiv preprint arXiv:2412.14164, 2024b.
2

Tong, S., Jones, E., and Steinhardt, J. Mass-producing fail-
ures of multimodal systems with language models.
In
NeurIPS, 2024c. 3

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S.
Eyes wide shut? Exploring the visual shortcomings of
multimodal LLMs. In CVPR, 2024d. 3

Wang, X., Antoniades, A., Elazar, Y., Amayuelas, A.,
Albalak, A., Zhang, K., and Wang, W. Y. Gener-
alization vs memorization: Tracing language models’
arXiv preprint
capabilities back to pretraining data.
arXiv:2407.14985, 2024. 2

Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W.,
Lester, B., Du, N., Dai, A. M., and Le, Q. V.
Finetuned language models are zero-shot
learners.
In International Conference on Learning Representa-
tions, 2022a. URL https://openreview.net/
forum?id=gEZrGCozdqR. 2

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F.,
Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought
prompting elicits reasoning in large language models.
Advances in Neural Information Processing Systems, 35:
24824–24837, 2022b. 2

Yang, J., Ding, R., Brown, E., Qi, X., and Xie, S. V-IRL:
Grounding virtual intelligence in real life. In European
conference on computer vision, 2024a. 1, 2, 3, 4, 6, 7,
13, 14, 15

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and
Xie, S. Thinking in space: How multimodal large lan-
guage models see, remember, and recall spaces. arXiv
preprint arXiv:2412.14171, 2024b. 3

Yang, Z., Lukasik, M., Nagarajan, V., Li, Z., Rawat, A. S.,
Zaheer, M., Menon, A. K., and Kumar, S. ResMem:
Learn what you can and memorize the rest. In Thirty-
seventh Conference on Neural Information Processing
Systems, 2023. URL https://openreview.net/
forum?id=HFQFAyNucq. 2

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao,
Y., and Narasimhan, K. Tree of thoughts: Deliberate
problem solving with large language models. Advances
in Neural Information Processing Systems, 36, 2024. 1,
2

Ye, T., Xu, Z., Li, Y., and Allen-Zhu, Z.

language models:

of
and the hidden reasoning process.
arXiv:2407.20311, 2024. 1, 2

Physics
Part 2.1, grade-school math
arXiv preprint

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G.,
Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. MMMU: A
massive multi-discipline multimodal understanding and
reasoning benchmark for expert AGI. In CVPR, 2024a.
3

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient founda-
tion language models. arXiv preprint arXiv:2302.13971,
2023. 2

Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S.,
Sun, Y., Yin, M., Yu, B., Zhang, G., et al. MMMU-
Pro: A more robust multi-discipline multimodal under-
standing benchmark. arXiv preprint arXiv:2409.02813,
2024b. 3

11

SFT Memorizes, RL Generalizes

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. STaR:
Advances
Bootstrapping reasoning with reasoning.
in Neural Information Processing Systems, 35:15476–
15488, 2022. 3

Zhai, Y., Bai, H., Lin, Z., Pan, J., Tong, S., Zhou, Y., Suhr,
A., Xie, S., LeCun, Y., Ma, Y., and Levine, S. Fine-
tuning large vision-language models as decision-making
agents via reinforcement learning. In The Thirty-eighth
Annual Conference on Neural Information Process-
ing Systems, 2024a. URL https://openreview.
net/forum?id=nBjmMF2IZU. 1, 2, 3, 5, 18

Zhai, Y., Tong, S., Li, X., Cai, M., Qu, Q., Lee, Y. J., and
Ma, Y. Investigating the catastrophic forgetting in mul-
In Confer-
timodal large language model fine-tuning.
ence on Parsimony and Learning, pp. 202–227. PMLR,
2024b. 3

Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals,
O. Understanding deep learning (still) requires rethink-
ing generalization. Communications of the ACM, 64(3):
107–115, 2021. 1

Zhang, C., Ippolito, D., Lee, K., Jagielski, M., Tramèr, F.,
and Carlini, N. Counterfactual memorization in neural
language models. Advances in Neural Information Pro-
cessing Systems, 36:39321–39362, 2023. 2

Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M.,
Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V.,
et al. Opt: Open pre-trained transformer language mod-
els. arXiv preprint arXiv:2205.01068, 2022. 2

Zhong, M., Zhang, A., Wang, X., Hou, R., Xiong, W., Zhu,
C., Chen, Z., Tan, L., Bi, C., Lewis, M., et al. Law of
the weakest link: Cross capabilities of large language
models. arXiv preprint arXiv:2409.19951, 2024. 7

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X.,
Efrat, A., Yu, P., Yu, L., et al. LIMA: Less is more for
alignment. Advances in Neural Information Processing
Systems, 36, 2024a. 2

Zhou, Y., Zanette, A., Pan, J., Levine, S., and Kumar, A.
ArCHer: Training language model agents via hierarchi-
cal multi-turn RL. arXiv preprint arXiv:2402.19446,
2024b. 2

Zhu, Z., Xue, Y., Chen, X., Zhou, D., Tang, J., Schuurmans,
D., and Dai, H. Large language models can learn rules.
arXiv preprint arXiv:2310.07064, 2023. 1

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Rad-
ford, A., Amodei, D., Christiano, P., and Irving, G. Fine-
tuning language models from human preferences. arXiv
preprint arXiv:1909.08593, 2019. 2

12

SFT Memorizes, RL Generalizes

A. Details on the General Points Environment

In this section, we demonstrate the design details for
GeneralPoints mentioned in Section 4.1. We first
present the data used for this environment (Appendix A.1).
Then, we show examples of the environment’s transition
dynamics (Appendix A.2), followed by a description of key
arguments and reward design specification (Appendix A.3).

A.1. Data

GeneralPoints card quadruples are sampled from a
deck of 52 standard poker cards. Each sampled quadruple
is guaranteed to have at least one solution equals the target
point, i.e. 24. We ensure this by using an expert solver
during the sampling process.

A.2. Detailed Examples on the Transition Dynamics

As shown in Figure 11 and Figure 12, we treat the system
prompt as vin
0 and then subsequently appending the future
outputs vout
1:t into the prompt for get-
ting the t + 1 output. Figure 11 provides an example with
the visual inputs, while Figure 12 shows the language only
case.

1:t and verifier info vver

A.3. Additional Eetails on the Environmental Design

Arguments. The GeneralPoints environment sup-
ports the following configurable arguments:

• Target point: Any positive integer

• Face cards rule: Two options

– 'J', 'Q', and 'K' all count as '10'
– 'J', 'Q', and 'K' count as '11', '12', and '13' respec-

tively

• Card sampling: Two options

shift experiments (Section 5.2), we train the model on black
suits ♠, ♣ and evaluate out-of-domain performance on red
suits ♥, ♦.

Reward design. An episode terminates when either a
correct equation is generated or the maximum verification
step of 5 is reached. The reward function is as follows:

• r = 5: For generating a legal equation that equals the

target point

• r = −1: For legal equations using each card once but

not equaling the target point

• r = −1: For exceeding maximum verification step

• r = −2: For legal equations containing numbers not

among the given choices

• r = −3: For all other illegal equations

In the vision-language variant (GeneralPoints-VL),
an additional penalty of r = −1.5 is applied when the agent
fails to correctly recognize the given cards.

B. Details on the V-IRL Environment

Similar to Appendix A, we present the design details for
V-IRL discussed in Section 4.2.
First, we introduce
the database used for this environment (Appendix B.1)
and demonstrate transition examples (Appendix B.2). We
then describe the environment by explaining its funda-
mental component—route. Finally, we outline our mod-
ifications and reward design choices made to adapt the
original V-IRL for reinforcement learning training (Ap-
pendix B.3).

– Sample 4 cards without replacement from a deck

of 52 poker cards

B.1. Data

– Sample at least one card from 'J', 'Q', and 'K'

• Card color: Three options

– Black suits only: ♣, ♠.
– Red suits only: ♥, ♦.
– All suits: ♠, ♥, ♣, ♦.

For all experiments, we fix the target point at 24. In Fig-
ure 5, training and in-domain evaluation use the rule where
face cards count as '10'. For out-of-domain evaluation, we
use the alternative face cards rule and require at least one
face card, forcing calculations with numbers above 10 that
are not encountered during training. For visual distribution

13

Leveraging the data collection pipeline of Yang et al.
(2024a), we construct a training database with 1000 unique
routes from New York City. We evaluate all rule-variant
experiments and visual in-distribution experiments using
randomly sampled routes from this database. For visual
out-of-distribution experiments, we directly adopt the VLN
mini benchmark from Yang et al. (2024a). This benchmark
consists of 18 distinct routes across nine cities: Milan, New
Delhi, Buenos Aires, London, Hong Kong, New York,4
Melbourne, Lagos, and San Francisco, with two routes per
city.

4These NYC routes in the VLN mini benchmark do not over-

lap with our training data.

SFT Memorizes, RL Generalizes

System Prompt (vin
0 )

[Task Description]
You are an expert 24 points card game player. You are observing these four cards
in the image. Note that 'J', 'Q', and 'K' count as '10', and each card must be used
once. Your goal is to output a formula that evaluates to 24 using numbers from the
cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}

Appending model and verifier outputs to obtain vin
t
t = [vout
vin
1 , . . . , vout

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

t

)

Model output (vout
{
"cards": ['A', '3', 'K', '6'],
"number": [1, 3, 13, 6],
"formula": "(1+6)*3+13=24",
}

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

Verifier Output (vver

t

)

You failed this trial because your formula is incorrect.

▷ vin

t+1 = concat(vin

t , vout

t , vver
t )

Figure 11: An example of our prompt update for constructing vin
t . This example provides an optional vision
input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts
denote the state (st) specific info. The blue and red describe the output from the model and verifier, respectively.

t+1 using vin

t and vver

t , vout

B.2. Detailed Examples on the Transition Dynamics

• Oracle information: Expert observation data for each

We provide detailed transition examples of the V-IRL en-
vironment in Figure 13 (vision and language) and Figure 14
(pure language).

B.3. Additional Details on the Environmental Design

Concept of route. The route serves as the fundamental
navigation object in the V-IRL environment. As illus-
trated in Figure 4, each route corresponds to a real-world
path with associated language instructions and visual sig-
nals. Using Figure 4 as an example, a route comprises:

• Destination: Shuka

• Starting point: Start

• Turning points: The Dutch, Lola Taverna

movable point

• Expert trajectory

• Instruction

Although the instructions in Figures 4, 13 and 14 are pre-
sented in different formats, they convey equivalent infor-
mation, with Figure 4 using natural language.

Simplification and arguments. We simplify the original
V-IRL design from Yang et al. (2024a) to better accom-
modate RL training. The modifications include eliminating
the 2-stage navigation pipeline that required a separate vi-
sual detector for street view processing, and removing on-
line queries to reduce training time and cost. Our V-IRL
environment contains 2 additional configuration arguments
compared with the original design:

• Straight road: Roads connecting turning points, start-

ing point, and destination

• Action space: two options

• Street views: 360-degree panoramic views at each

– Absolute direction:

movable point

"turn_direction(x)" where x∈{'north', 'northeast',

14

SFT Memorizes, RL Generalizes

System Prompt (vin
0 )

[Task Description]
You are an expert 24 points card game player. You are observing these four cards in the image. Note that 'J', 'Q',
and 'K' count as '11', '12', and '13' respectively, and each card must be used once. Your goal is to output a formula
that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Input]
Cards: ['A', '3', 'K', '6']

[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}

Appending model and verifier outputs to obtain vin
t
t = [vout
vin
1 , . . . , vout

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

t

)

Model output (vout
{
"cards": ['A', '3', 'K', '6'],
"number": [1, 3, 13, 6],
"formula": "(1+6)*3+13=24",
}

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

Verifier Output (vver

t

)

You failed this trial because your formula is incorrect.

▷ vin

t+1 = concat(vin

t , vout

t , vver
t )

Figure 12: An example of our prompt update for constructing vin
t . This example provides an optional vision
input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts
denote the state (st) specific info. The blue and red describe the output from the model and verifier, respectively.

t+1 using vin

t and vver

t , vout

'southeast',

'east',
'northwest'}, "forward()", "stop()"

'south',

'southwest',

'west',

– Relative direction:

"turn_direction(x)" where
'slightly left',
"stop()"

'slightly right'},

x∈{'left',

'right',
"forward()",

Reward design. An episode terminates when either the
navigation agent stops at the destination or the maximum
verification step of 2 is reached. The reward function is as
follows:

• r = 1: For generating a correct action at the current

coordinate

• Maximum straight road length: any positive integer

• r = −1: For generating wrong action at the current

coordinate

• r = −1: For exceeding maximum verification step

• r = −1.5: For failed detection of landmarks

The action space argument accommodates the rule variants
described in Section 4. For experiments shown in Fig-
ure 5, we use absolute direction action space during train-
ing and in-domain evaluation, while using the alternative
rule for out-of-domain evaluation. We implement a maxi-
mum straight road length to limit the number of movable
coordinates between turning points, preventing sequences
of repetitive "forward()" actions. We conduct visual dis-
tribution shift experiments (Section 5.2) via training the
model on New York City regions and evaluating the out-of-
domain performance on the worldwide navigation routes
from the benchmark released by Yang et al. (2024a).

15

SFT Memorizes, RL Generalizes

System Prompt (vin
0 )
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to
follow while observing your surrounding street views. You are also provided with
your observation and action history in text. your goal is to take the action based on
the current observation and instruction.
[Instruction]
1. First, turn left to face east.
2. Move forward until you reach the next intersection where Hotel 32One is on
your right behind.
3. Turn right to face north.
4. Move forward until you reach the next intersection where Dragon Gate Chinatown SF is on your right front.
5. Turn left to face east.
6. Move forward until the destination Café de la Presse is on your right.
[Current observation]
You observe a 2x2 grid of street view images with the following headings:
[front, right
back, left]
You need to identify if any of the landmarks in the instruction are visible in the street view grid.
[Action space]
- "forward()":
- "turn_direction(x)":
’south’, ’southwest’, ’west’, ’northwest’];
- "stop()": indicates the navigation is finished;
[Observations and actions sequence]
O_1: No landmarks nearby;
A_1: turn_direction(east)
O_2: No landmarks nearby;
A_2: forward()
O_3: No landmarks nearby;
A_3: forward()
O_4: You observe an image of 4 views; You observe an intersection
A_4:
[Output]
Your response should be a valid json file in the following format:
{
"current observation":
latest observation from the street view grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
}

indicates moving forward for 1 step;

indicates turn direction to the target heading, where x∈[’north’, ’northeast’, ’east’, ’southeast’,

Appending model and verifier outputs to obtain vin
t
1 , . . . , vout
t = [vout
vin

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

t

)

Model output (vout
{
"current observation": "Hotel 32One is on my right behind; I observe an intersection",
"current instruction": "Turn right to face north.",
"action": "turn_direction(north)",
}

Verifier Output (vver

t

)

Correct solution.

▷ Move to the next scene upon correct action

Figure 13: An example of our prompt update for constructing vin
t . This example provides an optional vision
input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts
denote the state (st) specific info. The blue and red describe the output from the model and verifier, respectively.

t+1 using vin

t and vver

t , vout

16

SFT Memorizes, RL Generalizes

indicates moving forward for 1 step;

System Prompt (vin
0 )
[Task Description]
You are an expert in navgation. You will receive a sequence of instructions to follow. You are also provided with
your observation and action histroy in text. Your goal is to first analyze the instruction and identify the next sen-
tence to be executed. Then, you need to provide the action to be taken based on the current observation and instruc-
tion.
[Instruction]
1. First, turn left to face east.
2. Move forward until you reach the next intersection where Hotel 32One is on your right behind.
3. Turn right to face north.
4. Move forward until you reach the next intersection where Dragon Gate Chinatown SF is on your right front.
5. Turn left to face east.
6. Move forward until the destination Café de la Presse is on your right.
[Action space]
- "forward()":
- "turn_direction(x)":
'south', 'southwest', 'west', 'northwest'];
- "stop()": indicates the navigation is finished;
[Observations and actions sequence]
O_1: No landmarks nearby;
A_1: turn_direction(east)
O_2: No landmarks nearby;
A_2: forward()
O_3: No landmarks nearby;
A_3: forward()
O_4: Hotel 32One is on your right behind; You observe an intersection
A_4:
[Output]
Your response should be a valid json file in the following format:
{
"current observation":
latest observation from the street view grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
}

indicates turn direction to the target heading, where x∈['north', 'northeast', 'east', 'southeast',

Appending model and verifier outputs to obtain vin
t
1 , . . . , vout
t = [vout
vin

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

t

)

Model output (vout
{
"current observation": "Hotel 32One is on my right behind; I observe an intersection",
"current instruction": "Turn right to face north.",
"action": "turn_direction(north)",
}

Verifier Output (vver

t

)

Correct solution.

▷ Move to the next scene upon correct action

Figure 14: An example of our prompt update for constructing vin
t . The brown parts marks the task and
related information, and the purple parts denote the state (st) specific info. The brown parts marks the task and related information, and
the purple parts denote the state (st) specific info. The blue and red describe the output from the model and verifier, respectively.

t+1 using vin

and vver

t , vout
t

17

SFT Memorizes, RL Generalizes

C. Experimental Setup

This section details the experimental setup used in Sec-
tion 5. We first describe our data collection setup for su-
pervised fine-tuning (Appendix C.1). Then, we present the
training pipeline (Appendix C.2). Finally, we describe our
evaluation metrics and the statistical tools used for gener-
ating plots (Appendix C.3).

C.1. Data

SFT data collection. As illustrated in Figures 11 to 14,
GeneralPoints and V-IRL environments naturally
align with prompt-response dialogue structures. We cre-
ate training samples by pairing each system prompt with its
corresponding expert response. All SFT experiments in the
main body use optimal single-turn prompt-response pairs,
without any verification or revision steps.

SFT on sub-optimal trajectories To examine how more
diverse SFT data affects the out-of-distribution perfor-
mance of SFT, we conduct an ablation study on GP-L us-
ing sub-optimal trajectories as training data. Unlike ex-
pert prompt-response pairs, these sub-optimal trajectories
include errors and verification messages in their prompts.
This format aligns with evaluation scenarios where multi-
ple verification iterations are allowed, similar to the data
being used for the downstream RL training. In Figure 15,
we observe that SFT still merely memorizes the training
data with degraded out-of-distribution performance. This
evidence suggests that memorization occurs due to the fun-
damental nature of SFT training rather than the SFT data.

ize the model with SFT, then separately scale up the com-
pute for SFT and RL (Schulman et al., 2017), starting from
this initialized model. For all experiments of SFT and RL
in the main body, we tune all components using a shared
learning rate per experiment. All training experiments are
conducted on an 8 H800 machine (80GB).

C.3. Evaluation Metric

Per-step accuracy. We report the per-step accuracy for
V-IRL-VL task in Figures 5 and 6. An individual step
is considered correct when the model’s chosen action
matches the expert trajectory at that position. Note that
intermediate verification steps are counted as independent
samples here.

Success rate. We report the success rate (%) of GP-L,
GP-VL, V-IRL-L and V-IRL-VL in Figures 5 and 6. In
the GeneralPoints task, success is defined as succeed-
ing at least once during the inference time verification. In
the V-IRL task, a sample is recorded as success when the
model takes correct action at each movable point on the
route.

Computation estimation. We estimate the FLOPs for
training X following the similar manner of
(Snell et al.,
2024; Hoffmann et al., 2023), where Xtrain = 6N Dtrain
and Xinf erence = 2N Dinf erence. Here, N represents the
model parameters and Dtrain represents the number of to-
kens during training. Suppose our SFT and RL experients
starts from a checkpoint trained on Dinit tokens, we can
estimate the training computation of SFT and RL via the
following equations:

XSF T = 6N (Dinit + DSF T )

XRL = 6N (Dinit + DRL) + 2N Dbuf f er

Note that the used on-policy RL algorithm PPO (Schulman
et al., 2017) contains iterative stages of replay buffer collec-
tion and optimization, hence requiring additional inference
computation. For simplicity, we approximate the term via:

Dbuf f er ≈

¯do
E ¯di
DRL
= λDRL

· DRL

Figure 15: SFT experiments on GP-L with suboptimal
trajectories. Similar to results in Figure 5, SFT overfits
the training data even we increase the trajectory diversity.

where E ∈ N denotes the number of auto-regressive gen-
eration processes, ¯di, ¯do denote average input tokens and
output tokens. We estimate the λ for GeneralPoints
and V-IRL as 6 and 5.1 respectively after calculation.

C.2. Training Pipeline

As illustrated in Section 5, we follow the training pipeline
by RL4VLM (Zhai et al., 2024a), where we first initial-

Line smoothing and error bar. All line plots in our pa-
per adopt Savitzky–Golay filter with polynomial order 3 as
smoothing function. We assume each evaluated data point

18

12345Computation (GFLOPs)1e9020406080100Success Rate (%)Out-of-distributionIn-distributionSFT Memorizes, RL Generalizes

Figure 16: Ablation studies on GeneralPoints-VL SFT. We ablate the learning rate and report the in-distribution
episode success rate (%) of all experiments. None of the experiments shows an increasing trend beyond 30% success rate.

follows a binomial distribution and approximate the stan-

dard error using
rate and N is the number of samples.

(cid:113) P (1−P )
N

, where P is the demical success

D. Additional Experimental Results

In this section, we provide additional experimental results
that are not covered in the main body.

D.1. Ablation Studies on GP-VL

As mentioned in Section 6, we observe an abnormal
phenomenon that SFT fails to achieve comparable in-
distribution performance with RL (see Figure 5 subplot row
1 column 3). To further explore this, we conduct ablation
studies over different hyperparameter choices.

SFT. We ablate the hyperparameter choices under the
same task setting of GP-VL in Section 5.1. For experi-
ments fine-tuning all parameters, we search learning rates
from {1×10−4, 1×10−4, 1×10−5, 1×10−6, 5×10−7, 1×
10−7}. Freezing the vision encoder, we search learning
rates {1 × 10−6, 1 × 10−7}. Freezing vision encoder and
adapter, we search learning rates {1 × 10−6, 5 × 10−7, 1 ×
10−7}. We provide the in-distribution success rate curve
in Figure 16.

RL. Finding suitable hyperparameters for RL experi-
ments requires minimal effort. We conduct a search over
learning rates 2 × 10−6, 1 × 10−6, with the in-distribution
success rate curves shown in Figure 17. All parameters are
tunable in our RL experiments.

D.2. More results on V-IRL-VL

Echoing per-step accuracy results in Figure 5, we report
the overall success rate of V-IRL-VL in Figure 18. Due to
the task’s complexity, both training methods achieve over-
all success rates no higher than 1%. For V-IRL, the overall

Figure 17: Ablation studies on GeneralPoints-VL
RL. Echoing Figure 16, we ablate the learning rate and rre-
port the in-distribution episode success rate (%) of the two
experiments. All components are tunable here.

success rate is a significantly more demanding metric since
it aggregates per-step errors. For example, a random policy
achieving 10% per-step accuracy would achieve achieve
only approximately 10−8% success rate on enough routes
averaging 10 steps in length.

D.3. Failure Cases

In this section, we present 2 failure cases in our experi-
ments as mentioned in Sections 5.4 and 6.

Without SFT, RL fails.
In Figure 9, we present the train-
ing dynamics of failed RL experiments without SFT initial-
ization. We additionally provide output examples of these
experiments in Figure 20, where the model tends to gener-
ate unstructured response and fail.

RL cannot save overfitted checkpoints. As shown
in Figure 19, RL cannot recover the out-of-distribution
performance when initialized from a extremely overfitted
checkpoint that has an initial per-step accuracy of less than
1%. We additionally provide an output example in Fig-
ure 19, where the model fails to adjust to the new rule.

19

2461e10051015202530All Components Tunable1e-75e-71e-61e-51e-42461e100510152025Freeze Vision Encoder1e-71e-62461e10051015202530Freeze Vision Encoder and Adapter1e-75e-71e-6Training Computation (GFLOPs)Success Rate (%)2.83.03.23.43.6Training Computation (GFLOPs)1e920253035404550Success Rate (%)1e-62e-6SFT Memorizes, RL Generalizes

Figure 18: Overall success rate (%) - GFLOPs for
V-IRL-VL under rule variants. Due to the nature of
the task requiring aggregating a trajectory of correct ac-
tions, neither training method achieves reasonable out-
of-distribution performance.

Figure 19: Out-of-distribution per-step accuracy
(%) - GFLOPs for V-IRL-VL under rule variants
with overfitted initial checkpoint. Evaluation metric
details can be found in Appendix C.3.

System Prompt (vin
0 )

[Task Description]
You are an expert 24 points card game player. You are observing these four cards in the image. Note that 'J', 'Q',
and 'K' count as '11', '12', and '13' respectively, and each card must be used once. Your goal is to output a formula
that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Input]
Cards: ['2', 'J', '3', 'K']

[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}

Appending model and verifier outputs to obtain vin
t
1 , . . . , vout
t = [vout
vin

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

t

)

Failed model output (vout
To solve this problem, we can use a brute force approach by generating all possible combinations of the cards and
checking if any of them can be used to form an equation that equals 24.
Here is a Python solution for this problem:
“‘python
def calculate_formula(cards):
# Define the mapping of cards to their corresponding numbers
. . .

Verifier Output (vver

t

)

You failed this trial because your formula is incorrect.

▷ vin

t+1 = concat(vin

t , vout

t , vver
t )

Figure 20: Example model outputs without SFT initialization. We record model responses using prompts similar to the
one shown in Figure 11. The results demonstrate that Llama-3.2-Vision-11B fails to follow instructions properly. We omit
the long response which tries to solve the puzzle via code but fails to finish within finite context length.

20

2345GFLOPs1e10102030405060V-IRL-VL2345GFLOPs1e100.000.250.500.751.001.251.50V-IRL-VL OODSuccess Rate (%)InitializationSFTRL0.51.01.52.02.53.03.5Training Computation (GFLOPs)1e90.00.51.01.5Per-step Accuracy (%)1e-72e-7SFT Memorizes, RL Generalizes

System Prompt (vin
0 )
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to fol-
low while observing your surrounding street views. You are also provided with
your observation and action history in text. Your goal is to first analyze the instruc-
tion and identify the next sentence to be executed. Then, you need to provide the
action to be taken based on the current observation and instruction.
[Instruction]
1. First, turn right to face northwest.
2. Move forward until you reach next intersection where Korean War Memorial is
on your left.
3. Turn left to face southwest.
4. Move forward until you reach next intersection where Korean War Memorial is on your left behind.
5. Turn right to face north.
6. Move forward until you reach next intersection.
7. Turn left to face east.
8. Move forward until you reach next intersection.
9. Turn left to face north.
10. Move forward until you reach next intersection.
11. Turn right to face east.
12. Move forward until you reach next intersection.
13. Turn left to face northeast.
14. Move forward until you reach next intersection where 9/11 Memorial & Museum is on your left.
15. Turn right to face northwest.
16. Move forward until you reach destination where The destination 9/11 Memorial & Museum is on your right
front.
[Action space]
"forward()": indicates moving forward one step
"turn_direction(x)": indicates adjust the ego agent direction towards x direction. x could be any following [’left’,
’right’, ’slightly left’, ’slightly right’]
"stop()": indicates the navigation is finished.
[Observations and actions sequence]
O_1: You observe an image of 4 views;
A_1:
[Output]
{
"current observation":
latest observation from the streetview grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
}

Appending model and verifier outputs to obtain vin
t
1 , . . . , vout
t = [vout
vin

t−1, vver

1 , vver

0 , vout

0 , vver

t−1]

t

)

Failed model output (vout
{
"current observation": "No landmarks nearby",
"current instruction": "Turn right to face northwest.",
"action": "turn_direction(northwest)",
}

▷ vin

t = concat (cid:0)vin

0 , [vout

k , vver

k ]t−1
k=0

(cid:1)

Verifier Output (vver

t

)

Incorrect action.

▷ Fail this sample.

Figure 21: Failed example of V-IRL transition due to overfitting. This phenomenon happens more frequently during
scaling up supervised fine-tuning.

21

