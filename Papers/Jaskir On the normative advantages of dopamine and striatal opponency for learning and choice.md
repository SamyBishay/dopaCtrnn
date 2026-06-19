# Jaskir On the normative advantages of dopamine and striatal opponency for learning and choice

 
Alana Jaskir, Michael J Frank
 
Department of Cognitive, Linguistic and Psychological Sciences, Carney Institute for Brain Science, Brown University, Providence, United States
 
Published: 22 March 2023
 
---
 
### Abstract
 
The basal ganglia (BG) contribute to reinforcement learning (RL) and decision-making, but unlike artificial RL agents, it relies on complex circuitry and dynamic dopamine modulation of opponent striatal pathways to do so. We develop the OpAL* model to assess the normative advantages of this circuitry. In OpAL*, learning induces opponent pathways to differentially emphasize the history of positive or negative outcomes for each action. Dynamic dopamine (DA) modulation then amplifies the pathway most tuned for the task environment. This efficient coding mechanism avoids a vexing explore-exploit tradeoff that plagues traditional RL models in sparse reward environments. OpAL* exhibits robust advantages over alternative models, particularly in environments with sparse reward and large action spaces. These advantages depend on opponent and nonlinear Hebbian plasticity mechanisms previously thought to be pathological. Finally, OpAL* captures risky choice patterns arising from DA and environmental manipulations across species, suggesting that they result from a normative biological mechanism.
 
---
 
### Introduction
 
Everyday choices involve integrating and comparing the subjective values of alternative actions. Moreover, the degree to which one prioritizes the benefits or costs in forming subjective preferences may vary between and even within individuals. In ecological settings, there are often multiple available actions, and rewards are sparse. In machine learning, this combination is particularly vexing for reinforcement learning (RL) agents due to a difficult exploration/exploitation tradeoff. We set out to study how the architecture of biological RL might additionally circumvent this problem.
 
We find that biological properties within the basal ganglia (BG) and dopamine (DA) system—specifically, the presence of opponent striatal pathways, nonlinear Hebbian plasticity, and dynamic changes in dopamine as a function of reward history—confer decision-making advantages relative to canonical RL models lacking these properties. This analysis provides a new lens into various findings regarding how learning and decision-making is altered across species as a function of manipulations of (or individual differences within) the BG and DA systems.
 
We focus on bandit learning tasks, where an agent learns to identify and reliably select the option which yields the highest rate of probabilistic reward. We consider how biological properties within the BG allow an agent to effectively explore early and subsequently better exploit. This entails learning separate ‘actors' that magnify the relative benefits of alternative options in highly rewarding environments or the relative costs in sparsely rewarding environments and dynamically shifting the contribution of these actors to govern action selection, depending on which is more specialized for the context.
 
In neural network models of such circuitry, the cortex ‘proposes' candidate actions available for consideration, and the BG facilitates those that are most likely to maximize reward and minimize cost. These models are based on the BG architecture in which striatal medium spiny neurons (MSNs) are subdivided into two major populations that respond in opponent ways to DA. Phasic DA signals convey reward prediction errors, amplifying both activity and synaptic learning in D1 neurons, thereby promoting action selection based on reward. Conversely, when DA levels drop, activity is amplified in D2 neurons, promoting learning and choice that minimizes disappointment.
 
Empirically, the BG and DA have been strongly implicated in motivated action selection and RL across species. Striatal DA manipulations influence RL, motivational vigor, cost-benefit decisions about physical effort, and risky decision-making. As striatal DA levels rise, humans and animals are more likely to select riskier options that offer greater potential payout than those with certain but smaller rewards.
 
However, for the large part, this literature has focused on the findings that DA has opponent effects on D1 and D2 populations and behavioral patterns, and not what the computational advantage of this scheme might be. The Opponent Actor Learning (OpAL) model summarizes the core functionality of the BG neural network models in algorithmic form, capturing a wide variety of findings of DA and D1 vs. D2 manipulations across species. Two distinguishing features of OpAL are that it relies on opponent D1/D2 actors that separately learn benefits and costs of actions rather than a single expected reward value for each action and learning in such populations is acquired through nonlinear dynamics, mimicking three-factor Hebbian plasticity rules.
 
But why would the brain develop this nonlinear opponent mechanism for action selection and learning, and how could healthy DA levels be adapted to capitalize on it? Standard RL models typically perform worse at selecting the optimal action in ‘lean environments' with sparse rewards than they do in ‘rich environments' with plentiful rewards. This asymmetry results from a difference in exploration/exploitation tradeoffs across such environments.
 
We propose a new model, OpAL*, which dynamically adapts its dopaminergic state online as a function of learned reward history. OpAL* dynamically modulates its dopaminergic states in proportion to its estimates of ‘environmental richness,' leading to high striatal DA motivational states in rich environments and lower DA states in lean environments with sparse rewards. To do so, it relies on a ‘meta-critic' that evaluates the richness/sparseness of the environment as a whole.
 
We demonstrate that the specialization of D1 and D2 pathways in OpAL* for discriminating between low rewarding and high rewarding options, rather than estimating veridical reward statistics, allows OpAL* to better equate performance in rich and lean environments. This dynamic modulation amplifies the D1 or D2 actor most well suited to discriminate amongst benefits or costs of choice options for the given environment, akin to an ‘efficient coding' strategy.
 
We compared the performance of OpAL* to alternative BG models and to several alternative models typically used in machine learning. We find that OpAL*, across a wide range of parameter settings, exhibits robust advantages over these alternatives across a range of environments with varying reward rates and complexity levels. This advantage depends on opponency, nonlinearity, and adaptive DA modulation and is most prominent in lean environments with large action spaces.
 
Finally, we apply OpAL* to capture a range of empirical data across species, including how risk preference changes as a function of D2 MSN activity and manipulations that are not explainable by monolithic RL systems even when made sensitive to risk. OpAL* can reproduce patterns in which dopaminergic drug administration selectively increases risky choices for gambles with potential gains. Moreover, OpAL* also accounts for recently described economic choice patterns as a function of environmental richness.
 
---
 
### Results
 
#### Robust advantages of adaptively modulated dopamine states
 
The main claim of this article is that endogenous changes in dopamine levels can leverage specialization afforded by opponent pathways under Hebbian plasticity, and accordingly optimize performance when environmental statistics are unknown. We characterize the robustness of OpAL* advantages across a large range of parameter settings relative to variants omitting DA modulation (OpAL+) or the Hebbian term (No Hebb).
 
We compared OpAL* to two control models: OpAL+ model equally weights benefits and costs throughout learning, and the No Hebb model reinstates the dynamic dopaminergic modulation but omits the Hebbian term in the three-factor learning rule. Improvement of OpAL* relative to the No Hebb model would therefore suggest an advantage of OpAL* over standard actor-critic models.
 
OpAL* outperformed its OpAL+ and No Hebb control models across all time horizons. OpAL* robustly outperforms comparison models in both environment types. The advantages of OpAL* are most evident in the lean environment, particularly relative to the No Hebb model.
 
#### OpAL* advantages in sparse reward environments grow with complexity of action space
 
We explored the advantages of dopamine modulation and Hebbian plasticity in progressively more complex environments by increasing the number of available choice alternatives. OpAL* outperformed the OpAL+ model across all time horizons and complexity levels. OpAL* also outperformed the non-Hebbian version, except for the lowest complexity lean environments after 1000 trials.
 
OpAL* shows better performance across a range of parameters than control models. The benefits of OpAL* are most evident in the lean environment. OpAL*‘s advantages grow monotonically with complexity, roughly doubling from low- to high-complexity levels in lean environments.
 
#### OpAL* robustly and optimally outperforms benchmark models
 
We demonstrated that the combination of OpAL*‘s components confers adaptive flexibility especially when an agent does not know the statistics of a novel environment. We evaluated whether OpAL* exhibits similar advantages compared to standard alternatives in the reinforcement literature, including Q-learning and Upper Confidence Bound (UCB).
 
OpAL* outperformed both UCB and Q-learning across a large range of parameter settings. OpAL* outperforms a standard Q-learning and UCB in the computationally easiest scenario and, more prominently, in the computationally most difficult scenario.
 
#### OpAL* adaptively modulates risk-taking
 
OpAL* dynamically updated its probability of gambling and improved performance in comparison to the balanced OpAL+. DA modulation showed a larger benefit in the lean environment relative to the rich environment. By lowering its dopamine levels, OpAL* can leverage the specialization of N weights, which are more sample efficient in lean environments relative to standard RL.
 
---
 
### Mechanism
 
#### Q learners show poor convergence and reductions in action gap with sparse reward
 
A key objective of a Q-learner is that Q values converge to the expected reward for each option. However, before the algorithm converges, the policy selects actions that will necessarily be influenced by misestimation errors. Q value convergence is impeded when the agent has to select between multiple options via a stochastic choice policy. This issue weakens the 'action gap': the gap between the expected reward value for the optimal action and that of the next best option, which in turn impedes performance.
 
#### Opponency and Hebbian nonlinearity allows OpAL* to optimize action gaps
 
Opponency and nonlinearity allow OpAL* to overcome differences in rich and lean environments. The nonlinearities in opponent actors imply that the two actors differentially specialize in discriminating between high and low reward probability options. Increasing DA amplifies the G actor's contributions to choice, which increases the action gap for high-probability options. Conversely, lowering DA amplifies the N actor's contributions, which increases the action gap for low-probability options.
 
OpAL* more quickly allows the N weights to discriminate between low-probability options. The Hebbian nonlinearity ensures that negative experiences induce disproportional distortions in N weights, more rapidly increasing the action gap between optimal and suboptimal options with less stochastic sampling required.
 
#### Advantages in lean environment are not seen in other opponent BG models lacking Hebbian nonlinearity
 
Opponency alone is not sufficient to remediate this divergence in rich and lean performance. The Hebbian term produces nonlinear dynamics in the two actors such that they are not redundant and instead specialize in discriminating between different reward probability ranges.
 
---
 
### OpAL* captures alterations in learning and choice preference across species
 
#### OpAL* accounts for counterintuitive human choice preferences for loss-avoiding options over those that produce net gains
 
OpAL* predicts that the relative value of other options can influence how an action is learned, which may produce counterintuitive behavior when the overall reward richness of a context changes. OpAL* captured counterintuitive and context-dependent human choice preferences.
 
#### Striatal D2 MSN activity and reward history alter risky choice in rodents
 
OpAL* can capture the impact of D2-receptor activity and manipulation, including outcome-dependent risk-avoidance paired with increase of D2 activity following a loss. Optogenetic stimulation of D2-expressing neurons induced decrease in risky choice in risk-seeking rodents in line with OpAL* predictions.
 
#### DA drug effects on risky decision-making and individual differences therein
 
OpAL* captured the selective effects of L-DOPA on gambling in gain trials. The model also accounted for individual differences of risk due to effective L-DOPA dosages.
 
#### Risky decisions are sensitive to environmental richness
 
OpAL* predicts increased gambling on common trials in the Rich block relative to the Lean block. This result reflects adaptively modulated DA levels in the Rich environment, which emphasized the benefits of the gamble during decision-making.
 
---
 
### Discussion
 
Our simulations provide a normative account for opponency within the BG and its modulation by DA. Nonlinear Hebbian mechanisms give rise to convexity in the learned D1 and D2 actor weights at different ends of the reward spectrum, which can be differentially leveraged to adapt decision-making. OpAL* alters its dopaminergic state as a function of environmental richness, so as to best discern between the costs or benefits of available options.
 
OpAL* robustly outperforms traditional RL and alternative BG models across environment types when sampling across a wide range of plausible parameters. These advantages grow monotonically with the complexity of the environment. The unity of all three key features of OpAL* (opponency, three-factor Hebbian nonlinearity, and dynamic DA modulation) offered particularly unique advantages in sparse reward environments.
 
This article intersects with theoretical and empirical work showing that changes in dopaminergic states locally within striatum reflect reward expectations and impact motivation and vigor. OpAL* can capture both shifts in vigor and cost-benefit choice as seen empirically with drug manipulations across species.
 
Notably, opponency alone is not sufficient to remediate divergence in rich and lean performance. The Hebbian term produces nonlinear dynamics in the two actors such that they are not redundant and instead specialize in discriminating between different reward probability ranges.
 
These findings contrast with Q-learning agents and with other theoretical models of striatal opponency which omit the Hebbian term but leverage alternate nonlinearities. OpAL* shows substantially improved performance in lean environments due to its opponent and nonlinear properties, especially when DA is modulated dynamically.
 
---
 
### Materials and methods
 
#### Parameter grid search
 
For OpAL* variants, we ran a grid sweep over a parameter space with various learning rates and softmax temperatures. Models were equated for computational complexity, with modulation hyperparameters of dynamic DA models held constant, and were compared using the same random seeds to best equate performance.
 
#### Upper Confidence Bound
 
To implement UCB, we used the sample mean of receiving reward for each action. The agent then greedily selected the action with the largest mean combined with an exploration factor.
 
#### Möller and Bogacz 2019 model
 
The Möller and Bogacz model offers another computational account of how benefits and costs may be encoded in the D1/D2 striatal subpopulations. This model defines benefits and costs as the absolute magnitude of positive and negative outcome for each action. Both OpAL and Möller and Bogacz's model have nonlinearities in the learning rule.
 
---
 
### Code
 
Code repository available at: https://github.com/amjaskir/opal-star
 
---
 
### Acknowledgements
 
AJ was partly supported by NIMH training grant T32MH115895. The project was also supported by NIMH R01 MH084840-08A1 and NIMH P50 MH119467-01. Computing hardware was supported by NIH Office of the Director grant S10OD025181.
 
---
 
### Author contributions
 
Alana Jaskir: Conceptualization, Software, Formal analysis, Validation, Investigation, Visualization, Writing - original draft, Writing - review and editing
Michael J Frank: Conceptualization, Formal analysis, Supervision, Funding acquisition, Validation, Project administration, Writing - review and editing
