
# Findling Neural Variability in the Medial Prefrontal Cortex Contributes to Efficient Adaptive Behavior
 
Charles Findling, Margaux Romand-Monnier, Vasilisa Skvortsova, Etienne Koechlin
 
Published: 2025
 
---
 
### Abstract
 
Neural variability, i.e., random fluctuations in neural activity, is a ubiquitous and sizable brain feature that impacts behavior. Its functional role, however, remains unclear, and neural variability is commonly viewed as a nuisance factor degrading behavioral efficiency. Using functional magnetic resonance imaging in humans and computational modeling, we show here that neural variability provides a solution to the open issue regarding how the brain produces efficient adaptive behavior in uncertain and changing environments without facing computational complexity problems. We found that neural variability in the medial prefrontal cortex (mPFC) enables decision-making processes in the mPFC to produce near-optimal behavior in uncertain and ever-changing environments without involving complex computations known in such environments to rapidly become computationally intractable. The results thus suggest that, in the same way as genetic variability contributes to adaptive evolution, neural variability contributes to efficient adaptive behavior in real-life environments.
 
---
 
### Introduction
 
Variability, i.e., random fluctuations in signal processing, is a ubiquitous and sizable feature of biological systems ranging from genetic to neural variability. In the brain, neural variability has a direct impact on behavior. Neural variability is both substantial and preserved throughout evolution and may be presumed to play an important functional role. Its function(s), however, remain(s) unclear, and neural variability is more commonly viewed as a nuisance factor degrading behavioral efficiency.
 
We report here behavioral and neuroimaging findings suggesting that, in contrast, neural variability plays a major role in producing efficient adaptive behavior in real-life environments featuring uncertain and changing situations.
 
In such environments, learning and adjusting to new situations optimally requires discounting past information in proportion to the environment volatility, i.e., the frequency of situation changes. Rodent, monkey, and human adaptive behavior was found to be consistent with this optimal adaptive principle, which was further associated with the dorsomedial prefrontal cortex (dmPFC), including the dorsal anterior cingulate cortex (dACC), along with the noradrenergic system. The dmPFC has a well-documented role in guiding behavior based on internal beliefs about action-outcome contingencies. However, the neural mechanisms implementing this optimal adaptive principle remain poorly understood, notably because estimating the environment volatility involves complex inferential computations that rapidly become intractable.
 
To clarify this issue, we hypothesized that neural variability might induce a behavioral flexibility consistent with this optimal adaptive principle without relying on complex computations. Indeed, previous studies first show that exploratory choices in uncertain and changing environments reflect computational imprecisions corrupting the learning of action-outcome contingencies. Second, neural network models suggest that the variability in neuron spiking induces internal representations encoded in populations of neurons to undergo a stochastic variability obeying Weber's law, i.e., scaling with the magnitude of changes in encoded representations. When this Weber variability corrupts the formation of internal beliefs about action-outcome contingencies, computer simulations further show that the corrupted beliefs elicited nearly optimal adaptive behavior in uncertain and changing environments without relying on complex volatility inferences. These results thus lead to the intriguing hypothesis that neural variability alone might induce the dmPFC to elicit efficient adaptive behavior by merely encoding beliefs assuming stable environments but undergoing stochastic Weber variability.
 
We tested this hypothesis (named the Weber-variability model) by using human functional magnetic resonance imaging (fMRI) along with computational modeling to measure the aggregated impact of neural variability onto internal representations the brain encodes to guide behavioral choices.
 
---
 
### Results
 
We scanned 22 participants while they were performing a standard two-armed bandit task with binary outcomes (reward vs. no reward) within a varying-volatility environment. Participants had to choose in every trial one of the two visually presented arms by pressing one of two response buttons. One arm led to rewards more frequently (probability η = 85% vs. 1 - η = 15%) but these reward contingencies reversed unpredictably with, unbeknownst to participants, a probability (named volatility) varying episodically and pseudo-randomly along experimental sessions (volatility levels: 5%, 7%, and 10%). Participants chose the current, more frequently rewarded arm in 72% of trials, well above chance level (= 50%). Participants were also sensitive to volatility: participants switched their responses after no-rewarded trials more often in high than low volatility trials (+5% of trials), indicating that consistent with the optimal adaptive principle, they discounted past information more in high than low volatility episodes.
 
#### Modeling Adaptive Behavior
 
**Optimal Models:**
In this task, the optimal adaptive agent learns reward probabilities and forms state beliefs regarding how reward probabilities map onto bandits' arms across successive trials. If the task contingencies were stationary (no reversals), these state beliefs would derive from merely registering online the outcomes associated with chosen arms over trials, what we name first-order inferences.
 
Because of reversals, the optimal agent needs to further discount the weight of its prior state beliefs in forming its subsequent state beliefs according to the probability of reversals, i.e., the environment volatility. The more volatile the environment is, the less the agent should rely on past information. The optimal agent infers the environment volatility from the history of action outcomes.
 
**The Weber-Variability Model:**
The first-order inference model assumes stationary action-outcome contingencies and evidently leads to poor adaptive performances in changing environments. The Weber-variability model remedies this limitation by assuming that belief updating in the first-order inference model undergoes computational imprecisions stemming from neural variability. In accordance with Weber's law, these imprecisions presumably scale with the magnitude of belief updating and consequently increase whenever reversals occur.
 
**Reinforcement Learning Models:**
We also considered reinforcement learning (RL) processes as potential alternative accounts of participants' adaptive performances. We investigated the standard Rescorla-Wagner and Pearce-Hall RL processes comprising constant and adaptive learning rates as free parameters, respectively. We found that among all these RL models, the noisy Rescorla-Wagner RL best fit participants' performances systematically.
 
#### The Weber-Variability Model Best Accounts for Human Performances
 
To capture potential noise in participants' action selection, all the models described above further included a softmax decision policy. Model fits were compared using Bayesian model comparisons with uniform priors (BMC), which optimally balance model complexity and adequacy to data and prevent overfitting issues.
 
BMC revealed that compared to the second-/third-order volatility inference and noisy Rescorla-Wagner RL model, the Weber-variability model decisively best fitted participants' choices. Its best-fitting free parameters confirmed that the Weber component dominantly contributed to Weber-variability in belief updating.
 
#### dmPFC Computes Choices from Beliefs Undergoing Weber Variability
 
We next examined the hypothesis that the dmPFC guides adaptive behavior and computes choices from corrupted beliefs, i.e., state beliefs from the best-fitting Weber-variability model. We entered fMRI activity in full variance regression analyses. According to previous studies, the fMRI signature of choice computations is that, besides increasing with reaction times (RTs), activations should exhibit two concomitant effects at choice time: (1) an effect reflecting the neural demand in reaching actual choices, i.e., activations decrease when the decision variable increasingly favors the chosen relative to the unchosen option; (2) an effect reflecting the neural demand in encoding the decision variable irrespective of chosen options.
 
The whole-brain analysis revealed that only activations in the dmPFC extending from the dACC to the pre-Supplementary Motor Area (pre-SMA) exhibited both choice computation effects. These dmPFC activations increased with RTs and independently exhibited both the predicted linear and quadratic effects, confirming previous evidence that the dmPFC plays a central role in computing choices from state beliefs.
 
#### Weber Variability Stems from Neural Variability Corrupting Belief Updating
 
We next tested whether Weber-variability arises from neural variability corrupting belief updating that unfolds between two successive trials from action outcome to next choice onsets. At choice time, belief updating is completed, and dmPFC activations reflected choice computations from corrupted beliefs. As Weber variability increases belief entropy, which augments the neural demand in computing choices, we reasoned that at choice time, dmPFC activations should overall increase with Weber variability. At outcome time, in contrast, belief updating starts, and dmPFC activations reflect the neural demand in belief updating.
 
The whole-brain regression analysis confirmed that at choice time, dmPFC activations increased with Weber variability. As predicted, activations at outcome time within this region were weakly associated with Weber variability.
 
#### Weber-Variability Explains dmPFC Activity Correlating with Volatility Estimates
 
According to the Weber-variability model, no inferences about the environment volatility are required to produce efficient adaptive behavior. Previous studies report that following action outcomes, dmPFC activations correlate with the volatility estimates from volatility inference models. We then examined whether the Weber-variability model might explain such volatility-related activations.
 
Replicating previous results, a whole-brain regression analysis confirmed that activations in the dmPFC correlated at outcome time with volatility estimates from the third-order volatility inference model. BMC then revealed that in each region, Weber variability accounted decisively better than volatility estimates for dmPFC activity from outcome to choice time.
 
---
 
### Discussion
 
The results confirmed that in uncertain and changing environments, the dACC and pre-SMA guide behavioral choices by encoding state beliefs deriving from first-order inferences about external contingencies. The encoded beliefs were found to further undergo a stochastic variability consistent with Weber's law and associated with the trial-by-trial neural variability in the dACC and pre-SMA corrupting belief-updating processes across successive trials. This neural variability also accounted for dmPFC activations previously reported to correlate with volatility estimates from volatility inference models.
 
As previously observed, we further found that human adaptive performances in uncertain and changing environments are consistent with optimal adaptive behavior. Critically, our results reveal that these efficient performances derive from the stochastic neural fluctuations corrupting the beliefs about action-outcome contingencies that the pre-SMA and dACC update to guide behavior rather than from additional complex neural computations/mechanisms dedicated to volatility estimates as previously proposed.
 
Our findings thus support the most parsimonious neural account of efficient adaptive behavior in uncertain and changing environments with no need to assume additional mechanisms and complex computations.
 
fMRI provides little indication about the neuronal origin of this corrupting variability. A likely hypothesis is that this variability originates in noisy neuron spiking activity resulting from noisy synaptic transmission or spike generation. In neural network models, this noisy activity elicits a stochastic Weber variability bearing upon information coding in populations of neurons.
 
Our results do not imply the neuronal sources of corrupting fluctuations to be confined to the dACC and pre-SMA. The stochastic corrupting fluctuations we observed in these regions are likely to reflect the cumulated effects of neural variability along the brain network, including notably the ventromedial prefrontal cortex known to be involved in forming internal beliefs from action outcomes.
 
Conceptually, our results suggest that the brain produces efficient adaptive behavior by rapidly forming simplified, locally accurate but globally inaccurate stationary world models, while neural variability enables rapid disengagement from an obsolescent local world model to form new ones to guide behavior. We found that the dmPFC guides behavior by encoding beliefs assuming stable action-outcome contingencies. The assumption is accurate at a short time scale and advantageously maximizes the speed of learning ongoing environment contingencies. However, the assumption is evidently inaccurate at longer time scales and considerably decreases the speed of adapting to contingency changes. The stochastic neural fluctuations we observed as corrupting these beliefs encoded in the dmPFC to guide behavior precisely suppress this downside.
 
In particular, when action outcomes are repeatedly inconsistent with the beliefs encoded in the dmPFC and prompt substantial belief updates, the observed stochastic neural fluctuations make such beliefs increasingly variable to the point that they essentially induce random switching across distinct courses of action. This may occur before beliefs possibly stabilize in favor of one course of action and, following Weber's law, undergo less stochastic neural fluctuations.
 
The present study has some limitations. fMRI allows measuring the aggregated impact of neural variability on information processing. Accordingly, fMRI reveals here that fluctuations in dmPFC activations induce the Weber variability corrupting the representations encoded in the dmPFC to guide behavior. As noted above, however, fMRI precludes the possibility of identifying the precise neural origins of such fluctuations and, consequently, Weber variability.
 
To conclude, our findings point to a general neural adaptive principle based on forming world models presuming stable environments but undergoing a variability scaling with internal changes inconsistent with this stability premise. This principle has two key evolutionary advantages: (1) computational frugality, as forming such stationary world models relies on elementary neural computations; (2) uncertainty robustness, as its adaptive efficiency relies on no assumptions regarding the true temporal structure of the environment.
 
---
 
### Methods
 
#### Participants
22 right-handed volunteers participated in the present study (12 females, mean age: 24.6 years, age range: 20–30 years). Participants had neither a history of neurological and psychiatric diseases nor current psychiatric medication and had normal or corrected-to-normal vision. Every participant provided written informed consent, and the study was approved by the French National Ethics Committee.
 
#### Behavioral Protocol
Participants were tested in two experimental sessions administered on two distinct days. Each session included three scanning runs. In every run, participants carried out a two-armed bandit task comprising 180 trials. One arm led to rewards more frequently (85% vs. 15%), but these reward contingencies reversed unpredictably with a probability varying episodically and pseudo-randomly along runs (volatility levels: 5%, 7%, and 10%).
 
#### fMRI Data Acquisition
A Siemens Verio 3T scanner and a 32-channel head coil were used to acquire both high-resolution T1-weighted anatomical MRI and a T2*-weighted multiband-echo planar imaging (mb-EPI). Image pre-processing included co-registration of anatomical T1 images with mean EPI, segmentation, and normalization to a standard T1 template.
 
#### Computational Models
**First-Order Inference Model:** Assumes external contingencies to remain stable over time, i.e., volatility is zero, and the environment remains in the same latent state.
 
**Second-Order Volatility Inference Model:** Assumes that environment latent states may change over time with a volatility/probability that remains constant across trials.
 
**Third-Order Volatility Inference Model:** Assumes that volatility varies across trials as a bounded, Gaussian random walk with an unknown variance.
 
**Weber-Variability Model:** Based on the first-order inference model but postulates that neural variability induces computational imprecisions increasing the entropy/variance in belief updating.
 
**Reinforcement Learning Models:** Standard Rescorla-Wagner and Pearce-Hall RL processes were considered, with the noisy Rescorla-Wagner RL best fitting participants' performances.
 
#### Decision Variable and Policy
The decision variable driving choices in every trial is the difference between reward expectations associated with each option. For RL models, the decision variable is the difference in action values. For inference models, reward expectations are computed by marginalizing over first-order beliefs.
 
#### Bayesian Model Comparisons
Model fits to human data were compared based on exact Model Posterior Probabilities (MPPs), balancing model complexity and adequacy to data.
 
#### fMRI Data Analyses
All fMRI data analyses were conducted using the SPM12 software. Statistical parametric maps of local brain activations were computed using the standard general linear model (GLM).
