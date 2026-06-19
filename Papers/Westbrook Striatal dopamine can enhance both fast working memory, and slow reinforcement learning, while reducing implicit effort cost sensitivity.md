# Westbrook Striatal dopamine can enhance both fast working memory and slow reinforcement learning while reducing implicit effort cost sensitivity


Andrew Westbrook, Ruben van den Bosch, Lieke Hofmans, Danae Papadopetraki, Jessica I. Määtä, Anne G. E. Collins, Michael J. Frank, Roshan Cools
Published online: 09 July 2025
 
---
 
### Abstract
 
Associations can be learned incrementally, via reinforcement learning (RL), or stored instantly in working memory (WM). While WM is fast, it is also capacity-limited and effortful. Striatal dopamine may promote WM by facilitating WM updating and effort exertion, and also RL by boosting plasticity. Yet, prior studies have failed to distinguish between the effects of dopamine manipulations on RL versus WM.
 
N = 100 participants completed a paradigm isolating these systems in a double-blind study measuring dopamine synthesis with [18F]-FDOPA PET imaging and manipulating dopamine with methylphenidate and sulpiride. We find that learning is enhanced among high synthesis capacity individuals and by methylphenidate, but impaired by sulpiride. Methylphenidate also blunts implicit effort cost learning. Computational modeling reveals that individuals with higher dopamine synthesis capacity rely more on WM, while methylphenidate boosts their RL rates. The D2 receptor antagonist sulpiride reduces accuracy due to diminished WM involvement and faster WM decay.
 
We conclude that dopamine enhances both slow RL and fast WM by promoting plasticity and reducing implicit effort sensitivity.
 
---
 
### Results
 
#### Both reinforcement learning and working memory contribute to performance
 
Conjoint contributions of WM and RL are implied by the shape of learning curves by stimulus iteration. Accuracy is higher with more iterations of each stimulus and for smaller set sizes. An interaction between these factors implies that effective learning rates are larger in smaller set size blocks.
 
#### Striatal dopamine variously enhances performance
 
Striatal dopamine signaling improves performance in multiple ways. Higher dopamine synthesis capacity and methylphenidate both increase accuracy, while sulpiride decreases accuracy. To understand why, we fit an RL model to behavior, examining how people learn to select the correct action for each stimulus in each block. The algorithm combines a WM component featuring instantaneous learning, capacity limits, and susceptibility to decay, and a capacity-unlimited RL component with incremental learning rates.
 
Fitted model parameters imply two ways in which striatal dopamine signaling boosts performance. First, it boosts performance by increasing the likelihood that people rely on WM, which facilitates fast and flexible acquisition of new associations. Second, striatal dopamine signaling also appears to increase the learning rate in the RL system, which describes the rate at which incrementally acquired stimulus-response associations can contribute to action selection.
 
A hierarchical regression of the parameter ρ (WM reliance) on dorsal caudate dopamine synthesis capacity and drug reveals a positive effect of dopamine synthesis capacity on placebo, no effect of methylphenidate vs placebo, and a negative effect of sulpiride vs placebo on WM reliance.
 
The hypothesis that participants with higher dopamine synthesis capacity rely more on WM is further supported by evidence that they tend to perform better early in a block when items are novel and WM affords better performance. Even when controlling for the effect of striatal dopamine signaling on WM, there is evidence that dopamine boosts incremental RL learning rates.
 
#### Methylphenidate blunts implicit effort cost sensitivity
 
Following stimulus-response learning, participants were presented with a surprise test phase in which they were asked to select which of two stimuli received greater rewards. Stimuli were selected pseudorandomly from those previously encountered across all training phase blocks. The intent of this test phase was to study RL-based value representations after learning.
 
We find that participants faithfully track reward statistics, correctly picking the stimulus which was rewarded at a higher rate, and increasingly so as the difference in actual rewards increased. Also, participants assign a lower reward value to stimuli that had been encountered in larger set-size blocks, reflecting implicit sensitivity to effort costs. Importantly, methylphenidate blunts this implicit effort-discounting effect.
 
---
 
### Discussion
 
Striatal dopamine promotes corticostriatal plasticity and thereby facilitates RL. In this study, we employ a task that dissociates the relative contributions of RL and WM systems to learning and examine how dopamine influences each. We find that striatal dopamine signaling—modulated by either dopamine synthesis capacity or dopamine drugs—can promote learning, with distinct effects on both RL and WM.
 
Specifically, higher dopamine synthesis capacity in the dorsal caudate nucleus predicts greater reliance on WM, while antagonism of dopamine receptors with sulpiride reduces performance by reducing reliance on WM. By blocking dopamine reuptake and thereby amplifying dopamine signaling, methylphenidate also boosts performance in interactions with striatal dopamine synthesis capacity. Specifically, methylphenidate increases RL learning rates for people who synthesize dopamine at a higher rate.
 
We speculate that reliance on WM correlates with striatal dopamine synthesis capacity because the latter may shape a trait policy to rely on WM in general, in novel contexts. This could help explain why dopamine synthesis capacity may correlate with individual differences in WM capacity.
 
The present findings constitute an important complement to prior work by showing that dopamine may shape value learning about effort costs themselves. Specifically, we find that people treat rewards earned in the context of higher WM demands as subjectively less rewarding—an effect which we interpret as implicit discounting of rewards by increasing cognitive effort costs. Critically, methylphenidate blunts this implicit effort-discounting effect.
 
Many forms of psychopathology have been associated with aberrant rates of RL. However, our results confirm prior work showing that it is crucial to control for the degree to which WM contributes to the learning process when trying to estimate learning rates for an RL system.
 
Controlling for the contributions of WM to the learning process, we also find evidence that striatal dopamine signaling accelerates RL. Specifically, methylphenidate boosts RL rates the most for people who synthesize dopamine faster in the dorsal caudate nucleus.
 
Although sulpiride clearly undermines performance, the mechanisms are somewhat less resolved. Sulpiride causes performance to decline both early and late in blocks, suggesting it may impact both RL and WM processes. Model parameters suggest that sulpiride diminishes performance because it reduces the degree to which WM contributes to behavior.
 
Our results support multiple, complementary mechanisms by which striatal dopamine influences task learning. Namely, we find that individual differences in dopamine synthesis capacity correlate positively with baseline propensity to rely on WM. We also find that sulpiride reduces reliance on WM, perhaps by undermining the stability of WM contents over time. These effects indicate that striatal dopamine can increase reliance on fast and flexible WM for learning. Yet, even after accounting for the effects of dopamine on WM, we find that striatal dopamine can accelerate slow learning processes as well.
 
---
 
### Methods
 
#### Participants
100 healthy, young adult participants (ages 18–43, 50 men) were recruited to participate in a within-subject, double-blind, placebo-controlled study. Participants were screened to ensure that they were right-handed, Dutch-native speakers, healthy, neurologically normal, and without a history of mental illness or substance abuse.
 
#### General procedure and tasks
Participants completed five visits as part of a broader study of the effects of dopamine on cognitive control: one screening session, three pharmaco-imaging sessions with multiple tasks performed in and out of the fMRI scanner after being administered placebo, sulpiride, or methylphenidate, and a final PET session for measuring dopamine synthesis capacity.
 
#### Reinforcement learning working memory task
The RLWM task was presented using Psychtoolbox-3 for MATLAB. Participants completed two task phases: a training phase and a test phase. In the training phase, participants were presented with stimuli in blocks of varying set sizes (between 2 and 5 stimuli in each block). Stimuli were presented one-at-a-time and participants responded with one of three button presses.
 
If participants responded correctly on a given trial, they were always given reward feedback (+1 or +2 points probabilistically), and if they were incorrect, they received zero points. At the end of the training period, participants completed a surprise test phase in which pairs of stimuli were drawn from across all blocks and participants were tasked with selecting which of each pair was rewarded at a higher rate.
 
#### Computational modeling of behavior
We adapted a learning algorithm involving both WM and RL modules to support stimulus-response learning and fit it to behavior. The model contains six free parameters and was fit using the mfit toolbox in MATLAB.
 
#### PET scanning
To measure dopamine synthesis capacity, participants completed a PET scanning session using a Siemens mCT PET-CT scanner. Presynaptic dopamine synthesis capacity was calculated as the F-DOPA influx rate per voxel using the Gjedde-Patlak linear graphical analysis method.
 
#### Statistics and reproducibility
Our sample size was determined based on the effect size of a previous pharmacological-behavioral study. All experiments were double-blinded and participants completed all drug sessions in a randomized order, using a crossover design.
