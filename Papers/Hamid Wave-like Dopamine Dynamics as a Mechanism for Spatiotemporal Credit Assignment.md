
# Hamid Wave-like Dopamine Dynamics as a Mechanism for Spatiotemporal Credit Assignment
 
Arif A. Hamid, Michael J. Frank, Christopher I. Moore
 
Published: May 13, 2021
 
---
 
Summary
 
Significant evidence supports the view that dopamine shapes learning by encoding reward prediction errors. However, it is unknown 
whether striatal targets receive tailored dopamine dynamics based on regional functional specialization. Here, we report 
wave-like spatiotemporal activity patterns in dopamine axons and release across the dorsal striatum. These waves switch between 
activational motifs and organize dopamine transients into localized clusters within functionally related striatal subregions. 
Notably, wave trajectories were tailored to task demands, propagating from dorsomedial to dorsolateral striatum when rewards 
are contingent on animal behavior and in the opponent direction when rewards are independent of behavioral responses. We 
propose a computational architecture in which striatal dopamine waves are sculpted by inference about agency and provide a 
mechanism to direct credit assignment to specialized striatal subregions. Supporting model predictions, dorsomedial dopamine 
activity during reward-pursuit signaled the extent of instrumental control and interacted with reward waves to predict future 
behavioral adjustments.
 
---
 
Introduction
 
Dopamine (DA) supports reward learning and motivated behaviors, but precisely what information it encodes and how it arrives at 
postsynaptic targets remain unclear. According to the reward prediction error (RPE) hypothesis, transients in DA signaling reflect 
devations from reward expectation that drive reinforcement learning (RL). This formulation generally treats DA as a global 
(spatiotemporally uniform) signal, a view based on two key findings. First, DA axon projections to the forebrain are extensively 
divergent, providing an architecture for broadcast-like communication. Second, midbrain DA neuron spikes are highly synchronized, 
putatively implementing a code for RPEs. These observations form the basis for an influential view of what DA communicates and 
how it is delivered: scalar RPEs that are uniformly broadcast to all recipient subregions.
 
It remains debated, however, whether DA signals convey such scalar, uniform decision variables. In the midbrain, DA neurons are 
reported to encode multiple behavior- and stimulus-specific features, or distributions of reward outcomes in an RPE framework. 
Moreover, the major subregions of the striatum receive vastly different patterns of DA following unpredicted reward delivery, during 
motivated pursuit, and to conditioned stimuli. If regional heterogeneity is an adaptive feature of striatal DA dynamics, what are 
the organizational rules for large-scale DA transmission, and how do they facilitate computational/circuit operations in the 
service of behavioral flexibility?
 
An important clue is the functional architecture of hierarchical corticostriatal loops, wherein multiple striatal actors (or 
subregions) gate the selection of cortical actions at various functional levels of abstraction. A global DA RPE would equally 
reinforce all of these circuits, leading to inefficient learning when only a subset of them are responsible for rewards. Indeed, in 
theoretical models, robust learning in complex tasks requires RPEs that are preferentially directed to credit striatal actors/subregions 
in proportion to the extent of their participation in action selection. While such regional, actor-specific striatal RPEs are reported 
in human fMRI studies, we currently lack an empirical demonstration of whether DA signals are tailored to subregions according 
to their functional/computational specialty. Here, we used widefield imaging to assay DA dynamics over large territories of the 
dorsal striatum. We report spatiotemporally heterogeneous DA responses characterized by wave-like patterns that are regionally 
tailored to striatal targets as a function of task demands and predict animals' behavioral adjustments.
 
---
 
Results
 
Related Striatal Subregions Receive Correlated DA Input
 
We set out to study the large-scale organization of DA responses across the dorsal striatum (DS). Standard methods for DA assay 
have restricted spatial scale; we overcame these limitations by injecting a cre-dependent fluorescent calcium indicator GCaMP6f 
into the midbrain of mice expressing cre recombinase selectively in DA cells (DAT-cre mice) and captured DA-axon dynamics 
through an ~7 mm² chronic imaging window over the DS. This approach provided optical access to 60–80% of the dorsal surface 
of the mouse striatum, with a view of dorsomedial (DMS), dorsolateral (DLS), and partial access to the posterior-tail (TS) region 
of the striatum. A separate group of mice received striatal injection of the fluorescent DA sensor dLight followed by window 
surgery. We combined DA activity indicators with the expression of tdTomato to simultaneously capture inert red frames under 
dual-color, head-fixed preparations at multiple levels of resolution with one- or two-photon microscopy.
 
We first focused on spontaneous DA signals in a dark chamber without external stimuli. To test whether DA responses are globally 
synchronized, we compared fluorescence signals in DS regions of interest (ROIs). While ROIs were sometimes globally synchronized, 
we observed evidence of decorrelated activity across striatal subregions that temporally evolved. This regional variability was 
observed both in DA concentration and axonal calcium signals (dLight and GCaMP6f fluorescence) and was also apparent on the 
micrometer scale of DA terminals. Moreover, DA activity showed strong local correlations that gradually decreased with anatomical 
distance, comparable with the organization of striatal spiny-neuron activity. Strikingly, this distance-dependent falloff had a strong 
bias toward the mediolateral (ML) axis, that was not observed in simultaneously captured tdTomato frames. Together, these results 
demonstrate that DA inputs can become recruited asynchronously, hinting that the global DA hypothesis may need to be refined.
 
To further examine the topographical organization of DS DA, we used standard clustering analyses. In every dataset, the highest 
cluster threshold identified two contiguous territories in the field of view, outlining well-established DS subregions: DMS and DLS 
striatum. Increasing cluster limits progressively revealed smaller areas of DS, resembling striatal subdomains previously identified 
based on glutamatergic input patterns and behavioral specialty. We did not observe these territories when clustering control tdTomato 
frames, and shuffling the pixel-wise spatial (or temporal) order of GCaMP6f and dLight signals produced random clusters. Together, 
these results provide evidence for regional coordination of DA transmission and served as an initial basis for evaluating whether DA 
inputs are modulated by the underlying subregion's computational specialty.
 
Wave-like Patterns Coordinate DA Activity Across the DS
 
We next noted that the distance dependence of correlated DA activity patterns reflected an underlying organization of spatiotemporally 
continuous trajectories. In particular, both GCaMP6f and dLight fluorescence initiated in localized striatal zones and migrated across 
DS as DA axons become sequentially recruited to affect DA release in spatially contiguous regions. These trajectories, which we 
quantify below, resembled those described as traveling waves in cortical and subcortical brain regions. From here on, we use the 
DA wave terminology as a shorthand to describe the spatiotemporally continuous, flow-like patterns of dopaminergic activity 
across DS.
 
To quantitatively characterize these DA trajectories, we leveraged optic flow algorithms that extract frame-by-frame flow fields. 
The transient activation in DA axons (and release) originated from spatially clustered source regions defined by divergent vectors 
that signify outward flow. Once initiated, fluorescence migrated to neighboring striatal regions before terminating as a result of 
flow toward sink locations. DA waves entered the DS with exponentially decaying inter-wave intervals and propagated with a 
range of velocities. Moreover, the overall direction of flow was bimodally distributed, significantly biased to a ML propagation axis 
that was not present in simultaneously acquired tdTomato frames.
 
The flow-like property exhibited similar statistics for DA axon activation and release, indicating that axonal excitation and release 
may be coupled. To concretely test this possibility, we made dual-color widefield recordings in DAT-cre mice with cre-dependent, 
red-shifted calcium indicator jRGECO1a injected into the midbrain and dLight broadly expressed in the DS. Indeed, we found strong 
coupling between the simultaneously acquired dLight and jRGECO1a spatiotemporal flow patterns, with highly correlated temporal 
dynamics in the major striatal subdivisions that was not affected by the locomotor state of the mice.
 
We also examined whether the complex DA trajectories resulted from various imaging artifacts and/or damage to cortex and 
glutamatergic afferents during surgery for cannula implantation. We first ruled out the contribution of imaging artifacts related 
to locomotion and blood flow by imaging multiple DA activity sensors with spectrally separated inert fluorophores that did not 
display fast, spatially heterogeneous fluctuations. Second, we confirmed similar flow-like, sequential DA signals in absence of 
cortical resection in a separate group of animals that received small-diameter optic fibers arranged into a grid to minimize 
cortical damage. These findings lead us to conclude that wave-like activation patterns reflect a striatal DA circuit specialization for 
spatiotemporally coordinated dynamics.
 
Motif Waves Implement Systematic DA Phase Shifts Across DS
 
The propagation of wave-like dynamics could produce temporal delays in the arrival of DA transients across the striatum that, 
in turn, may regulate regional DA-dependent plasticity mechanisms. We asked whether elementary propagation trajectories could 
realize assorted temporal lead/lags in DA activation across DS. Using multiple convergent methods for the analysis of spatiotemporal 
sequences, we identified rudimentary motif patterns that affect DA dynamics across the DS. We focused our analyses on three motif 
waves that produced 93% ± 3% of the DA transients. First, center-out (CO) waves initiate at the juncture of DMS and DLS and rapidly 
spread bilaterally outward to produce DA signals that arrive at different striatal regions with little delay. Second, lateromedial (LM) 
waves start from the lateral striatum and predominantly propagate medially to deliver delayed DA transients to the DMS relative to 
DLS. Third, ML waves are sourced in the DMS and propagate laterally, activating DA axons in the medial striatum first and 
progressively recruited DA in lateral regions. These findings demonstrate that motif waves specify how DA responses initiate and 
propagate across DS, codifying the relative timing of regional DA that may shape striatal plasticity.
 
Rewards Evoke Directional DA Waves
 
What is the functional role of DA waves in adaptive behavior? We set out to determine the computational significance of DA 
trajectories in the context of the well-studied role of DS in instrumental behavior. The DS exhibits graded behavioral specialty, 
with the DMS implicated in agentic, goal-directed behaviors involving action-outcome learning and DLS implicated in 
stimulus-response behaviors. Inactivation or manipulation of DA in DMS degrades goal-directed planning and action due to an 
inability to learn whether rewards are under instrumental control.
 
To study whether DS DA is tailored to the target region's computational specialty, we designed two operant tasks that manipulated 
action-outcome contingency and asked whether DA dynamics carry information about instrumental controllability (i.e., agency). 
Auditory tones that escalated in frequency indicated progress to rewards in both tasks. In the instrumental task, this reward progress 
was contingent on, and tied to, the mouse running on a wheel to traverse a linearized distance. The distance to reward was randomly 
selected from a uniform distribution on each trial. In a second Pavlovian task, mice were free to run, but the tone transitions 
occurred independent of running, and the time to reward was drawn from a uniform distribution. Thus, the two tasks differed in 
instrumental controllability, but were structurally identical: tones provided information about progress to reward, which could not 
be inferred from elapsed time alone. Trained mice exhibited anticipatory lick trajectories that increased with ascending tone 
frequency in both tasks, indicating that mice used escalating tones to update their online judgment of progress to reward.
 
As in the spontaneous conditions reported above, DA waves were ubiquitous during task performance and were especially prevalent 
at reward. Notably, reward delivery immediately resynchronized DA responses into propagating waves that had opponent directions 
depending on task conditions. Specifically, rewards after an instrumental trial triggered medially sourced, laterally propagating (ML) 
waves, whereas rewards in the Pavlovian task promoted laterally initiated, medially propagating (LM) waves. These divergent 
responses in the two task conditions affected the temporal order of DA recruitment on the ML axis: DMS achieved peak reward-induced 
DA significantly sooner than lateral regions in the instrumental condition, whereas DMS had delayed DA peaks in the Pavlovian 
task. Moreover, these wave trajectories evolved with task experience, with reward-induced waves exhibiting irregular trajectories 
in naive animals but becoming more consistent and directional across several training days.
 
Wave-like Dynamics Support Graded Credit Assignment in RL Simulations
 
The dynamic sculpting of these trajectories by training and task demands suggested that DA waves may be important for behavioral 
flexibility. In particular, the continuous propagation of DA across the striatum in space and time motivated a revision of standard 
temporal difference (TD) RL models wherein a single reward value influences learning about reward-predictive events. We reasoned 
that these views could be expanded to include spatiotemporal differences in which waves carry additional, graded information about 
structural sub-circuits that are most likely to be responsible for rewards. To formally explore this account, we simulated the 
consequence of spatially delayed rewards in the tone tasks within a TD framework. The simulation contained a bank of parallel agents 
representing striatal subregions, and tone/state transitions formulated as sequential semi-Markov states. To explore the consequence 
of ML propagating waves, the reward response for the most medial agent was delivered immediately at the end of the trial and 
progressively delayed for more lateral agents. The model also included eligibility traces so that any delays in rewards could still 
be attributed to earlier states that were no longer active, in proportion to their decaying eligibility.
 
As learning progressed across trials, the RPE response in the most medial agent back-propagated to the earliest predictor of 
reward. However, in more lateral agents, delays in reward response led to progressively reduced credit assignment to the earlier 
states. Indeed, these effects translated to produce steeper value functions in the most medial agents as the agent progressed to 
reward, and lateral agents shallower ramps. Given that the value function reflects the reward value of the agent's predictions, which 
can be used to guide action selection, these simulations provide an initial algorithmic demonstration that reward-induced waves can 
give rise to asymmetric structural credit assignment.
 
DA Waves Track Changing Task Contingencies and Predict Behavioral Adaptation
 
For our behavioral tasks, we posited that opponent DA waves would facilitate reward-credit dissemination to specialized striatal 
regions depending on the animal's instrumental agency in advancing progress to reward. This hypothesis is inspired by expert-like 
organization of DS anatomy and graded specialization for action-outcome learning on the ML axis. Testing this possibility 
required task conditions wherein agency is dynamically manipulated in the same session. We thus trained a separate cohort of mice in 
a serial reversal task with changing reward contingencies across instrumental and Pavlovian blocks lasting 25–35 trials each. 
Mice experienced multiple unsignaled reversals in the same session, requiring continuous learning about agency. We predicted that 
reward-epoch DA trajectories should reverse directions after block transitions and predict the animal's future behavioral 
adjustments, with ML waves signaling agency and increase future running.
 
Trained mice completed an average of 6.4 ± 0.3 reversals across 210 ± 10 trials per session and dynamically adjusted their 
performance according to task contingencies. Specifically, mice completed instrumental blocks with a significantly higher run velocity 
and ramped down their velocity after they entered Pavlovian blocks. Replicating our previous findings in a different cohort of mice, 
we observed robust DA waves following instrumental and Pavlovian trials at reward delivery, with ML waves in instrumental trials 
and LM waves following Pavlovian trials. The wave reversals persisted across multiple block transitions for both axonal activation 
and DA release but were not observed in tdTomato frames. The wave dynamics were not simply related to differences in motoric output 
or velocity: opponent wave directions were observed even when velocities were matched across tasks. Moreover, in Pavlovian trials 
with elevated running velocity, wave directions were influenced by the locomotion-sensory congruence, defined as the correlation 
of wheel displacement and distance to reward in 250 ms bins. In particular, we found that spurious correlations between sensory 
evidences and (non-contingent) advance to reward in high-velocity Pavlovian trials promoted ML waves, indicating that wave 
directions are shaped by spurious evidence for instrumental control. These results support our prediction that DS wave trajectories are 
sensitive to task demands across the two conditions.
 
While these results confirm that wave trajectories dynamically shift across task contingencies, they do not establish whether they 
are involved in future behavioral adjustments. We thus tested whether DA wave directions at reward predict future-trial running in a 
history-dependent manner. We found that past wave angles were related to next-trial running speed and significantly correlated with 
run velocity in successive trials. Moreover, the effect of past wave directions on next-trial velocity had a history dependence, with 
more recent-trial DA wave directions demonstrating the largest velocity regression coefficients. These results are reminiscent 
of the impact of reward history in canonical RL models and data and support our second prediction. Together, our observations support 
the conclusion that DA wave trajectories are sensitive to evidence for agency and deliver opponent DA responses that predict the 
animal's behavioral adjustments according to task demands, manifested as adaptive running speed.
 
A MoE RL Model for Inferring Agency and Guiding DA Waves
 
The above-mentioned data support general predictions about the role of opponent DA trajectories in instrumental learning by directing 
reward credit to (and away from) DMS regions specialized for agency. However, these findings do not reveal how the mouse and the 
DA system infer controllability. In our tasks, the animal must make the critical inference of whether it controls reward-predictive 
tone transitions and which specific contingency (i.e., distance to run to advance tones) applies in the current trial. Thus, for mice to 
learn about agency and dynamically adjust their behaviors, the trial-by-trial evidence for instrumental control should determine 
whether reward-evoked DA will strengthen action-outcome learning (i.e., favor the DMS). In other words, online evidence for agency 
prescribes wave direction that, in turn, promotes (or suppresses) instrumental performance in subsequent trials. To formalize this 
notion, we constructed a hierarchical multi-agent mixture of experts (MoE) model, building on earlier models of corticostriatal 
interactions in learning and action.
 
At the highest layer (level 1) is an expert, putatively corresponding to DMS, that computes evidence that the agent is in control 
of outcomes (i.e., that its actions cause tone transitions and rewards). To do so, this expert must consider multiple potential 
action-outcome relationships, given the distribution of time/distance contingencies experienced in the task. As such, the expert has 
access to multiple sub-experts within its domain (level 2), each specialized to represent different contingencies (e.g., the distance 
needed to run is short, medium, or long trials). The expert can recruit the sub-expert that best predicts the state transitions in the 
current trial (i.e., the one with the smallest RPEs, minimizing the Bellman error). Moreover, auditory tone transitions that occur 
earlier or later than predicted give rise to sub-expert RPEs (sRPEs; level 3). For example, during a short distance trial, the short-distance 
sub-expert experiences reduced sRPEs, whereas a long-distance sub-expert experiences large sRPEs at tone transitions that occur 
earlier than expected. At the end of a trial, reward credit is delivered to experts that are most predictive of state transitions, which will 
guide future model running. Finally, the agent will increase its speed only when the accumulated evidence is larger for agentic 
distance experts than non-agentic time experts.
 
This formulation allows an agent to learn and flexibly adapt behavior based on task contingencies and expands the RL account of 
striatal DA such that it is informed by the inferred causal contributions of recipient subregions. Thus, in contrast to previous global 
scalar DA accounts, our model provides a formal framework for adaptive DA signals that are spatiotemporally tailored to striatal 
subregions. Moreover, this architecture makes multi-level predictions about DA dynamics during the reward pursuit and outcome 
epochs, potentially tying together the role of DA in performance and learning.
 
DA Ramps in DMS Signal Evidence for Agency and Predict Subsequent Reward Dynamics
 
If DA waves at reward guide spatiotemporal credit assignment, what determines which subregion should receive the credit? As noted 
above, the model contains a DMS-like distance expert that accumulates online evidence for agency in the form of ramping signals 
that are proportional to the accuracy of underlying subregions' predictions. Ramping DA signals during reward pursuit in the 
midbrain and ventral striatum has been described as scalar decision variables corresponding to RPEs, value functions, or progress 
within a cognitive map. Instead, we posit here that anticipatory DA ramps in a given DS subregion reflect the accuracy or usefulness 
of the underlying regions' predictions about task contingency, thus providing a tag for how much reward-credit it should receive at 
outcome. Thus, our model predicts that anticipatory epoch DA dynamics also diverge across striatal subregions and task demands.
 
We tested this prediction by examining DA activity during anticipation as mice drew closer to reward. In the instrumental task, we 
observed a buildup of activity in the DMS, ramping in proportion to the progress to reward. Strikingly, the opposite profile was 
observed in the Pavlovian condition with negative ramps even as mice drew closer to rewards. The opponent DMS ramp slopes were 
also observed in blockwise reversal sessions, with dLight and GCaMP6f ramps dynamically reversing after block change. The opposite 
profile of anticipatory DA signals across the two task conditions is not explained by extant models of midbrain or accumbens DA 
ramps. Instead, we interpret DS DA ramp dynamics as reflecting the value of the underlying subregion's agentic predictions, providing 
a marker for this region's reward responsibility. In addition, because reward credit should be proportional to the accuracy of these 
agentic predictions, our interpretation ties together opponent anticipatory dynamics with the opponent reward waves. We 
specifically posited that if DA ramps relay the subregion's reward-predictive accuracy, they would impact the subsequent timing of 
DA increases at reward, with regions assigned the most credit receiving the earliest DA bursts at reward. As such, trials with steepest 
ramps (highest responsibility) should receive reward responses soonest (largest credit). Consistent with this interpretation, we 
observed that DMS ramp slopes were inversely correlated with the latency-to-peak fluorescence following reward for both task 
conditions. The negative relationship between DMS ramp slope and latency to reward peak was also observed in DA dynamics of 
contingency reversal sessions, but not in simultaneously captured tdTomato frames. These findings indicate that anticipatory DA 
dynamics in DMS are modulated by instrumental contingency and predict regional reward responses, demonstrating a relationship 
between eligibility and reward credit.
 
Regional DA Ramps Tailored to Instrumental Contingencies
 
Thus far, we have focused on the coarsest division of labor related to the highest level in our model (controllability, level 1), but the 
agent's ability to infer control depends on underlying sub-experts that learn distinct action-outcome contingencies (level 2). Such a 
hierarchical scheme implies that within the DMS, smaller subregions should differentially express DA ramps for different distance 
contingencies. Indeed, we observed that DA ramp slopes were expressed across the ML axis of the striatum to different extents. We 
next tested whether territories of the DS exhibit specialized ramp profiles for different distance conditions and found that 
contiguous striatal regions expressed steepest DA ramps for a preferred set of trials with related distance requirements. We did 
not observe this regional contingency preference in simultaneously acquired tdTomato frames. These results are consistent with 
previous studies on progressive instrumental specialization of DS on the ML axis and support our MoE interpretations that DMS 
consists of smaller subregions that learn and express predictions for a variety of potential instrumental contingencies.
 
Reward-Predictive Sensory Events Evoke DA Transients Reminiscent of Sub-Expert RPEs
 
At the smallest scale (level 3), the evidence for each sub-expert is accrued based on the degree to which they experience RPEs 
(sRPEs) at state transitions. In the model, each auditory tone is represented as a unique state within a sub-expert's semi-Markov 
process, and sRPEs arise at tone transitions that occur earlier (or later) than expected. Thus, evidence for a given sub-expert is 
signaled by the relative lack of sRPEs compared with other sub-experts. This account predicts that tone transitions would give rise 
to rapid DA deflections reflecting sRPEs and that these signals would be modulated by trial length and position of tone within a trial. 
Specifically, the model predicts that (1) sRPEs would be larger in shorter trials (because tone transitions are indicative of future 
reward arriving earlier than expected); and (2) within a given task contingency, tones arriving later in the trial would drive larger 
deflections than early-trial tones due to temporal discounting.
 
Supporting these predictions, we observed abrupt DA responses at tone changes in both widefield, one-photon and two-photon 
preparations. Specifically, individual pixels during widefield DA imaging responded to multiple tone changes and consistently 
accompanied the sensory indicators of progress to reward. In contrast to these multitone responses in individual pixels of the 
widefield data, we noted that DA axon segments in the two-photon condition reliably responded to single tone transitions, and 
different portions of the imaged axon lattice tiled the full sequence of escalating tones. Next, we assessed whether these DA signals 
exhibited properties of sRPEs outlined above (i.e., trial length and position of tone in trial and not just sensory events or elapsed 
time). Indeed, tone-responsive GCaMP6f and dLight pixels in the widefield were significantly modulated by trial length, with larger 
responses in short-distance conditions. Moreover, these transients scale according to the position of the tone transitions within a 
trial, a result not observed in the control frames. The DA axon responses in the two-photon condition were also modulated by trial 
distance contingency and tone position in trial. Together, these observations indicate that sRPEs are represented in rapid DA responses 
at state transitions during anticipatory epoch, consistent with our model predictions.
 
---
 
Discussion
 
Our observations provide evidence for a spatiotemporal organizing principle of striatal DA signals and their behavioral relevance. 
Wave-like DA activation patterns were expressed as directional motifs that regulated the relative timing of regional DA changes 
and served to correlate DA in functionally related striatal territories. We reasoned that the computational significance of these 
waves in RL might be to assign spatiotemporal credit to striatal subregions differentially. Indeed, temporal delays on a similar 
timescale to those induced by DA waves are reported to constrain corticostriatal plasticity in vitro. Our TD simulations show that such 
temporal lags in reinforcement signals can drive spatially asymmetric reward learning and credit assignment. Thus, as hierarchically 
recruited striatal subregions exhibit graded functional specialization, DA waves may serve to regulate plasticity in postsynaptic 
domains with diverse functional specialization.
 
We tested this hypothesis according to the documented specialty of DMS in action-outcome learning and goal-directed behaviors. 
Our tasks manipulated reward controllability, requiring mice to dynamically learn about agency. Consistent with our hypothesis 
that DMS DA dynamics would be tailored to task demands, we found that reward delivery triggered DA waves in opponent directions 
based on task contingency. ML waves that produce rapid DMS DA peaks were enriched in instrumental trials, whereas LM waves were 
prevalent following non-contingent Pavlovian trials. Notably, these wave directions reversed within a few trials after task reversal 
and predicted future-trial behavioral adjustments with history-dependent effects in line with reinforcement learning. Together, 
our studies provide evidence for the role of spatiotemporal propagation of DA in agency learning by codifying the relative timing 
of a corticostriatal plasticity modulator.
 
Evidence for a Computational Model of Regionally Tailored DA Signals
 
The MoE model served to formalize our empirical observations, building on hierarchical neural network models of corticostriatal 
interactions. The model captures regional reward credit assignment in functionally specialized cortico-basal ganglia (BG) loops, 
inspired by previous anatomical and functional reports. In the MoE, evidence for instrumental controllability was accrued in the 
form of ramps to the DMS expert. In particular, as a trial progressed, sub-experts experienced prediction errors (sRPEs) when sensory 
events did not align as expected based on their specialization. Conversely, congruence between actions and predicted outcomes for 
a given sub-expert led to progressive ramps signaling their prediction accuracy and responsibility for impending rewards. In turn, 
these anticipatory ramps in the model will bias reward credit to distance experts to increase the agent's motoric output during the 
instrumental task but reduced running in the Pavlovian task.
 
Consistent with the MoE account, we reported anticipatory epoch DA ramping dynamics within large DMS regions that reversed 
directions between task conditions. Additional specialization was observed for distinct contingencies within smaller striatal 
subregions in the two tasks, consistent with sub-experts. We reasoned that these dynamics may serve a dual purpose. First, they could 
promote online behavioral vigor flexibility according to the inferred task contingencies in the current trial. Second, these ramps 
could also signal which subregions were best predictive of reward outcomes, providing a tag for their responsibility (akin to an 
eligibility trace in RL). Such a tag would allow RPEs to preferentially credit the appropriate subregion and the eligible 
MSNs within it. While the two functions are not mutually exclusive, our data provide strong support for the second interpretation: On 
a trial-by-trial basis, the degree of ramping in a given subregion was predictive of the latency to reward peak elicited by the wave. 
Moreover, the ramp slope and wave direction were predictive of subsequent-trial behavioral adjustments in line with the credit 
assignment implemented in the MoE. These findings accord with views that DA signals can have different functions during reward 
pursuit and outcome that can be gated by local microcircuit elements that regulate plasticity windows.
 
Further supporting the MoE organization, we also reported localized, transient RPEs that signaled changes in sensory events. 
These local transients exhibited key properties consistent with sRPEs according to our TD RL simulations: they were increasingly 
larger as trials progressed, and when task contingencies required shorter rather than longer distance running. We interpret these 
sRPEs as a mechanism by which sub-experts can report when they fail to predict the current task state. By comparing these errors 
across multiple actors, the system can accrue evidence for the most accurate expert (in the form of ramps). Notably, this interpretation 
hints at a different role for sRPEs (facilitating inference about responsible actors) compared with the large RPEs following reward 
itself (facilitating reinforcement learning): a dual operation that can also be gated.
 
Mechanisms That May Support Spatiotemporal Coordination of Striatal DA
 
Circuit mechanisms that facilitate the spatiotemporal coordination of striatal DA activity remain critical gaps in our understanding DA 
signaling. One hypothesis motivated by the excitation-release coupling principle in neurobiology would suggest that DA waves may 
be inherited from the sequential firing of topographically projecting midbrain DA cells. Previous reports of spiking in DA cell pairs 
report highly synchronized responses that inspired prevailing views for global DA release in recipient regions. Indeed, we did observe 
such synchronized DA events across DS, so our findings do not directly refute these hypotheses, but expand our understanding of DA 
signaling to additional, spatiotemporally complex activation trajectories with functional consequences. Nonetheless, population-level 
synchrony in midbrain DA cells and their relationship to DA waves remain open questions as limited studies have assessed the 
simultaneous firing of large populations (many hundreds/thousands) of projection-defined DA neurons. Moreover, the extent to 
which midbrain-initiated action potentials can fully propagate through an entire DA axon arbor in the face of energetic costs and 
GABA shunt currents remains unknown. Future studies into details of the functional anatomy and spike propagation principles in 
DA cells may uncover previously unappreciated axonal specializations or patterns of sequential recruitment in the midbrain cell 
bodies.
 
Another likely mechanism for DA waves may involve local modulation of DA axons and release in the striatum. Notably, striatal 
DA release can be evoked by cholinergic interneurons that can relay cortical or thalamic glutamatergic drive. Wave-like, 
spatiotemporal activation patterns have been reported in the neocortex and striatal cholinergic interneurons. Thus, local striatal 
microcircuitry (including GABAergic interactions) may regulate regional DA dynamics. Moreover, DA waves at reward outcome may 
also be a consequence of the interaction between primed excitability of DA axons during the anticipatory epoch and midbrain-sourced 
synchronous reward bursts. Combining these spatiotemporal profiles may produce sequential DA activation at reward that propagates 
across the striatum in proportion to the ramps during anticipation.
 
Limitations of Study
 
Although DMS DA in our report supports the computations of the distance expert in the MoE, a limitation of our study is that we 
did not identify or assess the DA dynamics with properties of the time expert in the DS. Many studies investigating RPEs 
involve classical conditioning in which temporal representations are evident in the midbrain, and ramping signals related to 
timing may be present in other regions upstream of the DA system. Nonetheless, even without a time expert per se, our MoE would 
behave similarly with a single DMS expert that simply evaluates the evidence for agency relative to some prior expectation about 
control. Moreover, while we make the case for how spatiotemporally coordinated DA responses may be involved in reward learning, an 
additional limitation of our study is that we did not deduce the mechanistic origin of DA waves. We have discussed multiple candidate 
mechanisms above.
