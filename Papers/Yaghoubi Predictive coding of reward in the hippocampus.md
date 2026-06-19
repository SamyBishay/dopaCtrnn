

 
Mohammad Yaghoubi, M. Ganesh Kumar, Andres Nieto-Posadas, Coralie-Anne Mosser, Thomas Gisiger, Emmanuel Wilson, Cengiz Pehlevan, Sylvain Williams, and Mark P. Brandon
 
Published: 2026
 
--- 
 
# Yaghoubi Predictive coding of reward in the hippocampus
 
### Abstract
 
Anticipating future outcomes is a fundamental task of the brain. This process requires learning the states of the world as well as the transitional relationships between those states. In rodents, the hippocampal spatial cognitive map is thought to be one such internal model. However, evidence for predictive coding and reward sensitivity in the hippocampal neuronal representation suggests that its role extends beyond purely spatial representation. How this reward representation evolves over extended experience remains unclear.
 
Here we track the evolution of the hippocampal reward representation over weeks as mice learn to solve a cognitively demanding reward-based task. We find several lines of evidence, both at the population and the single-cell level, indicating that the hippocampal representation becomes predictive of reward as the mouse learns the task over several weeks. Both the population-level encoding of reward and the proportion of reward-tuned neurons decrease with experience. At the same time, the representation of features that precede the reward increases with experience. By tracking reward-tuned neurons over time, we find that their activity gradually shifts from encoding the reward itself to representing preceding task features, indicating that experience drives a backward-shifted reorganization of neural activity to anticipate reward. We show that a temporal difference model of place fields recapitulates these results. Our findings underscore the dynamic nature of hippocampal representations and highlight their role in learning through the prediction of future outcomes.
 
The hippocampus represents a mixture of spatial (such as place, landmark, and so on) and non-spatial (such as time, sound frequency, and so on) environmental features. The collective encoding of environmental features and their relationships, known as a cognitive map, is thought to support spatial navigation and memory-related behaviors. From an evolutionary standpoint, an animal's survival depends on using these cognitive abilities to efficiently learn and remember rewarding experiences, such as navigating to home, safety, and food. This computation is supported by an experience-dependent spatial cognitive map in the hippocampus. Therefore, the hippocampal representation of the environment is expected to undergo considerable changes once an animal has learned how to navigate to rewarding locations or has mapped environmental features associated with rewards.
 
Previous studies have shown that the hippocampus encodes reward-related events at multiple time points, including reward approach, onset, location, and history. During reward approach, running toward a known goal induces place-specific firing patterns along the path that differ from firing during random foraging in the same environment. Place fields also cluster near reward sites, generating a reward over-representation. At reward arrival, a distinct group of hippocampal neurons consistently encodes reward delivery, independent of location or context, indicating that reward signals can be separated from place coding. The hippocampus also encodes reward history: after probabilistic reward delivery and after leaving the reward site, neuronal firing changes depending on the reward outcome. Although these studies describe how hippocampal representations change before and after learning reward locations, how these dynamics emerge and evolve with experience over days, weeks, or months remains unknown.
 
The hippocampus has been shown to support predictive models in various species. We propose that a reorganization of hippocampal representations—in particular, reward representation—during learning of a reward-based task will occur to support reward prediction. We examine this hypothesis by tracking the evolution of the hippocampal representation across weeks as mice perform a reward-based task.
 
--- 
 
### Calcium imaging of CA1 neurons
 
We used a one-photon miniaturized head-mounted fluorescent microscope to perform calcium imaging of CA1 of dorsal hippocampus in seven mice. Mice were injected with a viral construct to express GCaMP6f in dorsal CA1 and were implanted with a gradient refractive index (GRIN) lens targeting CA1. Calcium recording data were preprocessed to correct for motion artifacts, segment cells, extract calcium transients, and deconvolve the traces. We recorded 504 ± 101 (mean ± s.d.) neurons across sessions and mice.
 
We used a 20 × 18-cm automated touchscreen recording box to monitor mouse behavior. The box consists of a touchscreen in front, a reward port in the back, and an infrared camera on top to record behavior. Mice were trained on a delayed non-matching-to-location task, where a sample appeared randomly on the left or right screen after trial initiation. After a nose poke to the sample, the delay starts. At the end of the delay, a tone and light cue signaled the mouse to move to the back of the cage and break a beam to initiate the choice phase. During the choice phase, two white squares are displayed, and mice must choose the non-matching square to receive the reward. Mice performed one session per day. When the mouse reached a high level of performance, we increased the delay between the sample and the choice phase to make the task more challenging.
 
Mice exhibited increased performance for each delay duration over time.
 
--- 
 
### Reward encoding decreases with experience
 
We investigated the dynamics of reward representation in the hippocampus as mice learn to solve the delayed non-match-to-location task. The learning period varied, taking a few weeks depending on each mouse's learning rate.
 
To quantify the reward-encoding signal across sessions, we measured reward information at the population level using an information-theoretic analysis in the CEBRA-derived latent space. This framework enabled us to track reward representation changes with experience. At the single-cell level, we used a shuffle-control approach to identify reward cells per session and tracked their percentage across days. Both analyses indicate that reward representation declines mainly with experience, not performance.
 
Our data suggest that a dedicated subpopulation of cells is responsive to reward. Notably, distinct subpopulations encode the reward depending on whether the mouse approached the reward from the left or right choice on the touchscreen. The sorted calcium traces show that reward neurons are not necessarily tuned to the reward onset but form a reliable sequence spanning the entire duration of reward consumption.
 
Using CEBRA, we projected our deconvolved calcium traces into a 32-dimensional latent space. To quantify the information content of the reward representation, we used a fivefold cross-validation approach to decode the reward moments from latent space. The cross-fold-averaged mutual information (MI) between the decoded reward traces and the actual reward traces was regarded as the reward information content for each session. Correlating reward information content with session number (day) and mouse performance indicates a negative correlation with session number and a weak correlation with mouse performance. The result is consistent across mice. A linear model showed that variance in reward MI is explained mainly by experience, not by performance.
 
At the single-cell level, we used a shuffle-control procedure to identify reward cells. This resulted in 8.5 ± 1.5% of the cells being identified as reward cells. Reward-cell tuning curves show two features: (1) responses depend on the mouse's approach direction to the reward port; and (2) cells are tuned to distinct moments of reward consumption, extending beyond reward onset. Furthermore, reward cells exhibit greater firing during task engagement than during inter-trial intervals. Notably, consistent with the population-level analysis, the percentage of reward cells declined with session number but showed only a weak correlation with performance. Both population and single-cell-level analysis reveal that the reward representation decreases with experience.
 
To rule out potential preprocessing effects, we identified reward cells using both deconvolved traces and the area under the curve (AUC) of raw calcium signals, finding similar dynamics in reward MI and cell recruitment. Additional analyses confirmed that the decline in reward representation was not due to task difficulty (delay length) or behavior variability (running speed before reward).
 
--- 
 
### Pre-reward encoding increases with learning
 
In this section, we apply the same methodology used to measure hippocampal reward representation to quantify the evolution of hippocampal encoding of pre-reward moments. Specifically, we analyzed two pre-reward events: (1) screen, the [-150, 150]-ms window around the choice touch; and (2) reward approach, the interval between a choice and a reward as the mouse runs to the port.
 
Using the same methods as for reward, we assessed both population- and single-cell-level encoding of these events to track how their representations evolve with time. Distinct neuron subpopulations encoded left versus right choices at the touchscreen. We applied the same analysis used for reward to measure population-level screen information content. In contrast to that observed for reward, screen information increased with both session number and mouse performance. A linear model indicates that both factors contribute significantly to explaining the variance in the dynamics of the screen information content. A similar analysis for reward-approach encoding indicates a similar positive correlation for the reward-approach information content.
 
At the single-cell level, screen and reward-approach cells were identified using a shuffle-control procedure. We identified 7.5 ± 0.7% of the cells as screen cells and 5.7 ± 0.7% as reward-approach cells. The percentage of identified cells for both screen and reward-approach cells shows a positive correlation with both session number and mouse performance. A linear model reveals that both session number and performance contribute significantly to the dynamics of recruitment of screen and reward-approach cells.
 
Together, these results show distinct dynamics: with experience, measures of reward encoding decline at both population and single-cell levels, whereas measures of screen and reward-approach encoding increase.
 
--- 
 
### Backward shift of reward coding during learning
 
Across all mice, we were able to track 1,814 neurons. Out of 1,814 cells, 225 were reward cells (12.4%), 225 were screen cells (12.4%), and 53 were reward-approach cells (2.9%). The remaining 1,311 cells (72.3%) are labeled as non-classified cells.
 
Our data reveal that a significant number of reward cells exhibit a backward shift across sessions from reward to the reward approach and screen, termed as backward-shifting reward cells. Specifically, we report a significant negative correlation between the response timing and the session number for reward cells and reward-approach cells. Using a shuffle-control method, we found that 21% (47 out of 225) of tracked reward cells showed backward shifting—well above the 5% chance level. A substantial portion (60%; 28 out of 47 cells) of backward-shifting reward cells shifted enough to be classified as screen or reward-approach cells in later sessions. Detection of forward-shifting reward cells was at chance levels. Unlike reward cells, we found that screen and reward-approach cells exhibited a mixture of backward and forward shifting.
 
We also examined whether neuron response amplitudes changed across sessions. Using a similar approach to that used for temporal shifts, we correlated each neuron's peak amplitude with session number instead of peak timing. Many neurons across all cell types show declining activity over sessions, suggesting that, alongside backward shifts, reduced firing of some reward cells contributes to the population-level decrease in reward representation.
 
--- 
 
### A TD error model recapitulates the backward shift
 
The marked similarity of the backward-shifting reward cells to the reward prediction error (RPE) response observed in midbrain dopamine neurons motivated us to see whether a temporal difference (TD) learning model of the hippocampal representation could explain our observations. We focused on the segment from choice at the screen to reward, modeling it as a one-dimensional (1D) navigation task: the agent moves from state 1 (choice at screen) through to state 7 (reward port nose poke) and receives a reward at terminal state 8.
 
In our model, at initiation, 1,000 place cells uniformly tile the 1D state space with each cell's state selectivity described by a Gaussian radial basis function. The place-cell population activity is passed to a critic for value estimation and TD computation. The objective is to minimize the TD error by updating both the value function and place-cell peaks. This causes backward shifting of TD error from the reward to the start state, driving backward updates in state-value estimates and correspondingly backward shifts in place-cell peaks.
 
Three main reorganization patterns appear: (1) reward-proximal cells shift monotonically backward; (2) reward-approach cells first move towards the reward, then shift backwards; and (3) screen-proximal cells shift forwards late in learning. In addition, we extended the model to a policy-learning agent, in which place cells evolve to maximize rewards, mirroring animal behavior. Despite the added complexity, spatial selectivity still shifts as described.
 
Our modeling underscores the crucial role of reward predictability. Specifically, the backward shift is seen only when the reward discount factor, which determines the influence of future state values in the TD error calculation, is greater than 0.1. When the discount factor is less than 0.1, place cells remain over-represented at the reward without shifting backwards. This indicates that incorporating future state-value estimates into the TD error is essential for driving the backward shift, supporting the idea that a RPE-like signal underlies the dynamics observed in our experiments.
 
--- 
 
### Discussion
 
We combined large population recordings of mouse CA1 neurons with an automated touchscreen reward-based task to investigate the long-term dynamics of reward encoding in the hippocampus. Our data revealed a reduction in reward signal and an increase in the response to the cues that anticipate the reward. This was further supported by tracking individual cells that are at first tuned to the reward and gradually shift backwards to encode aspects of the task that are reward predictors. This backward shift in coding can be explained by a temporal difference reinforcement learning (TDRL) model of hippocampal place fields.
 
These results highlight a dynamic reorganization of hippocampal representations that supports learning by gradually shifting its coding toward cues that best predict future reward. Previous studies have revealed that hippocampal place fields move towards goal locations early in learning, probably contributing to what others have observed as an over-representation of rewarded locations. We also observe an over-representation of the reward location early in our recordings. Other work has shown a backward skew of hippocampal place fields, independent of reward locations, on a faster, within-session timescale. Together, these outcomes suggest that the hippocampal representation over-represents rewarded locations at first, and that this is followed by a slower, weeks-long shift to represent the cues that predict these rewards. Notably, our TD model also over-represents reward at first, followed by a backward shift of reward-tuned cells with experience.
 
The dynamics observed in our CA1 data mirror those of the dopaminergic output of the ventral tegmental area (VTA). This system is central to reward learning by RPE, as formalized by TDRL. TDRL has profoundly shaped our understanding of dopaminergic reward coding, a concept that has also influenced our understanding of hippocampal physiology. Prevalent implementations of TDRL make two key predictions: (1) a gradual decrease in reward response coupled with a gradual increase in response to reward-predicting cues during learning; and (2) a gradual backward temporal shift of the error signal from reward to cues during learning.
 
Both of these are well-documented in dopamine neurons and are also evident in our data. This resemblance suggests that the dynamics of hippocampal reward representations emerge from interactions within a broader circuit involving the hippocampus and VTA.
 
The model presented here extends TDRL by using Gaussian basis functions as spatial features, which reorganize through the TD error to improve state-value estimation and policy learning for reward maximization. Because these functions are modulated by the backward-shifting TD error, the resulting place fields also shift backwards from the reward. The successor representation algorithm also exhibits a backward shifting of fields in the presence of a reward, although it tends to maintain or increase field density at the reward location, which differs from the decrease we observe in our experimental data.
 
In conclusion, our study uncovers a dynamic and organized backward shift of the hippocampal reward representation during extended experience. Far from serving as a stationary spatial map, the hippocampus exhibits predictive coding, progressively tuning its representation to anticipate future rewards. These insights advance our understanding of the role of the hippocampus in learning, highlighting its crucial contribution to the brain's overarching objective of forecasting and optimizing future rewards.
 
