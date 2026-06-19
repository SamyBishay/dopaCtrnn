# Naudé Dopamine Builds and Reveals Reward-Associated Latent Behavioral Attractors

 
Jérémie Naudé, Matthieu X. B. Sarazin, Sarah Mondoloni, Bernadette Hannesse, Eléonore Vicq, 
Fabrice Amegandjin, Alexandre Mourot, Philippe Faure, Bruno Delord
 
Published: November 13, 2024
 
---
 
Abstract
 
Phasic variations in dopamine levels are interpreted as a teaching signal reinforcing rewarded behaviors. However, behavior also 
depends on the motivational, neuromodulatory effect of phasic dopamine. In this study, we reveal a neurodynamical principle that 
unifies these roles in a recurrent network-based decision architecture embodied through an action-perception loop with the task 
space, the MAGNet model. Dopamine optogenetic conditioning in mice was accounted for by an embodied network model in which 
attractors encode internal goals. Dopamine-dependent synaptic plasticity created "latent" attractors, to which dynamics converged, 
but only locally. Attractor basins were widened by dopamine-modulated synaptic excitability, rendering goals accessible globally, 
i.e., from distal positions. We validated these predictions optogenetically in mice: dopamine neuromodulation suddenly and 
specifically attracted animals toward rewarded locations, without off-target motor effects. We thus propose that motivational 
dopamine reveals dopamine-built attractors representing potential goals in a behavioral landscape.
 
---
 
Introduction
 
Transient, phasic dopamine (DA) release contributes both to learning (updating the values of actions, used to make future decisions 
based on experience) and to motivation (affecting ongoing decisions and invigorating goal-oriented behaviors), but reconciling 
these two roles within a unified theory of DA function has remained challenging. The popular reinforcement learning (RL) theory 
interprets phasic DA signaling as a reward-related teaching signal, which functions by modulating long-term synaptic plasticity to 
build neural representations of the value of actions that have previously led to reward. This role of DA in value learning is well 
demonstrated by the robust conditioned place preference induced by optogenetic stimulation of DA cells in the ventral tegmental 
area (VTA).
 
Although the original RL theory did not define a role for phasic DA signaling in ongoing behavior, there has been renewed interest in 
both experimentally linking phasic DA activity with motivation and building RL models that account for motivational DA. A large 
body of evidence supports that phasic DA neuron activity occurs just before self-paced movement initiation. However, the causal 
role of such phasic DA activity in the ongoing movement remains debated. While phasic optogenetic excitation or inhibition of DA 
neurons have been found to affect action initiation in some studies, in other settings manipulating DA activity did not have any effect 
on ongoing behavior. These conflicting results have proven hard to reconcile within a RL framework, despite continuous efforts.
 
Theoretical accounts suggest either a "directional" role with DA signals specifying the decision to be taken or an "activational" (or 
energizing) role, with DA determining the action latency and/or the level of motor resources to engage in performing an action. 
The encoding capacity of DA cells is limited, which suggests that their role is not directional. However, if DA gates decision-making 
by lowering a decision threshold, increasing the probability and reducing the latency of actions, it remains unclear how such action 
gating by DA would also affect the content (i.e., speed or direction) of movements. Moreover, it is important to note that 
manipulating phasic DA often does not have the same impact on all actions, which contradicts latency or decision-threshold models 
that would predict content-independent DA effects. DA signaling is primarily associated with, and necessary for, non-stereotyped, 
anticipatory, distal, or effortful behaviors, i.e., when some physical or cognitive distance separates the animal from a reward. 
Therefore, the reason why manipulating DA activity can affect both action latency, action direction, and movement vigor, but only in 
certain animal states and behavioral settings, remains totally enigmatic.
 
In the following, rather than starting from behavioral DA effects to build a phenomenological model, we sought to express DA 
motivational roles as a dynamical consequence of the known biophysical effects of phasic DA on target neural networks. DA 
modulation of synaptic plasticity is believed to build "Hebbian" assemblies of strongly interconnected neurons, representing a 
decision that was repeatedly rewarded. Activity within such Hebbian assemblies constitutes goal-encoding attractors, which attract 
network dynamics in their vicinity (i.e., their basin of attraction in the state-space). In standard attractor models, convergence from 
an arbitrary current state to a goal-encoding attractor can be either triggered by a specific cue stimulus or driven by noise 
fluctuations. However, humans and other animals can self-initiate goal-directed movements. Cue-induced convergence to a 
goal-encoding attractor does not account for instrumental decisions that are autonomously generated, based on internal 
(action-outcome) associations. Noise-induced switching to a goal-encoding attractor also provides an unsatisfactory account for 
internally-generated decisions that are driven by motivational states, rather than simply occurring randomly. We therefore assessed 
the possibility that phasic DA, based on its biophysical action, could provide a motivational signal affecting goal-encoding 
attractors online, i.e., during the decision process itself.
 
Here, by testing dynamical model predictions with experimental data, we demonstrate that the motivational role of phasic DA 
signaling is to reveal latent network attractors previously built by DA-modulated plasticity, thereby promoting the engagement of 
network activity into decision-related attractor dynamics. Specifically, we present a recurrent network-based decision architecture 
hereafter referred to as "Motivational Attraction to Goals by Network dynamics" ("MAGNet") model. MAGNet is embodied, through 
an action-perception loop, within the task space. We demonstrate how DA revealing latent—i.e., not systematically expressed—
attractors generates goal-directed actions toward previously rewarded locations. Therefore, we reinterpret the motivational role of 
phasic DA signaling as controlling the accessibility of attractors representing behavioral goals within a behavioral energy landscape.
 
---
 
Results
 
Optogenetics Stimulation of VTA DA Cells Produces Precise Place Preference and Motivated Behaviors
 
To characterize the role of phasic, transient dopamine (DA) signaling from the ventral tegmental area (VTA) in both reinforcement 
learning and motivation, we used an un-cued optogenetic conditioning task. We designed a task which requires mice to learn an 
internal memory of rewarded locations. We achieved selective manipulation of dopamine neurons by selectively expressing 
channelrhodopsin (ChR2) in the VTA dopamine neurons from dopamine transporter (DAT)-Cre mice. In a circular open-field, DA 
neurons were stimulated when mice were detected in one among three explicit locations. Mice cannot receive two consecutive 
photostimulations on the same location, so they alternated between the rewarded locations. Mice increased the number of 
photostimulations earned with learning sessions. This increased performance in ChR2 mice was due to a decrease in the distance 
traveled between successive rewarded locations, compared to controls, together with an increase in maximal speed. Hence, increased 
performance following place photostimulation pairings was due to a combination of directional and activational effects, which 
characterize motivated behaviors.
 
Goal-Directed Actions in an Embodied Biophysical Recurrent Neural Network
 
To dissect the roles of DA in reinforcement, through DA effects on synaptic plasticity (DA-plasticity), and in motivation, through 
online DA biophysical effects (DA-excitability), in our conditioning task, we developed a biophysical model consisting of a 
decision architecture assessing how an artificial mouse (i.e., an “e-mouse”) navigates under DA regulation. The model is referred 
to as the ‘Motivational Attraction to Goals by Network Dynamics' (MAGNet). As place-reward association relies on a distributed 
circuit comprising the PFC, basal ganglia, thalamus, hippocampus, and amygdala, we designed MAGNet as a distributed decision 
architecture, with different degrees of biological realism. To assess the impact of DA on attractor dynamics, goals were encoded by 
a recurrent neural network model of leaky integrate-and-fire neurons, with detailed biophysical realism. This recurrent network 
can be considered as the prefrontal stage of decision-making. This model network displayed mixed selectivity, i.e., neuronal encoding 
of both space (the current and desired animal positions) and reward (through DA-mediated learning). The network was organized 
topologically: neurons had a receptive field for the mouse position, i.e., feed-forward inputs putatively from hippocampal place 
cells, and in turn biased the animal's goal toward their preferred location. To restrain the model dimensions, and ensure that the effects 
are primarily due to attractor dynamics, we modeled the other stages of decision-making algorithmically. The internal goal was 
decoded from the recurrent network using a softmax selection rule, potentially representing some of the basal ganglia operations. 
Finally, the e-mouse converged toward its internal goal with speed ballistics, accounting for commands set by motor structures.
 
The recurrent neural network was thus embodied, in the sense that its activity determined the navigation of the e-mouse, which 
subsequently affected the spatial feedback input to the network. Hence, this formed an action-perception loop with the environment. 
Operation of such an embodied decision architecture fundamentally differs from that of a simple input-output network architecture. 
This is due to the non-trivial, circular causality between the animal and the task space it is immersed in. When spiking in the 
recurrent network was dominated by inputs encoding the e-mouse position, the internal goal was determined by the e-mouse position. 
When the internal goal was confounded with the current position, a default behavior (i.e., circling along walls with some inroads) 
mimicked that of real mice before learning. Conversely, a significant bump of activity in neurons encoding for a position distant 
from the current e-mouse position resulted in a shift of the internal goal to that distant position. Consequently, navigation was 
dominated by a convergence to the position encoded by the bump, i.e., the e-mouse position was driven toward the internal goal.
 
Motivation Emerges Through Two Distinct Biophysical Effects of Dopamine
 
We considered two different effects of DA on the biophysical properties of the neurons in MAGNet. DA enables long-term synaptic 
plasticity in cortical/subcortical areas. By reinforcing synaptic weights, DA links sensory states to rewarded actions. Here, we 
modeled DA consolidation of spike-timing dependent plasticity (STDP): correlated pre-postsynaptic activity led to eligibility 
traces (or synaptic tags) that were transformed by DA into actual excitatory synaptic changes. However, DA also modulates effective 
synaptic excitability by instantaneously potentiating the efficacy of NMDA currents, which are paramount in setting network dynamics. 
To disentangle the behavioral effects associated with these two biophysical properties, we considered different versions of the model 
including DA consolidation of synaptic plasticity (DA-plasticity), instantaneous DA NMDA upregulation (DA-excitability), or both.
 
Simulated phasic DA was delivered as a reward when the e-mouse crossed the rewarded locations, but also randomly during navigation 
to account for spontaneous DA occurring in mice. Prior to learning, navigation was governed by default behavior toward and along 
arena walls. As observed with real mice, e-mice learned three place–reward associations when navigating in the arena. Both DA 
effects on plasticity and excitability amplified the directional (decreased distance to reward) and activational (increased maximum 
speed) effects of DA, resulting in increased performance, as in real mice. The symmetric nature of the graphs suggests a synergistic 
effect of the two properties. Hence, multiple combinations of increases in both DA-excitability and DA-plasticity could equally account 
for our experimental data, suggesting that RL-type explanations of decision-making exclusively based on DA-plasticity may be incomplete.
 
We next assessed whether fluctuations in the phasic DA activity occurring before mice started a navigation bout toward the reward 
location (hereafter “pre-movement” DA activity), which is observed in similar settings, may help distinguish between the effects of 
DA-plasticity+excitability and DA-plasticity alone. We considered two alternative scenarios. First, pre-movement DA activity could 
passively reflect the history of previous DA release (i.e., coding the magnitude of DA-plasticity), as would be observed with a 
reward prediction error containing a prediction term. Second, pre-movement DA activity could constitute a motivational command 
on ongoing behavior (i.e., coding the magnitude of DA-excitability). Whatever the scenarios, DA activity correlated with movement 
speed, further suggesting that standard experimental measures cannot distinguish between DA-plasticity+excitability and 
DA-plasticity alone.
 
We then implemented the MAGNet model with a single rewarded location in the center of the arena to better distinguish between 
long-term (DA plasticity) and on-line (DA excitability) effects of DA signaling on decision making. With DA-plasticity only, 
simulated phasic DA delivered when the e-mouse crossed the rewarded location yielded long-term synaptic plastic modifications that 
accumulated over trials. The resulting strongly-connected Hebbian assembly encoded the place–reward association. By contrast, 
DA-excitability only transiently increases synaptic efficacy in the whole network, as a consequence of NMDA potentiation on a short 
timescale.
 
When navigating in the arena, the e-mouse converged more toward the rewarded location when considering that DA affected both 
plasticity and excitability, rather than only plasticity or only excitability. In the DA-plasticity+excitability condition, 
instantaneous NMDA potentiation had a larger, multiplicative effect on synapses already potentiated by DA-plasticity, resulting in 
a massive co-activation of neurons from the Hebbian assembly. DA-plasticity+excitability thus set the internal goal on the learned 
reward location, attracting the e-mouse.
 
Reduced Theoretical Model Uncovers That Dopamine Reveals Latent Attractors
 
We then exploited the radial symmetry of the environment to provide a reduced equation accounting for DA effects on animal behavior, 
and to derive experimental predictions. According to these biophysically-informed e-mouse simulations, instantaneous DA-excitability 
reveals long-term DA-plasticity reinforcement and drives goal-directed actions in mice. We analytically derived from the biophysical 
model a reduced model, which captures the hypothesized dynamical effects of DA without having the large number of free parameters of 
the biophysical model. In the MAGNet theory, the decision architecture including the animal position, neural network activity and 
internal goal, can be captured through a one-dimensional behavioral potential energy (BPE) governing e-mice behavior, similar to a 
particle in an energy landscape. Because of the revolution symmetry of the one-reward environment, BPE could be determined as a 
function of the e-mouse distance to the rewarded Hebbian assembly location. This reduced model summarizes that convergence to 
the rewarded location was dictated both by strong, local attractor dynamics, where the progressive increase in synaptic weights nearby 
the Hebbian assembly works to destabilize and attract neural activity, and weaker, global attractor dynamics due to focalization of the 
internal goal at the Hebbian assembly. Both of these terms required an instantaneous DA-excitability action on a previously 
DA-plasticity-reinforced Hebbian assembly, as they were negligible in the absence of either DA-plasticity or DA-excitability. 
Overall, under the DA-plasticity+excitability condition, phasic DA signaling induced the transient unfolding of a large and deep 
BPE basin of attraction, the subsequent focalization of the internal goal, and, ultimately, the convergence of the e-mouse to the 
rewarded location. Thus, DA-plasticity generated latent attractors that allowed only weak local convergence of internal goal and 
e-mouse positions. DA-excitability revealed these latent attractors, by amplifying both their depth and width, resulting in strong global 
convergence, which was not possible without previous reward learning.
 
MAGNet theory thus highlights the effects DA activity can have on ongoing behavior and the necessary conditions for DA to exert 
such effects. This allowed specific predictions to be tested with reward-seeking behavior in actual mice. Specifically, MAGNet theory 
predicts that, following an initial reinforcement of a central location, artificially stimulating DA when animals are in the periphery of 
the environment will increase the cumulative probability of convergence to the rewarded location if DA affects both plasticity and 
excitability, compared to other conditions. This effect would result from the unveiling of a goal-encoding attractor, inducing a sudden 
"magnetic" effect consisting in energization, with increased speed, attraction, with a decrease in the animal's distance to the reward, 
and a reorientation of their approach angle toward the reward location. Compared to previous experiments in which optogenetics DA 
release caused reinforcement, we expect that randomly stimulating DA in the periphery will avoid cumulating the DA-plasticity effects 
at the same location and thus specifically test how DA-excitability reveals previous DA-plasticity effects. Hence, MAGNet theory 
predicts that manipulating DA will affect animal movements only if there is an attractor in the behavioral energy landscape, e.g., 
in a context in which a central location has been previously rewarded, whereas DA manipulation will not exert any effect on behavior 
in another (neutral) context.
 
---
 
MAGNet Predictions Are Confirmed by Dopamine Manipulation in Mice
 
We then experimentally tested the predictions from the MAGNet model in an equivalent experimental setting. In a circular arena, 
we paired the central location with MFB electrical stimulation to establish the reward-place association, with mice having to leave 
the location before being stimulated again upon re-entry. This circular arena with an MFB-reinforced central location was considered 
as the reward (R) context, while a square open-field without any history of reinforcement was considered as the no reward (no-R) context. 
Once the association was learned, we then used VTA photostimulation to test for MAGNet's predictions on the context-dependent 
effects of increased DA on movement ballistics. We provided brief photo-stimulations when mice were away from the central position 
in the R and no-R contexts. VTA photostimulations increasing phasic DA signaling in the environment periphery were provided 
randomly in space to avoid cumulating the DA-plasticity effects that could eventually create a new rewarded location.
 
In ChR2-expressing mice tested in the R context, VTA photostimulation decreased the delay to the reward location compared to 
control times. This effect was neither observed in YFP-expressing animals, nor in ChR2-transduced animals in the no-R context. 
We next investigated whether this reduced delay following VTA photostimulation reflected an increase in speed. VTA photostimulation 
in the reward context resulted in an increase of animal speed, which was not observed in YFP controls. This online effect of VTA 
photostimulation on speed was consistent with the MAGNet model. Furthermore, VTA photostimulation did not affect speed in ChR2 
animals in the no-R context. Online manipulation of VTA DA signaling thus affected the speed of action, but only in the context in 
which a location had been rewarded, consistent with MAGNet prediction. Hence, the increase in animal speed after photostimulation 
of VTA DA neurons was directed toward the central location, consistent with MAGNet's first predictions. VTA DA photostimulation 
only attracted the animals toward the center location if this location had been previously rewarded, validating the second MAGNet's 
prediction.
 
MAGNet Theory Reconciles Previous Conflicting Results on Motivational Dopamine
 
Finally, we used a last prediction from MAGNet to reconcile the seemingly contradictory literature on the effects of optogenetic 
DA manipulation on ongoing behavior. MAGNet predicts that goal-directed convergence increases from relatively distal positions upon 
DA release at the periphery (when the initial position is far from the attractor), but not from positions closer to the central location 
(when the initial position is already close to the attractor). In our experimental data, we confirmed that the probability to go 
toward the central location was not affected when the VTA DA photo-stimulation occurred on proximal positions, but was increased 
for distal VTA DA photo-stimulation. In this context of investigating the “distance-to-attractor”-dependent effects of DA on ongoing 
behavior, we then assessed how MAGNet re-interprets previous experiments. Among the mixed evidence for a motivational role for 
DA, Hamid et al. found that optogenetic phasic activation of DA neurons shortened the latency for rats to engage in a reward-related 
task, but only when the rat was not already engaged in the goal-directed behavior. The reduced MAGNet model accounts for this increased 
action pace, as the effects of increased phasic DA signaling on the following movements depend on the animal's distance to the goal. 
Actions associated with a high initial distance to the attractor are sensitive to decreased phasic DA signaling. However, when the 
animal is already in close proximity to its goal, decreasing phasic DA signaling has no effect. Hence, MAGNet provides a dynamic 
biophysical ground and theoretical framework for the concept that high DA is not needed when an action is underway or when the 
goal is nearby, but crucial for a flexible approach toward distant, non-trivial goals.
 
---
 
Discussion
 
MAGNet theory interprets goal-directed actions as a two-step process: neural assemblies representing a potential goal are learned 
through synaptic plasticity regulated by reward-signaling dopamine (DA), but not systematically expressed, i.e., they constitute latent 
attractors in a behavioral energy landscape. Then, spontaneous or reward-predicting phasic DA neuromodulation renders these 
attractors accessible from distal starting positions, by widening and deepening their basins of attraction. We validated some of the 
predictions from the MAGNet theory experimentally, using optogenetics, showing that online phasic VTA DA signaling immediately 
orients the animal toward rewarded locations and energizes specific, context-dependent actions previously associated with phasic 
VTA DA. MAGNet also reconciles conflicting experimental evidence regarding whether phasic VTA DA affects ongoing behaviors or 
not, based on the distance between the animal state and its potential goal's basin of attraction.
 
How the MAGNet Theory Differs from Other Theories of Dopamine Function
 
Originally, reinforcement learning theory did not assign any effect to DA during ongoing behavior, once the value of actions has been 
learned. DA has then been suggested to exert either directional effects towards a specific goal, by itself or through stimulus-driven DA 
release that directs the behavior toward a cue, or activational effects, i.e., with DA increasing the probability and vigor of any motor 
behavior. Our theory proposes that DA exerts joint directional and activational effects, but only in contexts where a goal-encoding 
attractor exists in the behavioral landscape.
 
DA nuclei do not seem to have enough encoding capacity, and DA projections are not selective enough for a precise directional role, 
even though it is correlated with broad movement directions. Alternatively, DA is proposed to add incentive salience to the stimulus 
cue being currently processed, promoting approach. The DA-associated cue is described in incentive-salience accounts as becoming 
"magnetic," which is exactly what is expected in MAGNet theory for a state suddenly attracting the decision network's dynamics. 
However, actions that are not cue-driven but self-generated rely on internal representations, in which case the role of DA in 
incentive-salience is less specified. Our proposal generalizes the concept of incentive motivation by suggesting that it involves 
making goal-encoding attractors (either cued or internally generated) accessible.
 
Activational accounts of DA assign a role to phasic DA in gating decisions and/or energizing actions. In modified RL models, phasic 
DA could increase the probability to accept decision, and in drift-diffusion models, phasic DA has been suggested to move the 
decision threshold. These models predict an increase in the probability of all actions following VTA photostimulation, in opposition 
to our data showing an absence of DA effects outside the Reward context. Furthermore, we show that angle and speed profiles, not 
just latency or average speed, are affected by phasic DA, which go beyond the scope of these models.
 
In the context of working memory, tonic levels of prefrontal DA may maintain persistent activity encoding a goal. In this account, 
D2R favors stimulus-driven transitions toward another state by rendering attractors more shallow, while the current state is 
stabilized by D1R-mediated deepening of its basin of attraction. This model differs from ours, in which phasic DA activates D1R 
to widen basins of attraction, setting a new goal. DA may achieve a "double duty" in cognitive motivation by widening (to promote 
a decision) and deepening (to stabilize its working memory) basins of attraction.
 
Biophysical Implementation of the MAGNet Theory
 
Our biophysical implementation of MAGNet is derived from widespread findings from the literature. The attractorial principle of 
MAGNet is consistent with brainwide attractor dynamics; e.g., in the frontal cortex, but also in premotor, visual, or limbic structures. 
The current implementation of MAGNet relies on a recurrent network with cortical connectivity, but other implementations are 
possible, e.g., with cortico-striatal loops, given the known importance of mesolimbic DA for approaching rewards. In basal 
ganglia models, navigation toward goals can be learned through reinforcement-learning of synapses between space- and action-coding 
(striatal) neurons. Other basal ganglia models have proposed a link between action selection and action intensity, accounting for 
some of the roles of basal ganglia in energizing behaviors. DA regulation on both synaptic plasticity and excitability could 
result in multiplicative effects of DA on action selection and energization in a striatal model combining these features.
 
At the cellular level, we focused on NMDA modulation by DA at both the plasticity (long term) and excitability (short term) 
levels, but DA can also affect a vast diversity of receptors and ionic channels, depending on DA receptors. Here we mainly modeled 
D1R effects to account for approach behaviors, but D2R may not be as antagonistic to movement as previously believed: D1R and D2R 
may actually be synergistic for cortical plasticity, when considering the cAMP-PKA pathway we considered. For the regulation of 
intrinsic excitability, D2R may exert destabilizing influences (rather than inhibitory) that promote or oppose D1R effects depending 
on down or up-states, respectively. These interactions hint at complementary roles in our dynamical framework.
 
"Latent Attractor" as a Dynamical Framework Distinguishing Learning from Performance
 
Self-generated actions have proven hard to account for in classical attractor models. In such models, neural state transitions from 
spontaneous activity to decision attractors may be triggered by a destabilizing stimulus or driven by neural fluctuations. However, 
goal-directed decisions are neither random nor necessarily cue-triggered. Rather, they are self-generated, based on internal 
(action-outcome) representations. More refined models consider partially stable attractors, allowing dynamics to eventually 
escape and converge to another attractor. This requires specific mechanisms, either synaptic inhibition designed to repel the neural 
dynamics from the attractor or neuronal fatigue ensuring the attractor to be only transient once activated. Contrary to these models, 
the decision attractor simply vanishes in MAGNet, once the excitability effect of phasic DA decays due to DA recapture. Hence, in 
our theory, both entering into, as well as exiting from, a decision attractor are controlled by an internal operation.
 
Such internal control also effectively decouples the neural dynamics from synaptic changes, which is key to account for goal-directed 
actions. Usually, reward-dependent synaptic plasticity directly leads to a change in models' neural dynamics, yielding behavioral 
adaptation. However, animals do not always express learning as behavioral changes. Instead, some forms of learning are latent. 
For instance, a sated animal may learn to navigate a labyrinth containing a food source without increasing the visits to the food 
source, and, upon food deprivation, display a change in its behavior. MAGNet accounts for such latent learning by DA-modulated 
synaptic plasticity building latent attractors that do not necessarily affect neural dynamics. MAGNet theory decouples learning and 
performance because it considers the dynamical convergence in the joint neural and behavioral spaces. DA exerts a distant, 
discontinuous role that widens the decision's basin of attraction, so that the internal goal can be instantaneously set at a goal 
distant from the initial position.
 
Nevertheless, in the MAGNet theory, phasic DA should not be mistaken for a "homunculus" taking the decision to move. While we 
focused on the effect of DA rather than on the origin of phasic DA (which we considered here triggered either by a reward, 
spontaneous, or manipulated externally), the question of how phasic DA occurs in self-paced actions remains open. Our theory 
can combine with time-difference accounts of reward-prediction errors, in which the reward prediction term (that would be observed 
at the beginning of self-paced movements) would be used for initiating and controlling goal-directed actions.
 
Overall, this study is in line with the current paradigmatic shift regarding neurodynamics: instead of being permanently attracted 
by Hebbian attractors, collective dynamics within neural circuits may rather be governed through latent attractors controlled by 
context-related phasic neuromodulation, thus expressing specific, learned goal-directed actions only in certain brain states.
 
