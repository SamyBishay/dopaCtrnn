---
title: Lloyd Tamping Ramping; Algorithmic, Implementational, and Computational Explanations of Phasic Dopamine Signals in the Accumbens
---

# Lloyd Tamping Ramping; Algorithmic, Implementational, and Computational Explanations of Phasic Dopamine Signals in the Accumbens

Kevin Lloyd, Peter Dayan
 
Published: December 23, 2015

Abstract
 
Substantial evidence suggests that the phasic activity of dopamine neurons represents reinforcement learning's temporal difference 
prediction error. However, recent reports of ramp-like increases in dopamine concentration in the striatum when animals are about to act, 
or are about to reach rewards, appear to pose a challenge to established thinking. This is because the implied activity is persistently 
predictable by preceding stimuli, and so cannot arise as this sort of prediction error. Here, we explore three possible accounts of 
such ramping signals: (a) the resolution of uncertainty about the timing of action; (b) the direct influence of dopamine over 
mechanisms associated with making choices; and (c) a new model of discounted vigour. Collectively, these suggest that dopamine 
ramps may be explained, with only minor disturbance, by standard theoretical ideas, though urgent questions remain regarding their 
proximal cause. We suggest experimental approaches to disentangling which of the proposed mechanisms are responsible for dopamine ramps.
 
---
 
Author Summary
 
Dopamine has long been implicated in reward-motivated behaviour. Theory and experiments suggest that activity of dopamine-containing 
neurons resembles a temporally-sophisticated prediction error used to learn expectations of future reward. This account would appear to 
be inconsistent with recent observations of 'ramps', i.e., gradual increases in extracellular dopamine concentration prior to the execution 
of actions or the acquisition of rewards. We explore three different possible explanations of such ramping signals as arising: (a) when 
subjects experience uncertainty about when actions will be executed; (b) when dopamine itself influences the timecourse of choice; and 
(c) under a new model in which 'quasi-tonic' dopamine signals arise through a form of temporal discounting. We thereby show that 
dopamine ramps can be integrated with current theories, and also suggest experiments to clarify which mechanisms are involved.
 
---
 
Introduction
 
Ideas from the field of reinforcement learning (RL) have played an important role in neuroscientific theories of how animals choose 
actions to gain rewards and avoid punishments. Prominently, it has been suggested that the phasic responses of midbrain dopaminergic 
neurons resemble a temporal difference (TD) error, a learning signal which facilitates prediction and control of rewarding events. 
Consistent with this notion, these neurons are activated by unpredicted primary rewards and by cues that predict such rewards, but not 
by rewards that are themselves reliably predicted. More recent experiments using fast scan cyclic voltammetry (FSCV) to measure rapid 
changes in extracellular dopamine concentration within projection areas, notably the nucleus accumbens (NAc), find transients which 
show similar TD-like properties.
 
However, recent reports of ramp-like increases in dopamine concentration preceding self-initiated instrumental responses and during 
approach to spatial locations associated with reward appear to pose a challenge to established thinking. The central issue for TD 
accounts of dopamine is why such ramping should be observed at all, since TD provides a mechanism for predicting away later 
dopaminergic activity by earlier—as in the case of the transfer of activity from the time of reward to the time of predictive cues.
 
One possibility is that these signals have no functional importance, for instance being the result of a process of gated release. In this, a form 
of ramping activity in the glutamatergic cortico-striatal input might cause the terminals of the dopamine neurons to discharge more of the 
neuromodulator. This would account for the excess release without any implication for the activity of dopamine neurons—and would mainly 
pose the question as to how the altered pattern of release could have no effect on striatal activity or plasticity. In the current paper, 
however, we consider three possible, non-mutually exclusive, functional explanations of NAc dopamine ramps.
 
---
 
Phasic Dopamine and TD Error
 
In the main class of TD models of the phasic dopamine response, the computational goal of learning is to predict from each state s the 
expected discounted sum V(s) of the rewards that will be encountered during a trial:
 
V(s) = E{γ^0 r_t + γ^1 r_(t+1) + γ^2 r_(t+2) + ... | s_t = s},
 
where r_t is the reward delivered at time t, and 0 ≤ γ ≤ 1 is a discount factor that controls how much weight is given to future relative 
to immediate rewards. Crucially, the definition of this state value function satisfies a (Bellman) consistency condition with respect 
to each possible next state s':
 
V(s) = E{r_t + γ V(s')}.
 
This leads to the idea of using local discrepancies in the value of sampled successive states to drive learning. Thus, the TD error δ_t 
is defined as:
 
δ_t = r_t + γ V(s_(t+1)) - V(s_t),
 
and can be used to improve estimates of V(s). It is exactly this TD error that phasic dopaminergic activity has been hypothesized to represent.
 
As noted, in this paper, we consider data on dopamine concentrations in target structures (denoted [DA]) rather than the phasic activity 
of dopaminergic neurons. These quantities are known to be related; we assume this relationship is simple—a ‘dopamine response 
function' (DRF) based qualitatively on the signal evoked in NAc by VTA stimulation. We model the DRF using an alpha function:
 
f(t) = t ξ e^(1 - 1), with time constant ξ = 0.7s set to match experimental observations.
 
In other words, dopaminergic activity at time t, which we denote δ_t^P (a phasic TD error), causes an increase in dopamine concentration 
that peaks after a delay of ξ seconds and then decays with time constant ξ. Thus, changes in dopamine concentration levels relative to 
baseline, Δ[DA], are acquired by convolving time-varying activity δ_t^P with the DRF described.
 
We should note two important caveats to this model. First, there is evidence for richer temporal and non-linear structure in the DRF, 
albeit perhaps most affecting timescales and strengths of responding that are different from those considered here. Of more immediate 
note is that while there is evidence that fluctuations in dopamine concentration within NAc symmetrically encode positive and negative 
prediction errors, other studies do not show such clear negative deviations from baseline corresponding to a negative prediction error. 
Indeed, evidence suggests that negative prediction errors are represented differently from positive prediction errors in the activity 
of midbrain dopaminergic neurons: while positive prediction errors appear to correlate positively with the firing rates of dopaminergic 
neurons, the magnitude of negative prediction errors correlates rather with the duration of a pause in burst firing.
 
The second caveat is that modulation of striatal dopamine concentrations can occur independently of changes in the observed firing 
rates of dopaminergic cells. Thus, tonic levels of striatal dopamine are thought to be controlled by the number of active dopaminergic 
cells rather than by the firing rates of a fixed pool of neurons. Furthermore, a range of mechanisms local to the striatum are known to 
play a role in regulating dopamine release, including a host of other neurotransmitters such as glutamate, acetylcholine, and GABA.
 
---
 
Actors and Critics
 
In a case more general than that of learning purely to predict, animals may be allowed to select actions to achieve desired outcomes. A 
mapping from states to actions is usually referred to as a policy, denoted π, and the more general problem is to find a policy which 
maximizes some measure of reward. The TD error signal defined can be used to evaluate state values with respect to a given policy, 
V^π(s). Given this value function, the agent can potentially improve on its current policy by selecting actions that lead to successor 
states of higher value. Iteration between successive steps of policy evaluation and policy improvement characterizes the policy iteration 
algorithm, which is a cornerstone of RL methods.
 
The actor-critic algorithm, an asynchronous version of policy iteration, is just one of a number of TD-based suggestions for RL. However, 
it has played a particularly salient role in neural RL modelling. In the actor-critic architecture, state values and policy are explicitly 
represented in different memory structures. The policy structure is known as the actor, since it is responsible for selecting actions; 
and the value structure is known as the critic, since it criticizes actions taken by the actor, where this critique takes the form of the 
TD error described above.
 
In terms of neural substrate, it has been suggested that the dual learning functions of the actor-critic map to a fundamental division 
in the functional anatomy of striatum into dorsal and ventral subregions. In particular, the ventral striatum (NAc) is implicated in 
reward and motivation, while the dorsal striatum is implicated in motor and cognitive control. This dissociation is consistent with an 
implementation of actor and critic components in the dorsal and ventral striatum, respectively.
 
---
 
Tonic Dopamine and Vigour
 
Initial theorizing in neural RL focused on tasks involving a simple action or choice between different discrete actions in response to an 
explicit experimental cue. More recent modelling work has sought to extend standard RL models to other dimensions of choice, thereby 
making contact with the large experimental literature on free operant tasks in which subjects not only choose between different actions but 
also when and how quickly to act.
 
Two key differences from previous work have been involved in the first collection of models of free operant tasks. Firstly, the agent not 
only chooses an action a to perform, but also an associated latency τ with which to perform it. Formally, this entails moving from the 
usual discrete Markov decision process (MDP) model, in which agent-environment interactions progress at fixed time intervals, to a 
semi-Markov decision process (SMDP), which permits the time spent in a particular state to follow an arbitrary probability distribution. 
Secondly, rather than assuming that the agent aims to maximize an expected sum of discounted future rewards, models have assumed an 
average reward criterion. In this case, the aim is to find a policy that maximizes the long-run average reward rate, which is independent 
of starting state, assuming ergodicity.
 
The value of a state under policy π is now defined relative to the long-run average reward under that policy, and can be denoted as the 
relative value. Similarly, the relative action value of taking action a in state s is defined as the expected sum of differences between 
rewards and the average reward rate.
 
The connection to current concerns is the proposal that the tonic level of dopamine, especially in NAc, represents the long-run average 
rate of reward, effectively signalling an opportunity cost of sloth. This suggestion is based on a long literature implicating dopamine in 
the modulation of behavioural vigour.
 
---
 
Ramping Dopamine Concentrations
 
A first example of the phenomena of interest comes from an experiment by Roitman et al. very similar in structure to the lever pressing 
case considered above. Following presentation of an explicit cue, a rat could press a lever at a time of its own choosing to receive a sucrose 
reward. Cue presentation evoked an increase in dopamine concentration in NAc, but not in control animals for which a lever press did not 
yield reward. However, Roitman et al. also observed that, when aligned to the time of lever pressing, average dopamine concentration 
began to increase a short time before the time of the lever press itself, reaching peak concentration around the time of pressing. This 
occurred not only on the majority of the trials in which animals pressed the lever at relatively short latencies following the initial cue, 
but also on the smaller number of trials in which animals responded at longer latencies. Similar increases in extracellular dopamine just 
prior to response have been reported in other FSCV studies.
 
A second, perhaps more dramatic, example of dopamine ramping has recently been reported by Howe et al. In this study, dopamine 
concentrations in the striatum were measured using FSCV while rats navigated mazes to obtain remote rewards. It found a gradual increase 
in dopamine concentration that began at trial onset and ended after reaching the goal. Whether rats took a relatively short or long time to 
reach the goal, dopamine peaked at similar concentrations at the goal. Similarly, dopamine peaked at comparable concentrations at the 
goal for mazes of different length. Single-trial examples in which rats paused mid-run showed a remarkable correspondence between 
proximity to the goal and dopamine concentration. Furthermore, dopamine ramps scaled with size of reward, so that peak dopamine was 
higher for larger than smaller rewards.
 
While we take both of these examples to be instances of dopamine ramping, their explanations may not be identical. Nevertheless, 
neither case seems to fit neatly with standard RL models because apparently reliable activity is not predicted away by earlier reliable cues.
 
---
 
Models and Results
 
We consider three possible, non-mutually exclusive, explanations of NAc dopamine ramps. First, we consider possible sources of 
predictive uncertainty arising within the actor-critic about when actions will be performed. We show that a TD account in which a 
prediction error is generated when such uncertainty is resolved just before the action itself may explain pre-response increases in dopamine 
such as those observed by Roitman et al. Second, we consider a more direct role for dopamine in decision-making, specifically in setting 
the gain of a diffusion-to-bound process of value integration. We show that both tonic and phasic fluctuations in dopamine concentration 
produce what look like average ramping signals in dopamine leading up to the time of decision. Third, we consider the possibility that 
the prolonged ramping signals observed by Howe et al. may reflect an average reward-like signal that arises within the discounted reward 
framework. We show that the quasi-tonic signal suggested by our analysis has just the right properties to explain the ramping phenomena 
observed by Howe et al.
 
---
 
When Will I Act? Uncertainty About Action Timing Within the Actor-Critic
 
Whether an animal faces a task in which it is free to respond as often and as quickly as it likes, or is limited to a single response within an 
interval following a cue, it typically has at least some freedom to choose its time of response. In the case of Roitman et al. described above, 
rats were free to lever press at a time of their own choosing following a cue marking the start of a new trial. As reported in a number of 
similar studies, ramp-like increases in NAc dopamine concentration which preceded the time of lever-pressing were observed.
 
From a conventional TD perspective, phasic dopaminergic activity reflects a prediction error. Such errors can be occasioned by changes 
in latent states associated with the subject's internal execution of the task, provided that there is some uncertainty associated with 
these changes. Such uncertainty can be generated by two forms of ignorance: what the critic fails to know about the actor's choice of when 
to act, and what both actor and critic fail to know about the passage of time.
 
Consider first the critic's knowledge about the temporal decisions of the actor. We assume that internal information proximal to the action, 
such as some form of motor preparation, is communicated to the critic via efference copy just before it is evident to the experimenter. 
This resolves any uncertainty the critic may have about the time of the impending action. The question is what happens at the time that 
the actor makes its decision about the latency of lever pressing following the initial cue. There are two natural possibilities. One is 
that the actor also intimates its decision about when to act directly to the critic at that time. The other is that the critic has no such 
privileged access to the actor's initial decision, implying that its predictions could be based only on its experience of downstream signals 
resulting from the actor's choices.
 
A second, related issue concerns the realization of timing. If the actor communicates its choice to the critic and the two share the same 
clock, then there seems to be little room for timing uncertainty to affect the critic's predictions. On the other hand, if the actor does 
not specify an exact time of action, or its decisions are subject to additional sources of what the critic will experience as uncontrolled 
variability, timing uncertainty may play a role in the critic's predictions and resulting prediction errors.
 
To explore these issues, we consider the same lever-pressing task described previously, though with a state space that is augmented 
to reflect the assumption that the critic may receive internal information about the lever press just before it occurs. As before, an initial 
cue is observed, prompting selection of a latency τ with which to press the lever. After the selected duration τ, which may or may not 
be known by the critic, the animal transitions to a state of preparedness to press, assumed to be communicated to the critic via efference 
copy. Completion of the lever press itself occurs only after a further interval τ_post. A reward of utility r = 1 is delivered on press 
completion. Completion of the lever press and reward delivery is followed by a fixed inter-trial interval τ_I = 30 s, after which the 
process begins anew.
 
The role of the actor in this scenario is simply to make repeated choices about the latency to lever press. We assume that this choice 
is always made immediately after presentation of the cue. What matters for present purposes is that either through stochastic selection 
or stochastic execution, there will be a distribution of times that it takes for proximal news of the action to be reported to the critic 
via efference copy. We therefore treat this efference copy time as a random variable T which, for convenience, we assume to follow a 
gamma distribution.
 
The role of the critic is to learn the relative state values corresponding to the actor's policy. In the case where the critic only receives 
indirect information about the actor's choices, the critic will nevertheless have expectations about T based on past experience. Such 
expectations can be summarized in the form of a ‘prior' distribution P(T). If the critic additionally receives direct information about the 
actor's choice, the critic can update its beliefs about when engagement will occur based on this information.
 
Given the critic's relative state values, we are particularly interested in TD errors and their dopaminergic instantiation. TD errors are 
inevitable in all cases we consider, either due to the random nature of T in the case of indirect communication, or due to timing 
uncertainty in the case where there is additional direct communication of τ. Under the conventional average reward formulation, TD errors 
take the form:
 
δ_t = r_t + V^π(s_(t+1)) - V^π(s_t) - ρ^π,
 
where δ_t^p is assumed to constitute the phasic component of the error signal reflected in phasic dopaminergic activity, and average 
reward rate ρ^π is assumed to be reflected in a constant, tonic level of dopamine.
 
---
 
Uncertainty Resolution, TD Errors, and the Pre-Response Dopamine Signal
 
Given the models described in the previous section, we consider results from three different cases: two in which the critic only 
receives information about the lever press indirectly, and one in which the critic additionally receives direct information from the actor. 
In each case, we consider the effect of the critic receiving notice of impending action at different times—T = {1, 3, 10} seconds—on 
the TD error δ_t^p, and evaluate the resulting change in dopamine concentration Δ[DA] under both symmetric and asymmetric encoding 
assumptions.
 
In the case of a constant hazard function, corresponding to T ~ G(1,1), the size of TD error occurring on transition to the state of 
preparedness does not vary with latency. This is precisely because the conditional probability of this transition does not vary over time. 
Note also that after an initial positive TD error, the error signal remains at a constant negative value between the time of cue 
presentation and the time at which the critic receives efference copy. This constant negative TD error is again a consequence of the 
flat hazard function.
 
In the G(2,1) case, the hazard rate is not constant. Then, efference-related TD errors decrease with longer latencies. This is due 
to the monotonic increase in probability that the lever press will occur with the passage of time—the event is increasingly expected. 
The decrease in TD error for longer latencies is mirrored in a decrease in the peak [DA] signal.
 
In the case that the critic additionally receives initial information about the actor's choice of latency, exactly the opposite trend is 
observed in TD errors occurring just before pressing: they increase with latency. This is due to the assumption that the critic is more 
uncertain about the time of engagement for longer choices of τ. Conversely, TD errors occurring just after the cue, corresponding to 
the time at which the critic receives initial information about the actor's choice, decrease with τ.
 
It is this case, especially when positive and negative TD errors are differentially scaled, that seems to offer the best qualitative fit 
to the results in Roitman et al. Not only do we see a similar signal produced by presentation of the cue, but we see a qualitative match 
in press-aligned average signal for short- and long- latency trials. Thus, on short-latency trials, we see a pronounced ramping which peaks 
at the time of the press. Furthermore, we observe no difference in peak signal when aligned to either cue or press events. On long-latency 
trials, just as seen in Roitman et al.'s data, ramping is somewhat less pronounced but similarly begins prior to the press and peaks around 
the time of press completion. Furthermore, unlike short-latency trials, the peak [DA] signal is significantly larger around the time of the 
press than at the time of the cue.
 
---
 
A More Direct Role for Dopamine: Setting the Gain of Value Accumulation
 
Our first possible account of ramping, the TD account of pre-response signals described above, assigns dopamine a passive role in 
decision-making: increases in dopamine reflect a latent state transition arising from a decision to act which has already been made. 
However, experimental evidence suggests that accumbens dopamine could also play a more causal role. For example, Phillips et al. 
found that electrically-evoked dopamine transients in NAc increased the probability that rats would lever press for cocaine immediately 
afterwards. Such findings have led to the suggestion that accumbens dopamine is necessary for ‘flexible approach'.
 
We next explore a second potential mechanism for ramping signals. In particular, we show that a particular decision-making scheme which 
couples dopamine directly to the decision process also generates dopamine ramps.
 
A rich vein of work in psychology and neuroscience revolves around the idea that the brain implements some version of the sequential 
probability ratio test (SPRT), a sometimes optimal procedure for two-alternative forced-choice decisions under uncertainty. A prominent 
realization of the SPRT is the so-called drift-diffusion model (DDM). In the DDM, evidence x(t) is accumulated according to:
 
dx = A dt + c dW, x(0) = 0,
 
where the constant drift A represents the average increase in evidence supporting the correct choice per unit time, and c dW represents 
white noise which is Gaussian-distributed. In the free-response case of interest here, the process terminates when x reaches a fixed 
threshold ±z.
 
We consider the slightly augmented DDM in which the drift and diffusion constants vary over time:
 
dx = g(t)[A dt + c dW], x(0) = 0,
 
where g(t) is the time-varying gain which controls the drift and noise, and which we assume directly reflects dopamine concentration.
 
Dopamine dynamics. We consider the additive effects of two sorts of fluctuation in g(t): tonic and phasic. In the tonic case, dopamine is 
assumed to fluctuate in an autocorrelated manner around some constant level. In the phasic case, we consider the addition of a more 
dramatic change in dopamine concentration, notionally driven by TD-related phasic activity of dopamine cells occasioned either by an 
external cue or by a latent event internal to the animal.
 
Tonic and phasic dopamine fluctuations produce average ramping signals. Tonic fluctuations: Even though dopamine fluctuations here 
are driven purely by noise, averaging over dopamine signals aligned to the time of decision reveals a clear ramping of this average 
signal towards decision time. This averaging phenomenon is due to threshold-crossing events being more likely to occur when [DA] is high, 
and also to the fact that the [DA] time series is autocorrelated.
 
Phasic fluctuations: The addition of strong phasic fluctuations, notionally driven by TD-related activity, also generates an average 
ramping signal. Of note in this case is the negative correlation between the magnitude of the TD response h and latency. This is in 
accord with the finding that the size of phasic responses of dopaminergic cells to a start cue is associated with a shorter latency of 
behavioural response.
 
---
 
Ramping as State Prediction
 
We now consider a third account of dopamine ramps based on a new model of discounted vigour. Incorporating the observations and suggestions 
of Howe et al., together with a partially free-operant experiment, we suggest that the concentration of dopamine measured by FSCV in the 
accumbens might be strongly influenced by the discounted value function V(s) of the state. This will show evidence of ramping 
towards final goal states when the discount factor is less than 1, consistent with the observations of Howe et al.
 
We reconciled an apparent inconsistency between the definitions of TD errors in the cases of average and discounted reward via an 
analysis in which ramp-like signals would be expected to emerge. In particular, we suggested that the quantity (1 - γ)⟨V^γ(s_(t+1))⟩ in 
the discounted reward model plays an equivalent role to the average reward rate ρ in the average reward model. Since values often 
change modestly as a result of the passage of time, this signal is quasi-tonic, and thus a candidate for what would be recorded using 
a technique such as FSCV. This signal can explain the ramping phenomena observed by Howe et al. and also those observed in more 
recent experimental work.
 
Our analysis suggests that ramps are scaled by the discount factor γ, prompting the question of how this discount factor is set, 
whether it is variable or fixed, and indeed, whether it is unique. There is substantial evidence that human and animal discounting 
takes a hyperbolic form rather than being exponential as considered here. This can arise from a combination of two or more exponentials, 
and it would be most interesting to extend our analysis to this case.
 
---
 
Discussion
 
The observation of ramp-like increases in dopamine concentration within the nucleus accumbens appears to pose a challenge to existing 
computational accounts of dopamine's role. Here, we explored three different explanations for such signals: (a) resolution of uncertainty 
about the timing of action within an actor-critic, leading to a prediction error shortly preceding the action itself; (b) positive 
correlations between the time of action and dopamine levels generated by dopaminergic gain control of the decision-making process; 
and (c) a quasi-tonic signal replacing the average reward in the exponentially discounted setting. These explanations, along with the 
possibility mentioned earlier that release from dopamine axons might be directly occasioned by a form of spillover from cortico-striatal 
activity, are by no means mutually exclusive.
 
The various cases of ramps may be caused by different, or combined, mechanisms. Indeed, the possible explanations that we 
considered mainly in the context of pre-response transients, in which ramp-like signals are observed leading up to completion of an 
instrumental action, were somewhat distinct from the explanation offered for ramping in the spatial reward task, in which the subjects 
are already engaged in acting. Nevertheless, the account of discounted vigour suggested in the latter case should be relevant in all contexts 
where some degree of discounting is probable (i.e., γ < 1), such as in the temporally-extended tasks considered here.
 
TD accounts of pre-response dopamine signals: What TD accounts of pre-response dopamine signals predict depends on the assumptions 
made about the relationship between actor and critic. We considered three possibilities associated with different predictions of how a 
TD error occurring just prior to pressing, and the resulting change in dopamine concentration, should change as response latencies increase: 
remain constant, decrease, or increase.
 
The model in which the critic receives both direct and indirect information, but suffers from timing uncertainty, yielded results most 
consistent with the experimental data reported by Roitman et al. In particular, this case replicated the observation that peak dopamine 
concentration around time of pressing was larger than at time of cue for long latency trials.
 
Dopaminergic gain control: We showed that ramping dopamine signals can be generated by a mechanistic decision-making model in which 
dopamine sets the gain of value-based accumulation. Furthermore, we saw that this direct coupling of dopamine to decision-making could 
generate a negative correlation between the size of TD error and decision time, consistent with the experimental observation that a larger 
phasic response of dopaminergic cells to a start cue is associated with a shorter latency of behavioural response.
 
Discounted vigour: We reconciled an apparent inconsistency between the definitions of TD errors in the cases of average and discounted 
reward via an analysis in which ramp-like signals would be expected to emerge. In particular, we suggested that the quantity 
(1 - γ)⟨V^γ(s_(t+1))⟩ in the discounted reward model plays an equivalent role to the average reward rate ρ in the average reward model. 
Since values often change modestly as a result of the passage of time, this signal is quasi-tonic, and thus a candidate for what would be 
recorded using a technique such as FSCV.
 
Complexities of dopamine release: Phasic, tonic, and quasi-tonic: Whereas the TD account of pre-response transients naturally 
attributes the observed signal to the phasic activity of dopaminergic neurons, the sources of tonic and particularly ‘quasi-tonic' 
dopamine signals are less clear. One long-standing suggestion is that phasic and tonic modes of firing in dopaminergic cells provide 
independent control of phasic and tonic dopamine levels within NAc. Thus, burst firing of dopaminergic neurons is thought to mediate 
a fast, high-amplitude dopamine transient which is spatially-restricted to a region within or proximal to release terminals by dopamine 
reuptake. By contrast, the comparatively slow, irregular, ‘tonic' mode of activity exhibited by a pool of dopaminergic neurons, potentially 
of varying size, is thought to control the more stable, tonic levels of extrasynaptic dopamine.
 
Alternative accounts: We noted above that ramping ostensibly disrupts TD's explanation for dopaminergic release, since it would have, 
oxymoronically, to be a predictable prediction error. Alternative accounts have been suggested according to which prediction errors indeed 
persist.
 
Experimental tests: The most pressing consideration is a set of experiments that can test and refine or reject these various mechanisms, 
and understand how they might work together. Perhaps the most straightforward to test is the last suggestion, since it is unique in its 
dependence on discounting. Given that the rate of this should be sensitive to things like the reliability of the environment, it would be 
interesting to manipulate these factors, determine the extent to which behaviour changes appropriately, and concurrently measure ramping.
 
More generally, key issues surround the relationships between the number of dopamine cells that are active, the phasic and tonic 
activity of those neurons, the spatiotemporal profile of the concentration of dopamine at receptor targets in the accumbens, and the 
action of this dopamine on those receptors.
 
