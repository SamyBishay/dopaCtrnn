# Reframing dopamine: A controlled controller at the limbic-motor interface

Kevin Lloyd, Peter Dayan
 
Published: October 17, 2023
 
---
 
Abstract
 
Pavlovian influences notoriously interfere with operant behaviour. Evidence suggests this interference sometimes coincides with the 
release of the neuromodulator dopamine in the nucleus accumbens. Suppressing such interference is one of the targets of cognitive 
control. Here, using the examples of active avoidance and omission behaviour, we examine the possibility that direct manipulation of 
the dopamine signal is an instrument of control itself. In particular, when instrumental and Pavlovian influences come into conflict, 
dopamine levels might be affected by the controlled deployment of a reframing mechanism that recasts the prospect of possible 
punishment as an opportunity to approach safety, and the prospect of future reward in terms of a possible loss of that reward. 
We operationalize this reframing mechanism and fit the resulting model to rodent behaviour from two paradigmatic experiments 
in which accumbens dopamine release was also measured. We show that in addition to matching animals' behaviour, the model predicts 
dopamine transients that capture some key features of observed dopamine release at the time of discriminative cues, supporting the 
idea that modulation of this neuromodulator is amongst the repertoire of cognitive control strategies.
 
---
 
Author Summary
 
Evolution provides us with behavioural tendencies that are usually adaptive, but sometimes interfere with our goals. The glimpse of a 
cream bun in the bakery window may lure us to actions defeating our dieting aims; the sound of a loud car horn as we cross a street may 
cause us to freeze reflexively when it would be better to hasten out of the way of oncoming traffic. Such 'Pavlovian' influences over 
behaviour, in these examples respectively promoting active approach and behavioural inhibition, involve the action of neuromodulators, 
such as dopamine, in subcortical brain areas. Here, we consider the possibility that one strategy the brain employs to attempt to 
control such occasionally errant processes is to manipulate the neuromodulatory signal itself. We examine experimental results from 
two rodent studies that measured subcortical dopamine release while rats made active responses to evade punishment, or inhibited 
their responses to gain reward. We build a model of the rats' behaviour that includes the possibility of controlling dopamine release, 
and show that the model can capture key patterns in the data. This lends support to the idea that the brain may sometimes exert control 
by manipulating neuromodulation itself.
 
---
 
Introduction
 
Evolution has endowed animals with behavioural tendencies such as approaching and engaging with sources and predictors of food, 
and freezing or withdrawing from sources and predictors of punishment. Such 'Pavlovian' inductive biases provide an effective way to 
obviate the need for learning in situations that are common and occasionally critical, and exert a powerful influence on behaviour. 
However, they can also lead to counterproductive Pavlovian-instrumental conflict—'Pavlovian misbehaviour'—if animals must act 
vigorously to avoid predicted punishment, or withhold actions to gain potential reward. They then need to be suppressed or supplanted, 
in what can be interpreted as a form of cognitive control.
 
One contributor to the Pavlovian-instrumental conflict may be the neuromodulator dopamine (DA), in a clash between its dual roles 
in positive reinforcement and motivational vigour. Evidence from canonical versions of these conflict paradigms suggests that DA 
in the core of the nucleus accumbens (NAc), at least when performance is successful, follows its motivational, rather than its 
reinforcement, role, with enhanced DA concentrations being observed during active avoidance and suppressed DA concentrations when 
behavioural suppression is required to gain reward.
 
Partly inspired by the two-factor theory of active avoidance, Boureau and Dayan suggested that such DA dynamics might be 
conceptualized as arising from a shift of the origin in a valence-action space. In the active avoidance case, a shift to a negative 
valence corresponding to expected punishment means that a neutral outcome (avoidance) appears positive; enhanced release of DA 
associated with the prospect of safety could then play a role in energizing the necessary active avoidance response. Conversely, 
when behavioural suppression is required to gain reward, a shift of the origin to the associated positive valence means that a 
neutral outcome (no reward) appears negative; suppression of DA release associated with the prospect of this loss may promote 
behavioural inhibition.
 
Subsequent work elaborated on this suggestion in relation to active avoidance, but did not provide a process account of the reframing 
required to turn a situation that is, at best, neutral into one that appears positive. Furthermore, an account of the opposite reframing—to 
turn a situation that is, at worst, neutral into one that appears negative—has been lacking.
 
In the current work, we address these shortcomings via a modelling framework that characterizes the putative reframing operations, 
and associated effects on DA signalling, as resulting from internal cognitive control actions. As modelling targets, we focus on two 
recent experimental studies in rodents, both involving measurement of NAc DA release: a study by Gentry et al. involving active 
avoidance, and a study by Syed et al. involving behavioural suppression to obtain reward. After briefly outlining the main idea 
of the model, we describe these experiments and their principal findings, and show how our model may account for certain critical 
features of the observed dynamics of DA release associated with cue and control (leaving out the outcome). We also consider the 
important issue of how the putative reframing mechanism could remain stable given the plasticity typically associated with DA release.
 
---
 
Results and Discussion
 
Model
 
In instrumental or operant conditioning, animals learn to make responses given particular sensory stimuli. These responses are based, 
at least initially, on the outcomes that are contingent on those responses—animals typically prefer responses that lead to greater 
rewards or lesser punishments, and avoid responses that lead to greater punishments or lesser rewards. Conversely, in Pavlovian or 
classical conditioning, animals learn the predictive relationship between sensory stimuli and affectively important outcomes, and then 
those stimuli come to elicit a set of automatic, conditioned, responses irrespective of the contingency between those responses and 
the outcomes. Pavlovian responses include approach and engagement for appetitive predictors, and withdrawal and inhibition for aversive 
predictors.
 
The involuntary nature of conditioned responses implies that difficulties can arise in situations such as active avoidance (in which 
animals avoid a punishment only if they act in a short time after a predictive cue) and omission schedules (in which animals will 
receive a reward following a cue only if they do not act). This is because the instrumental requirement (acting or withholding, 
respectively) is directly opposed by the Pavlovian conditioned response (freezing to the punishment predictor, or engaging to the 
reward predictor).
 
In a simple case in which stimuli are potentially associated with the emission ('Go') or withholding ('NoGo') of active responses, 
this was operationalized by state-action values (Q-values), which respectively capture the long-run benefit of responding or not, 
being additively corrupted by a quantity proportional to the predicted affective value of the state.
 
First consider the case of the omission schedule. In this case, provided that performance is at least somewhat successful, the state 
will predict some reward, and so the predicted affective value will be positive. Then, the Pavlovian factor will boost the propensity to 
act/'Go'—and so may in some cases interfere with the very behaviour (suppression of action) that led to success in the first place.
 
Noting that the TD error typically associated with a state is just the value of that state if there is no extrinsic reward at that moment 
and no precise prior expectation, then one contributor to this Pavlovian effect might be the incentive salience-associated release of DA; 
this would energize action, perhaps via its action on direct and indirect pathways in the striatum.
 
Conversely, in the case of active avoidance, if the animal is at least partially incompetent and so receives some shocks, the predicted 
affective value will be negative. In this case, the Pavlovian factor will suppress the propensity to act (i.e., favours 'NoGo'), e.g., by 
promoting freezing or withdrawal.
 
Our central conceit is that the coupling of DA with action provides both the opportunity and need for a form of cognitive control in 
which DA release is manipulated by a reframing of values. This generalizes the suggestion that the origin of the valence axis of the 
affective circumplex can be adjusted. Such control might fully determine a particular trajectory for DA release; however, we explore 
a more limited construct in which it induces new, counterfactual states associated with default expectations, with an effect on 
dopamine concentrations associated with the discriminative cues in these experiments.
 
---
 
Active Avoidance
 
A particular example involving active avoidance is provided by Gentry et al., who used fast-scan cyclic voltammetry (FSCV) to 
examine DA release in the core of the NAc during performance of a mixed-valence task. In one class of trials, rats heard a tone telling 
them that they had to press a lever within a 10 s response window to avoid a shock. Half the animals often struggled to respond actively 
in time, and showed higher rates of freezing—a typical example of Pavlovian-instrumental conflict. However, across the population, 
on trials when they did press, cue-elicited DA release was similar just after shock or reward cues, notably being stimulated rather than 
suppressed.
 
For convenience, we write s_pre for the state before the tone that indicates trial type, with V(s_pre) ≈ 0 (from long, subjectively uncertain, 
inter-trial intervals), and s_χ, for the different states entered depending on the tone. Then, for shock trials and imperfect 
avoidance, V(s_shk) < 0. Thus, the TD error would be:
 
δ_shk = V(s_shk) - V(s_pre) < 0.
 
Our assumption is that for shock trials, the deployment of cognitive control instills a counterfactual state s_fail that substitutes 
for s_pre, with V(s_fail) ≪ 0 quantifying the full explicit cost of the shock. Then,
 
δ_shk = V(s_shk) - V(s_fail) > 0,
 
promoting Pavlovian action. This relocation of the origin of the affective circumplex to the negative affective value associated with 
presumed failure and thus the shock harmonizes Pavlovian and instrumental control in the service of active responding—and would 
explain the positive DA transient for successful avoidance.
 
To test this, we fitted a model that incorporates Pavlovian influences, via an effective value, and a probability of employing control 
to the animals' behaviour. Averaging over the resulting mixture of differential TD errors for no-control vs. control shock trials 
then indeed implies a net-positive DA signal on trials where animals successfully avoid shock, assuming that the TD signal is conveyed 
by DA transients. The predicted suppression of DA release for poor avoiders on failed avoidance trials would be consistent with such 
failures of control, and with observations that enhanced or suppressed DA release given a warning cue predicts successful or failed 
active avoidance.
 
We briefly note some discrepancies between the data and model behaviour. Firstly, the relative magnitude of DA release on poor-avoidance 
neutral trials in the model is lower than is observed in the data. One possibility is that there is partial confounding of cues, leading 
to a degree of generalization in the DA response. Secondly, the model appears to predict greater cue-evoked DA release (on average) 
during successful avoidance for poor avoidance sessions, something which is not evident in the data. However, the plots in the data depend 
on splitting sessions according to a particular operationalization of good vs. poor avoidance, which may have obscured relationships 
that would be apparent by instead considering avoidance on a continuum.
 
Finally, we only set out to model the DA transients associated with the cue. However, the data suggests that even within this limited 
time window, cue-evoked DA release on press trials is prolonged for the reward cue relative to the shock cue. One possibility is that 
this arises from incomplete ‘predicting away' of the rewarding outcome on those trials.
 
---
 
Go/No-Go
 
While Pavlovian influences may take the form of counterproductive behavioural inhibition in the case of active avoidance, they may 
also appear as unhelpful behavioural activation when suppression would be preferable. Syed et al. used a Go/No-Go task in which one 
of four auditory cues indicated whether rats had to leave a nose-poke ('Go') and execute an active response (press a lever twice) or 
stay in the nose-poke until the tone turned off ('No-Go') in order to get a small or large reward. Animals were reliably successful on 
Go large-reward (GL) trials, but less so on No-Go large-reward (NGL) trials. We attribute this to Pavlovian misbehaviour caused by 
the prospect of a large reward, consistent with the faster ultimate reaction time on successful large-reward trials in both Go and No-Go 
conditions. Mirroring the case of active avoidance, on successful NGL trials, after a minor peak, there was a suppression of DA below 
baseline during the No-Go period (followed by a rise at movement initiation), despite the prospect of large reward; by contrast, on 
successful GL trials, there was a marked early increase.
 
Again, we write s_pre for the state before the disambiguating cue, with V(s_pre) ≈ 0, and s_χ for the states inspired by the respective 
cues. Partial success on NGL trials, and thus large rewards, would make V(s_ngl) > 0, with:
 
δ_ngl = V(s_ngl) - V(s_pre) > 0,
 
promoting Pavlovian action, No-Go failure, and ultimately a decrease in V(s_ngl).
 
In this case, we consider cognitive control as instilling a counterfactual state s_succ with V(s_succ) ≫ 0 quantifying the full value of 
succeeding in the No-Go requirement. Then,
 
δ_ngl = V(s_ngl) - V(s_succ) < 0,
 
again harmonizing Pavlovian and instrumental control, this time by facilitating inaction. This amounts to moving the origin of the 
affective circumplex to the positive value associated with presumed NGL success, switching the sign of the TD error and leading to 
suppression of DA release.
 
---
 
Stability of Reframing
 
An important remaining problem with the proposed reframing is the apparent absence of learning. For instance, if the DA signal is 
positive, why does normal plasticity, associated with conventional TD learning, not zero out this egregious prediction error?
 
One possibility is that downstream systems might be informed directly about the counterfactual status of the reframing, and so avoid 
untoward plasticity. One could only speculate as to how this information could flow and take effect.
 
A second possibility is that cortico-striatal plasticity is confined to precise temporal windows, occasioned for instance by the activity 
of tonically active cholinergic neurons. This window could be explicitly closed as part of the operation of control and so avoid the 
undesired plasticity.
 
Third, there might instead be an active mechanism associated with opponency. That is, rather than the value update being proportional 
to the TD error, it could be proportional to the difference between the TD error and an opponent prediction error. This would render 
reframing stable, as the value update would be zero when the TD error equals the opponent prediction error.
 
This last perspective elucidates other cases with apparently non-zero asymptotic DA. Thus, the evidence from tasks demanding 
substantial work from subjects is that DA release does not inversely covary with demands on vigour, but that compromising DA 
(e.g., by selective lesions) compromises the willingness of subjects to overcome substantial effort costs in their active responding. If 
we imagine that those effort costs are conveyed by opponent terms, then the net influence on action in the striatum would depend on the 
difference between the TD error and the opponent prediction error, which would evidently be compromised by DA deficits.
 
The opponent might also help resolve a tension in our model of active avoidance between the apparent consistency of good-avoiders' 
behaviour with negligible Pavlovian influence and the putative origin of positive DA on shock trials in the deployment of control.
 
Key areas for future work include modelling the cost, learning, and anatomical realization of cognitive control, along with the 
likely mesocortical DA influence over its prefrontal operation; addressing the ultimate habituation of the relevant action and 
obviation of cognitive control; encompassing the known spatial and temporal heterogeneity in DA release; capturing the fuller 
temporal extent of the DA signal rather than just the cue-associated response; explaining the effects of pharmacological manipulation 
in the Go/No-Go task; and incorporating the dorsal striatum, with its suggested focus on the instrumental components of control, 
and its own dopaminergically-impacted bias in favour of action.
 
In sum, we have suggested a neurocomputational architecture in which simple rules coupling action and valence are subject to a 
form of cognitive control whose mode of action exploits this very coupling.
 
---
 
Methods
 
General Model
 
Both tasks are modelled in essentially the same way. First, an 'internal' decision is made when a cue arrives about whether to apply 
self-control or not. There then follows an 'external' decision about the physical action. It is ultimately the physical action that 
determines success or failure for the current trial. Following the inter-trial interval (ITI), and any additional time penalty for 
failure, the next trial begins.
 
The recurrent nature of the tasks, and the fact that faster responses on the current trial can generally increase the rate of rewards 
(or punishments) by hastening opportunities to earn future outcomes in subsequent trials, mean that it is natural to employ an 
average-reward framework for the analysis.
 
We denote trial type by χ. For simplicity, we assume that the initial, internal decision is made only on the trial types of most interest, 
i.e., on shock trials in the mixed-valence task, and on No-Go large-reward (NGL) trials in the Go/No-Go task. Thus, we assume that 
the default choice is always no control, only possibly deviating on these particular trial types.
 
The choice of whether to apply self-control or not is modelled in a very simple way. For shock and NGL trials, we simply assume that 
there is a fixed probability of applying self-control, with this probability being fit to summary measures of the data.
 
As described in the main text, the importance of this internal choice is its effect on the cue-elicited temporal difference (TD) error. 
For no control, we assume that the state is in essence an extension of the cue state, and so there is no change to the TD error 
elicited by cue onset. By contrast, for control, we assume that this is transformed by the baseline/control signal that implements 
the putative reframing.
 
In describing the models in greater detail, we make use of the following common notation:
- R_χ(s): immediate expected utility in current state s and trial type χ.
- T_χ(s): expected time until the next state from current state s and trial type χ.
 
Mixed-Valence Task
 
In this case, the trial types are reward, neutral, and shock, and we assume that the animal’s external choice is between press and 
other, where other can be thought of as some alternative activity that the animal may choose to engage in and which may itself be 
rewarding, but will mean that the animal fails to press on the current trial.
 
In the experiment, there was a 5 s interval between cue onset and the insertion of the response lever; in the model, for simplicity, we 
assume that the external choice is made at the time of the cue, and that implementation of that choice only begins at lever insertion.
 
We assume that successfully pressing on a reward trial delivers positive utility, while failing to press on a shock trial leads to 
delivery of a punishing shock with disutility. The value of pressing with latency τ is then defined by a combination of vigour 
costs, the value of the successor state, and the average reward rate.
 
The distribution of pressing latencies is assumed to be influenced by both instrumental and Pavlovian factors, with positive TD errors 
tending to speed up responding and negative TD errors tending to slow responding down.
 
Go/No-Go Task
 
In this case, the task demands a slightly different choice structure. As in the mixed-valence case, there is an initial internal choice 
about self-control. However, we then assume that the next immediate choice facing the animal is when to leave the nose-poke. The trial 
types are Go small-reward (GS), Go large-reward (GL), No-Go small-reward (NGS), and No-Go large-reward (NGL). The value of leaving 
at time τ for Go trials is assumed to be influenced by costs associated with maintaining fixation and the vigour of leaving.
 
On Go trials, an additional choice is required on exiting the nose-poke: whether to subsequently press the lever or perform some 
other activity. The value of pressing at different latencies is defined similarly to the mixed-valence task, with the choice between 
press and other being influenced by instrumental factors.
 
Dopamine
 
We assume that the Pavlovian influence operates via the TD prediction error. While a large body of evidence supports the idea that 
this quantity is signalled by the activity of midbrain DA neurons, there is also evidence that DA activity may not be its sole 
representational substrate.
 
We emphasize that we seek to model only DA transients associated with the task cues that differentiate between trial types, and not 
the later release which is contemporaneous with movements and/or delivery of outcomes. We have therefore focused on particular 
epochs proximal to cue onset in each experiment.
 
We follow previous work in assuming that DA signals only part of the full TD error, and principally signals transitions that are better 
than expected. In particular, we assume that the dopaminergic component is given by a weighted combination of positive and negative 
TD errors, with a greater weight on positive errors.
 
We additionally consider how the TD error, putatively represented by the firing of DA neurons, is reflected in changes in DA release 
measured in the accumbens. Here, we assume that this term is convolved with an alpha function, so that changes in DA concentration 
relative to baseline are given by the convolution of the dopaminergic TD error with this alpha function.
 
To allow for the possibility that it may take some non-trivial amount of time for control/reframing to be applied, for trials on which 
this is the case, we assume that we initially have the TD error at cue onset, but that this is followed by the transformed TD error 
after a short delay.
 
To assess model-derived DA responses on success vs. failed trials separately, we compute the posterior probability of having 
employed control given success or failure, and use this to derive the average DA signal on successful and failed trials.
 
Model-Fitting
 
For a given set of parameters, the self-consistent set of differential state values and associated behaviour can be found using value 
iteration. For each task, we fitted parameters to minimize the difference between animal and model behaviour. The error function 
includes terms for differences in success rates, reaction times, and DA signals, with weights determining the relative importance 
of these terms.
 
Data Analysis
 
The data from Syed et al. were downloaded and analyzed. Following the original study, data were smoothed using a moving window 
and baselined by subtracting the average signal during a pre-cue period. As a basic test of the hypothesis that the cue-evoked DA 
response would be greater on failed No-Go large-reward (NGL) trials than on successful NGL trials, we integrated the DA signal over 
a 1 s period immediately following cue-onset, averaged this measure for each session, and applied a statistical test.
