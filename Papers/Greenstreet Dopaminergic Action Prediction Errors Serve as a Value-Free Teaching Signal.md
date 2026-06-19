

 
Francesca Greenstreet, Hernando Martinez Vergara, Yvonne Johansson, Sthitapranjya Pati, Laura Schwarz, 
Stephen C. Lenzi, Jesse P. Geerts, Matthew Wisdom, Alina Gubanova, Lars B. Rollik, Jasvin Kaur, 
Theodore Moskovitz, Joseph Cohen, Emmett Thompson, Troy W. Margrie, Claudia Clopath, Marcus Stephenson-Jones
 
Published: July 31, 2025
 
---

# Greenstreet Dopaminergic Action Prediction Errors Serve as a Value-Free Teaching Signal
 
Abstract
 
Choice behavior of animals is characterized by two main tendencies: taking actions that led to rewards and repeating past actions. 
Theory suggests that these strategies may be reinforced by different types of dopaminergic teaching signals: reward prediction error 
to reinforce value-based associations and movement-based action prediction errors to reinforce value-free repetitive associations. 
Here we use an auditory discrimination task in mice to show that movement-related dopamine activity in the tail of the striatum 
encodes the hypothesized action prediction error signal. Causal manipulations reveal that this prediction error serves as a value-free 
teaching signal that supports learning by reinforcing repeated associations. Computational modeling and experiments demonstrate 
that action prediction errors alone cannot support reward-guided learning, but when paired with the reward prediction error circuitry, 
they serve to consolidate stable sound–action associations in a value-free manner. Together we show that there are two types of 
dopaminergic prediction errors that work in tandem to support learning, each reinforcing different types of association in different 
striatal areas.
 
---
 
Introduction
 
When animals and humans make choices, they exhibit two key tendencies: pursuing rewarding actions and repeating past actions. 
Dopamine neurons that encode reward prediction error (RPE) provide a critical teaching signal for value-based learning, reinforcing actions 
that lead to reward. Recently it has been proposed that repetitive ‘habitual' choices may instead be updated by a movement-based teaching signal.
 
The hypothesized teaching signal encodes an action prediction error (APE), the difference between the action that is taken and the extent 
to which the action was predicted. Although experimental evidence is lacking, theoretical models suggest that this value-free learning 
system could operate alongside canonical value-based learning in the basal ganglia. These models predict two types of dopaminergic teaching 
signals: RPE, which reinforces reward-driven actions, and APE, which reinforces repeated actions. Notably, dopaminergic neurons 
located in the ventral tegmental area (VTA) and medial parts of the substantia nigra pars compacta (SNc) encode RPE, whereas those in the 
lateral SNc and substantia nigra pars lateralis (SNl), which project to the tail of striatum (TS), prominently respond to movement.
 
Here we investigated whether movement-related dopaminergic activity encodes an APE, providing a value-free teaching signal to reinforce 
repeated state–action associations. To address this, we examine the role of dopamine in the TS as mice learn an auditory discrimination, 
cloud-of-tones (COT) task.
 
---
 
TS Dopamine Facilitates Learning
 
In the COT task, mice initiate trials by nose-poking a central port, triggering auditory stimuli composed of a short train of overlapping 
pure tones (5–40 kHz). They selected a left or right reward port on the basis of whether the stimulus contained primarily low (5–10 kHz) 
or high (20–40 kHz) frequencies. Bilateral inactivation of the TS with muscimol impaired task performance in expert mice. 
Unilateral optogenetic inactivation of either type of striatal projection neurons (SPNs) in the TS also had an opposing and significant effect 
on choices of mice. These results demonstrate that the TS is needed to execute the learned behavior and that both populations of SPNs 
exert opposing contributions to the auditory-guided choices.
 
To test whether the TS was also needed to learn the task, we ablated the TS prior to training using a viral-mediated caspase-based strategy. 
Lesions of the TS caused a deficit in learning, reducing both the learning rate and the maximum performance reached on the task. 
Ablation of the TS-projecting dopamine neurons also recapitulated the general TS lesions effects. TS dopamine-ablated mice had deficits 
in learning without influencing the time taken to move from the centre port to the choice ports, or the time taken between trials. 
Together, these results confirm that both the TS and its dopaminergic innervation are required to facilitate learning and execute the 
auditory discrimination task.
 
---
 
TS Dopamine Release Is Correlated with Movement
 
To understand the role of TS dopamine in the task, we measured its dynamics with dLight1.1, a genetically encoded fluorescent dopamine sensor. 
TS dopamine responses correlated in time with contralateral movements from the centre port, in sharp contrast to the large reward 
responses in ventral striatum (VS). VS responses were best explained by the outcome kernel, capturing the large responses to rewards and 
dips for unrewarded trials. By contrast, TS showed minimal outcome-related dopamine activity. The largest TS dopaminergic response 
was contralateral movement-locked activity, which was also seen when mice made contralateral movements to return from the side ports 
to the centre port. VS movement-related activity was smaller, with no significant difference between contralateral and ipsilateral actions. 
These results show that VS dopamine activity significantly encodes reward outcome, consistent with RPE, whereas the TS dopamine 
activity encodes movement information.
 
To confirm that TS dopamine activity was unrelated to sound, we omitted the cue on some trials and found no significant difference in response. 
To assess task dependence, we recorded TS dopamine activity as mice explored an open arena. As in-task recordings, TS dopamine increased 
during contralateral movements, and its signal scaled with movement amplitude. Turn angle significantly correlated with TS dopamine, 
a correlation that was absent in VS. These results confirm that TS dopamine encodes information about movement.
 
---
 
Evidence for APE
 
Our results indicate that TS dopamine is crucial for learning the auditory frequency discrimination task and encodes information about movement 
rather than reward or sound. Although dopamine RPEs are known to drive cortico-striatal plasticity and learning, the role of movement-related 
dopamine remains unclear. Habit-formation models predict that movement-related dopamine could encode a value-free APE, representing the 
discrepancy between an executed action and its predicted occurrence in a given state.
 
If the movement-related dopaminergic activity in the TS encodes an APE, then dopamine activity in the TS should decrease over time as 
mice learn to predict the action that they will take in response to the sound. To test this prediction, we recorded TS and VS dopamine signals 
over the course of learning and compared these signals to values of RPE and APE from a dual value-based/value-free reinforcement learning model. 
Characteristic of an RPE signal, cue-related dopamine activity in the VS grew over time. In line with encoding an APE signal, the movement-related 
dopamine activity in the TS decreased as mice learned the task.
 
If the movement-related dopaminergic activity in the TS encodes an APE, then changes in the dopamine activity in the TS should also reflect the recent 
history of sound–action pairings that have been experienced. Therefore, the size of the dopamine signal should be smaller when the same 
action is taken in response to the same sound on subsequent trials. In agreement with this, the TS dopamine response was on average 
significantly smaller when mice repeated the same action in response to the same stimulus in the past trial. By contrast, the size of the 
cue response in the VS was larger when the correct sound–action pairing was repeated on subsequent trials, consistent with an RPE signal.
 
The learning-associated changes in movement-related TS dopamine suggests that responses are modulated by how predictably an action follows 
a given state (auditory cue). Therefore, if trained mice were to make a familiar movement in response to an unfamiliar sound, the action would 
not be predicted by the stimulus and consequently the resulting APE should be larger. The novel stimulus decreased the accuracy of mice and 
significantly increased the movement-related TS dopamine signal, as expected from an APE signal. There was no significant TS dopamine 
response to the unfamiliar white noise stimulus when it was played as mice freely explored an open arena.
 
Another hallmark of an APE is that it should be value-free and should not be modulated by either the outcome or the predicted value of an action. 
In agreement with this, the dopamine signal in the TS did not significantly respond to a larger or smaller than predicted reward. By contrast, 
the VS dopamine signal was altered in a manner consistent with RPE. The size of the TS dopamine signal was also not modulated by changes 
in the predicted value of an action.
 
---
 
TS Dopamine Reinforces State-Action Associations
 
To determine whether the TS dopamine signal can act as a teaching signal, we optogenetically stimulated TS dopamine release at different task epochs. 
To mimic the endogenous movement-related TS dopamine signal, we stimulated unilaterally at the centre choice port, in trials where there was 
more sensory evidence for a contralateral choice. Within sessions, stimulation induced a significant contralateral choice bias. This bias 
developed over the course of the session as would be expected if it influenced learning. Optogenetic stimulation did not bias action directly 
as there was no choice bias on individually stimulated trials. Stimulating dopamine release in the VS at the time of choice had no significant 
effect, nor did stimulation of dopamine release in the TS or the VS at the time of choice outcome. In a free choice paradigm TS dopamine 
stimulation did not induce a choice bias but there was a significant bias towards the stimulated port when we stimulated dopamine release in the VS.
 
Other theories suggest that movement-related dopamine facilitates movement initiation or modulates ongoing action. However, closed-loop 
optogenetic stimulation of TS dopamine release in an open-field arena did not influence movement likelihood or alter movement parameters 
when mice did move. These findings suggest TS dopamine activity reinforces state--action associations rather than influencing ongoing action.
 
Since TS dopamine stimulation reinforced state--action associations, we investigated whether endogenous TS dopamine release functioned similarly. 
A logistic regression model predicting choice repetition from the previous trial's dopamine response and current log uncertainty showed 
significant positive correlations for both factors, indicating that mice were more likely to repeat their previous choice when the TS dopamine 
response was larger and sensory uncertainty was higher. By contrast, dopamine reward response size in the VS did not correlate with 
choice bias. These results suggest that movement-related dopamine at choice timing serves as a value-free teaching signal, reinforcing 
stimulus--action associations in the TS so that mice learn to repeat the action that they have taken in the past when they hear the auditory stimulus.
 
If the TS dopamine response encodes an APE on the current trial, it should not only bias mice toward repeating stimulus--action associations but 
also influence trial kinematics, making subsequent actions more similar. To test this, we performed a linear regression between the current 
trial TS dopamine response at time of choice and the Fréchet distance between current and subsequent trial trajectories. The analysis revealed 
a significant negative correlation between TS dopamine response size and Fréchet distance, indicating that a larger TS dopamine response in 
a given trial led to a more similar subsequent trajectory.
 
---
 
A Dual-Controller Model for Learning
 
To examine interactions between striatal regions receiving RPEs or APEs, we built a basal ganglia model that simulated learning with these 
prediction errors. Our dual value-based/value-free model had an actor and critic, updated by RPE, that learned to control actions and generate 
reward predictions. The value-free control system, updated by APE, learned to control action and generate action predictions. A model with only 
a value-free controller was unable to learn the task, as the model continues to repeat what it has done in the past, which is to randomly choose left 
or right in equal proportion. By contrast, a model with only the value-based controller was able to independently learn the task. Notably, a model 
with both controllers learned faster than the value-based system alone. Initially, there was no difference between the learning rate of these 
two models, because early learning is exclusively driven by RPE updates. However, as the dual model begins to consistently choose a particular 
action in response to a particular sound, APE updates started to reinforce specific stimulus--action associations, boosting performance and 
learning rate.
 
Although both the value-based and value-free controllers can control behaviour, our muscimol inactivation data highlights that in well-trained 
mice, behavioural performance relies on the TS (value-free controller). Since the value-free controller cannot learn the task alone, the task 
must first be learned by the value-based controller and then control must transfer over time to the value-free controller. In other dual-controller 
models, this transfer is achieved by an external arbiter. We examined if a similar transfer would occur if the actor's weights decayed over time 
when RPE was low, consistent with striatal literature. When the value-based actor's weights decayed, there was indeed a transfer of control 
in our model. Initial performance was controlled by the value-based controller, and over time, the value-free controller took over.
 
To test whether the TS also slowly takes over control of performance like the value-free controller in our model, we optogenetically inactivated 
either type of TS projection neuron throughout learning. Early in training optogenetic inactivation of neither type of TS-SPNs had a significant 
effect on behaviour. However, as training progressed the inactivation began to have a significant effect on choices and these effects grew larger 
as the mice became experts at the task. These behavioural effects began to consistently increase after the mice were performing above 65%.
 
In the COT task, frequency-specific cortico-striatal plasticity supporting appropriate sound--action associations develops in each hemisphere 
of the TS throughout learning. The same appropriate sound--action associations also form in our value-free controller, showing that APEs 
could be used to drive the frequency-specific changes in cortical striatal plasticity as mice learn the task.
 
---
 
Discussion
 
Here we show that movement-related dopaminergic activity in the TS acts as a teaching signal to reinforce state--action associations. TS dopamine 
activity encodes an APE, the difference between the action taken and the predicted action in a given state. This value-free signal teaches mice to 
repeat past actions. Alone the value-free system (APE → TS) is not able to support reward-guided learning but in conjunction with the canonical 
RPE system it learns to mimic and store the value-guided state--action associations. Together we show that there are two types of dopaminergic 
prediction errors that work in tandem to support learning, each reinforcing different types of association in different areas of the striatum.
 
The identification of dopamine transients prior to movement initiation led to the idea that movement-related dopamine release may act to 
trigger the initiation of movement. However, more recent experiments that used optogenetic stimulation parameters that were calibrated to mimic 
physiological levels of SNc dopamine release did not trigger body movement or lead to an invigoration of ongoing action. The calibrated stimulation 
was however, as we observe, able to act as a teaching signal to drive conditioned place-preference learning and reinforce the use of particular 
behavioural syllables. This suggests that, as we have seen in the TS and consistent with APE, movement-related dopamine activity in other striatal 
areas may also act as a value-free teaching signal rather than as a trigger or modulator of ongoing movement.
 
The caudate tail (CDt)—the primate homologue of the TS—is also innervated by an anatomically distinct population of dopamine neurons that 
are not activated by differences in reward outcome. Rather, as we observe here, these neurons are preferentially active prior to contralateral 
orienting eye movement or when stimuli that predict contralateral orienting responses are presented, as would be expected if they also encode an APE. 
The contralateral nature of the response in primates was interpreted as a unilaterally processed visual response. Our results now show that, 
at least in our task, the TS dopamine responses in mice are related to contralateral movement initiation and not auditory cues.
 
Other laboratories have also shown that dopaminergic activity in the TS is not activated by reward but by threatening stimuli, such as an air puff 
or loud sounds. In addition, large TS dopaminergic transients are active when mice initiate avoidance behaviour, and these responses decrease 
as mice stop performing avoidance responses. These observations led to the proposal that TS dopaminergic activity encodes a threat prediction error (TPE). 
In agreement with these results, we can also observe threat responses in the same region (and same mice) where we observe APE responses. 
However, whereas optogenetic stimulation of dopamine release in the TS was able to reinforce state--action associations, we could not detect 
any aversive effect of this stimulation even when we used higher power than was needed to influence associations in our task. Although our TS 
dopamine stimulation was not able to induce avoidance behaviour, it does appear clear that TS dopamine release is critical for maintaining 
innate aversive associations. How TPE and APE work in the TS to support their different functions at different timescales will need to be 
determined in the future.
 
APE was first conceived as a teaching signal that could update habitual value-free state--action associations. These models offer a parsimonious 
account for hallmarks of habitual behaviour, such as slower adaptation to contingency degradation or reversal and an insensitivity to outcome 
devaluation. This is because the stable associations in these models are driven by APE so are stored in an outcome-independent manner and are 
therefore insensitive to changes in outcome value. Our results show that dopamine neurons that project to the TS encode this type of 
movement-based prediction error. The CDt is where habitual stimulus--action associations are stored. Typically for instrumental learning 
the posterior DMS and the DLS have been shown to be critical for goal-directed and habitual learning respectively. Recently the pDMS has been 
shown to receive reward-related dopaminergic teaching signals. By contrast, movement-related dopamine signals, that could reflect APEs, 
are prominent in the DLS. Taken together, we propose that RPE and APEs may be used as different teaching signals to drive goal-directed 
(value-based) and habitual (value-free) learning in different regions of the striatum.
 
The value-free system updated by APEs learns to repeat the actions that have been taken in the past. This does not seem like an advantageous 
strategy, but this system must exist for a reason. The first thing to note is that although APE is a value-free teaching signal, this does not mean 
that the associations that are formed in the TS will be insensitive to value. Rather, the ‘value-free' associations form through repetition of 
actions that were initially taken in pursuit of value. In this way, the associations that form in the TS can be thought of as storing the long-term 
‘value' of an association even though they were never reinforced in value-based manner per se. Sustaining these associations in an 
outcome-independent manner makes them more stable to short-term fluctuations.
 
Another idea is that stimulus--action associations in the value-free system form a prior for Bayesian inference of action that encodes the 
policy that has worked in the past. During inference this prior would be useful in biasing decisions when there is a high degree of uncertainty 
regarding the value of the action computed in the value-based controller. Finally, recent theoretical work has also predicted that if dopamine 
neurons encoded an APE, then it would allow the basal ganglia to implement off-policy learning algorithms.
