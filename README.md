# recsys-simulation


## Patient
Factors that effect patient's performance
- stats
- dms
- novelty of the protocol
- mood/motivation

`TODO`: add patient profile

`TODO`: add a way for stats to change (how do we model changes in underlying skill levels?)

`TODO`: model how patient mood etc. affects performance

`TODO`: model the probability of the user finishing an assigned protocol


### Notes -
- Percieved competence and challenge lead to best learning outcomes.
- bimodal learning curves for dms (fast while learning protocol, slow when improvements depend on improving underlying stats)


## RecSys
Factors that should affect recommendations -
- Patient protocol fit (PPF)
- Patient history
    - Adherence (completed_duration/prescribed_duration)-per protocol
    - Recent performance
    - Recent dm changes
    - novelty of the protocol
    - associations of protocols with subjective assessment
- current mood/motivation (from subjective assessment)
    - if the patient is tired, unmotivated etc. we can give them protocols with high previous mood associations, and high previous dm changes (relatively easy).

$ score = a*PPF + b*\delta dm + c*association_{subjective} + d*adherence $

Turn the distribution closer to softmax the more tired, unmotivated the person is.

$dist_{modified} = \frac{(15-score_{subjective})*softmax(dist_{original}) + dist_{original}}{15}$

## Protocols
`TODO`: add protocol profile

## Sessions