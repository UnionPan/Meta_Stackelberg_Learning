"""Design: reward functions consume trusted-root feedback, never attack/test labels.
Data flow: before/after root metrics and retained risk -> configurable scalar reward.
Register another function in REWARDS to change the objective independently of observations.
"""


def root_loss_delta(feedback, *, scale=1.0):
    before, after = feedback['before']['loss'], feedback['after']['loss']
    return scale * (before - after) / (1 + before)


def utility_delta(feedback, *, accuracy_weight=1.0, loss_weight=1.0, risk_weight=1.0, scale=1.0):
    gain = root_loss_delta(feedback)
    accuracy = feedback['after']['accuracy'] - feedback['before']['accuracy']
    return scale * (accuracy_weight * accuracy + loss_weight * gain - risk_weight * feedback['retained_risk'])


def absolute_cost(feedback, *, loss_weight=1.0, risk_weight=1.0, scale=1.0):
    return -scale * (loss_weight * feedback['after']['loss'] + risk_weight * feedback['retained_risk'])


REWARDS = {'root_loss_delta': root_loss_delta, 'utility_delta': utility_delta, 'absolute_cost': absolute_cost}
