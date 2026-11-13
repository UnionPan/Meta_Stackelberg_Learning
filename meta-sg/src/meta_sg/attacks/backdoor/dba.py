"""Design: assign one of four stable sub-triggers to each malicious identity.
Data flow: sampled clients -> sub-trigger poisoning -> scaled local updates.
Simultaneous and cyclic schedules reuse the shared backdoor training path.
"""

from meta_sg.attacks.backdoor.training import backdoor_round


def dba_round(*args, **kwargs):
    return backdoor_round(*args, distributed=True, **kwargs)
