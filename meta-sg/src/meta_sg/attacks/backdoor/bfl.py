"""Design: configure a collaborative backdoor using the complete trigger.
Data flow: malicious clients -> full-trigger poisoning -> optionally scaled updates.
Defaults preserve full source-class poisoning without model replacement.
"""

from meta_sg.attacks.backdoor.training import backdoor_round


def bfl_round(*args, source_label=0, poison_fraction=1.0, model_replacement_scale=1.0, **kwargs):
    return backdoor_round(
        *args, distributed=False, source_label=source_label, poison_fraction=poison_fraction,
        model_replacement_scale=model_replacement_scale, **kwargs,
    )
