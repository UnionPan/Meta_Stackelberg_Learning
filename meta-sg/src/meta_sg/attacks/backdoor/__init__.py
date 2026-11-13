"""Export targeted backdoor training, DBA/BFL injection, and ASR evaluation."""

from meta_sg.attacks.backdoor.bfl import bfl_round
from meta_sg.attacks.backdoor.dba import dba_round
from meta_sg.attacks.backdoor.evaluation import evaluate_asr
from meta_sg.attacks.backdoor.training import poison_batch, train_backdoor_update


__all__ = ['bfl_round', 'dba_round', 'evaluate_asr', 'poison_batch', 'train_backdoor_update']
