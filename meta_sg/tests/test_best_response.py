from meta_sg.learning.best_response import AttackerBestResponse
from meta_sg.strategies.types import ATTACK_DOMAIN


class CountingAgent:
    def __init__(self):
        self.update_calls = 0

    def update(self, buffer):
        self.update_calls += 1
        return {"critic_loss": 1.0, "q_mean": 2.0, "actor_loss": 3.0}


def test_attacker_best_response_zero_steps_skips_update():
    agent = CountingAgent()
    best_response = AttackerBestResponse(
        attacker_agents={"rl": agent},
        attacker_buffers={"rl": object()},
        n_a=0,
    )

    losses = best_response.update(ATTACK_DOMAIN["rl"])

    assert losses == {}
    assert agent.update_calls == 0
