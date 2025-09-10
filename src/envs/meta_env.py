
from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv

class MetaEnv(ParallelEnv, ABC):
    """
    An abstract base class for meta-game environments in PettingZoo.

    A meta-game is defined as a game where one or more agents have a "type" that can be sampled from a set of possible types. This type can affect the agent's behavior, rewards, or other aspects of the environment.
    """

    @property
    @abstractmethod
    def opponent_types(self):
        """A list of possible types for the opponent(s)."""
        pass

    @abstractmethod
    def reset(self, seed=None, options=None):
        """
        Resets the environment to a starting state.

        In a meta-environment, this method is also responsible for sampling a new type for the opponent(s) for the upcoming episode.
        """
        pass

    @abstractmethod
    def step(self, actions):
        """
        Takes a dictionary of actions from the agents and returns the new state of the environment.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.
        """
        pass

    @abstractmethod
    def _get_rewards(self, actions):
        """
        Calculates the rewards for the agents based on their actions and the current state of the environment.

        In a meta-environment, this method will often depend on the current type of the opponent(s).
        """
        pass
