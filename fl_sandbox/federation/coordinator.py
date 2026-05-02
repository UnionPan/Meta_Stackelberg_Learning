"""Federated coordinator public entrypoint.

The compatibility runner remains the concrete implementation during this
structural migration; new code should import this class.
"""

from fl_sandbox.core.fl_runner import MinimalFLRunner as FederatedCoordinator

__all__ = ["FederatedCoordinator"]
