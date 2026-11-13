"""Create independent copies of a crafted update for colluding clients."""

from meta_sg.federated.state import ModelState


def repeat_update(update: ModelState, count: int) -> list[ModelState]:
    return [tuple(part.clone() for part in update) for _ in range(count)]
