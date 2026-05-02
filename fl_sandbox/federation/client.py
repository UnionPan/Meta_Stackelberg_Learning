"""Client-side local training executor."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class ClientTrainer:
    """Run local client optimization against one dataloader."""

    criterion: object
    device: object
    lr: float
    local_epochs: int
    use_amp: bool = False

    def train(self, model, loader, optimizer=None, scaler=None) -> tuple[float, float]:
        if torch is None:
            raise RuntimeError("torch is required for client training")
        model.train()
        runtime_optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=self.lr)
        runtime_scaler = scaler or torch.amp.GradScaler("cuda", enabled=self.use_amp)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for _ in range(self.local_epochs):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=getattr(self.device, "type", None) == "cuda")
                labels = labels.to(self.device, non_blocking=getattr(self.device, "type", None) == "cuda")
                if getattr(self.device, "type", None) == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                runtime_optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                        logits = model(images)
                        loss = self.criterion(logits, labels)
                    runtime_scaler.scale(loss).backward()
                    runtime_scaler.step(runtime_optimizer)
                    runtime_scaler.update()
                else:
                    logits = model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    runtime_optimizer.step()
                batch_size = labels.size(0)
                total_loss += loss.detach().item() * batch_size
                total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            return 0.0, 0.0
        return total_loss / total_examples, total_correct / total_examples
