"""FP16 GradScaler wrapper + loss-scale history for Inv-6."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch


class Fp16State:
    """Wraps ``torch.amp.GradScaler``; records scale-factor history."""

    def __init__(self, enabled: bool = True, maxlen: int = 1024) -> None:
        self.enabled = enabled
        grad_scaler_cls: Any = torch.amp.GradScaler  # pyright: ignore[reportPrivateImportUsage]
        self.scaler = grad_scaler_cls("cuda", enabled=enabled)
        self.scale_history: deque[float] = deque(maxlen=maxlen)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self.scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
            self.scale_history.append(float(self.scaler.get_scale()))
        else:
            optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def current_scale(self) -> float:
        return float(self.scaler.get_scale()) if self.enabled else 1.0
