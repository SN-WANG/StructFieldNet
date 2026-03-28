# Trainer for StructFieldNet
# Author: Shengning Wang

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.base_trainer import BaseTrainer


class FieldTrainer(BaseTrainer):
    """Trainer specialized for fixed-mesh stress reconstruction."""

    def __init__(
        self,
        *args: Any,
        gradient_clip_norm: Optional[float] = None,
        use_amp: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_clip_norm = gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0 else None
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        pred = self.model(batch["coords"], batch["design"])
        return self.criterion(pred, batch["stress"])

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        self.model.train(is_training)
        losses = []

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False, dynamic_ncols=True)
            for batch in pbar:
                batch = self._move_batch_to_device(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    loss = self._compute_loss(batch)

                if is_training:
                    if self.use_amp:
                        self.grad_scaler.scale(loss).backward()
                        if self.gradient_clip_norm is not None:
                            self.grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        loss.backward()
                        if self.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                        self.optimizer.step()

                loss_value = float(loss.detach().item())
                losses.append(loss_value)
                pbar.set_postfix({"loss": f"{loss_value:.4e}"})

        return float(np.mean(losses)) if losses else 0.0
