import gc
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from utils.checkpoint import save_checkpoint
from utils.dataset import ShardDataset

class Trainer:

    def __init__(self, model, optimizer, criterion, scheduler, device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        self.writer = SummaryWriter(cfg.LOG_DIR)
        self.scaler = torch.amp.GradScaler()

        self.global_step = 0
        self.current_epoch = 0

    def compute_loss(self, outputs, targets):

        outputs = outputs[:, 1:]
        targets = targets[:, 1:]

        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        return self.criterion(
            outputs,
            targets
        )

    def train_step(self, batch):

        source, target = batch

        source = source.to(self.device, non_blocking = True)
        target = target.to(self.device, non_blocking = True)

        with torch.amp.autocast(device_type = self.device.type):

            outputs = self.model(
                source,
                target
            )

            loss = self.compute_loss(
                outputs,
                target
            )

        loss = loss / cfg.ACCUMULATION_STEPS
        
        self.scaler.scale(
            loss
        ).backward()

        return loss.item() * cfg.ACCUMULATION_STEPS

    def validation_step(self, batch):

        source, target = batch

        source = source.to(self.device, non_blocking = True)
        target = target.to(self.device, non_blocking = True)

        with torch.no_grad():

            with torch.amp.autocast(device_type = self.device.type):

                output = self.model(
                    source,
                    target,
                    teacher_forcing_ratio = 0.0
                )

                loss = self.compute_loss(
                    output,
                    target
                )

        return loss.item()

    def train_loader(self, loader, shard_index = None):

        self.model.train()

        total_loss = 0.0

        pbar = tqdm(
            loader,
            leave = False
        )

        self.optimizer.zero_grad(set_to_none = True)

        for step, batch in enumerate(pbar, start = 1):

            loss = self.train_step(batch)
            total_loss += loss
            self.global_step += 1

            if (step % cfg.ACCUMULATION_STEPS == 0 or step == len(loader)):

                self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.GRAD_CLIP
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(set_to_none = True)

            pbar.set_postfix(
                loss = f"{loss:.4f}"
            )

            if self.global_step % cfg.CHECKPOINT_BATCH == 0:

                save_checkpoint(
                    cfg.CHECKPOINT_DIR / "batch_latest.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.current_epoch,
                    total_loss / step,
                    None,
                    shard_index = shard_index
                )
            
        return total_loss, len(loader)

    def validate_loader(self, loader):

        self.model.eval()

        total_loss = 0.0

        pbar = tqdm(
            loader,
            leave = False
        )

        for batch in pbar:
            
            loss = self.validation_step(batch)
            total_loss += loss

            pbar.set_postfix(
                loss = f"{loss:.4f}"
            )
        
        return total_loss, len(loader)

    def fit(
        self,
        train_shards,
        valid_shards,
        batch_size,
        collate_fn,
        epochs,
        num_workers,
        pin_memory,
        persistent_workers,
        start_epoch = 0,
        start_shard = 0
    ):
        
        best_loss = float("inf")
        
        for epoch in range(start_epoch, epochs):

            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1} / {epochs}")

            train_total_loss = 0.0
            train_batches = 0

            for shard_index, shard in enumerate(train_shards):

                if shard_index < start_shard:
                    print(f"Skipping (already trained): {shard.name}")
                    continue

                print(f"Training: {shard.name}")

                dataset = ShardDataset(
                    shard
                )

                loader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle = True,
                    collate_fn = collate_fn,
                    num_workers = num_workers,
                    pin_memory = pin_memory,
                    persistent_workers = (
                        persistent_workers if num_workers > 0 else False
                    )
                )

                total_loss, batches = self.train_loader(
                    loader,
                    shard_index
                )

                train_total_loss += total_loss
                train_batches += batches

                del loader
                del dataset

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            start_shard = 0

            train_loss = (
                train_total_loss / train_batches if train_batches > 0 else 0.0
            )

            valid_total_loss = 0.0
            valid_batches = 0

            for shard in valid_shards:

                print(f"Validation: {shard.name}")

                dataset = ShardDataset(
                    shard
                )

                loader = DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle = False,
                    collate_fn = collate_fn,
                    num_workers = num_workers,
                    pin_memory = pin_memory,
                    persistent_workers = (
                        persistent_workers if num_workers > 0 else False
                    )
                )

                total_loss, batches = self.validate_loader(
                    loader
                )

                valid_total_loss += total_loss
                valid_batches += batches

                del loader
                del dataset

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            valid_loss = (
                valid_total_loss /valid_batches
            )

            if self.scheduler:
                self.scheduler.step()

            self.writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch
            )

            self.writer.add_scalar(
                "Loss/Valid",
                valid_loss,
                epoch
            )

            print(f"Train: {train_loss:.4f}")
            print(f"Valid: {valid_loss:.4f}")

            save_checkpoint(
                cfg.CHECKPOINT_DIR / "latest.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                train_loss,
                valid_loss
            )

            if valid_loss < best_loss:

                best_loss = valid_loss
                save_checkpoint(
                    cfg.CHECKPOINT_DIR / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    train_loss,
                    valid_loss
                )