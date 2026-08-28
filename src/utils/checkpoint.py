from pathlib import Path
import torch

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, train_loss, valid_loss, shard_index = None):

    checkpoint = {
        "epoch": epoch,
        "shard_index": shard_index,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss
    }

    torch.save(
        checkpoint,
        path
    )

def load_checkpoint(path, model, optimizer = None, scheduler = None, scaler = None):
    
    checkpoint = torch.load(
        path,
        map_location = "cpu"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    if scaler is not None:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    return checkpoint