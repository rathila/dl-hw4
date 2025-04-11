import argparse
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from .models import  load_model, save_model
from .datasets import road_dataset
from .metrics import PlannerMetric

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 100,
    lr: float = .005,
    batch_size: int = 128,
    seed: int = 2024,
    transform_pipeline:str ="transform_pipeline",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    metric = PlannerMetric()
    writer = SummaryWriter(log_dir="logs")
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = load_model(model_name)
    model = model.to(device)
    model.train()
    train_data = road_dataset.load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2, transform_pipeline=transform_pipeline)
    val_data = road_dataset.load_data("drive_data/val", shuffle=False)


    loss_func = torch.nn.CrossEntropyLoss()

    # optimizer = ...
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # training loop
    global_step = 0
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_accuracy = []       
        
        model.train()

        for batch in train_data:
          track_left = batch['track_left']  # (B, 10, 2)
          track_right = batch['track_right']
          waypoints = batch['waypoints']
          mask = batch['waypoints_mask']

               
          # #logger.add_scalar('train_loss', loss_value.item(), global_step=global_step)
          # train_accuracy.append((pred_labels.argmax(dim=1) == label).float().mean().item())
          # writer.add_scalar("train/loss", loss_value.item(),global_step)
          # global_step += 1

        writer.add_scalar("train/accuracy", np.mean(train_accuracy),epoch)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            val_accuracy = []
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred_test_label = model(img)
                # TODO: compute validation accuracy
                val_accuracy.append((pred_test_label.argmax(dim=1) == label).float().mean().item())

        # log average train and val accuracy to tensorboard
            writer.add_scalar("val/accuracy", np.mean(val_accuracy),epoch)
        epoch_train_acc = torch.as_tensor(train_accuracy).mean()
        epoch_val_acc = torch.as_tensor(val_accuracy).mean()
         # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    #torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    #print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="transform_pipeline")

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
