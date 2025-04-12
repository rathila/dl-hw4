import argparse
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from .models import  load_model, save_model
from .datasets import road_dataset
from .metrics import PlannerMetric

def masked_l1_loss(pred, target, mask):
    # pred, target: (B, n_waypoints, 2)
    # mask: (B, n_waypoints)   
    error = (pred - target).abs()   
    error_masked = error * mask[...,None]
    return error_masked.sum() / mask.sum().clamp(min=1)
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


    loss_func = masked_l1_loss

    # optimizer = ...
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # training loop
    global_step = 0
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_accuracy = []       
        running_loss = 0.0
        model.train()

        for batch in train_data:
          track_left = batch['track_left'].to(device)  # (B, 10, 2)
          track_right = batch['track_right'].to(device)
          waypoints = batch['waypoints'].to(device)
          mask = batch['waypoints_mask'].to(device)
          pred = model(track_left, track_right)
          
          loss = loss_func(pred, waypoints, mask) 
       
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()  
          metric.add(pred, waypoints, mask)
          running_loss += loss.item() 
          # #logger.add_scalar('train_loss', loss_value.item(), global_step=global_step)
          # train_accuracy.append((pred_labels.argmax(dim=1) == label).float().mean().item())
          # writer.add_scalar("train/loss", loss_value.item(),global_step)
          # global_step += 1

        results = metric.compute() 
        avg_train_loss = running_loss / len(train_data)      
        print(f"[Epoch {epoch}] Loss: {avg_train_loss:.4f} ")
        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            metric.reset()
            val_loss = 0.0
            for batch in val_data:
              track_left = batch['track_left'].to(device)  # (B, 10, 2)
              track_right = batch['track_right'].to(device)
              waypoints = batch['waypoints'].to(device)
              mask = batch['waypoints_mask'].to(device)
              pred = model(track_left, track_right)
              loss = loss_func(pred, waypoints, mask)
              val_loss += loss.item()  
              metric.add(pred, waypoints, mask)              

        val_results = metric.compute() 
        avg_val_loss = val_loss / len(val_data) 
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("LateralError/Val", val_results["lateral_error"], epoch)
        writer.add_scalar("LongitudinalError/Val", val_results["longitudinal_error"], epoch)     
         # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
             print(f"[TLong Err: {results['longitudinal_error']:.4f} | Lat Err: {results['lateral_error']:.4f}-[VLong Err: {val_results['longitudinal_error']:.4f} | Lat Err: {val_results['lateral_error']:.4f}")
              # === Logging ===
        
          
    # save and overwrite the model in the root directory for grading
    save_model(model)
    writer.close()
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
