#!/home/johann/dev/machine-learning-playground/.env/bin/python3

import sys 
import os

#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#DIRS = SCRIPT_DIR.split("/")
#SCRIPT_DIR = '/'.join(DIRS[:-1])
#sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append("/home/johann/sonstiges/machine-learning-playground/src")

import time 
from datetime import datetime 
import math 
import tqdm
import argparse 
from yacs.config import CfgNode

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.cuda.amp import GradScaler

from rnn_attn import build_model
from utils import epoch_time, count_params
from dataset import prepare_data, Collater, get_iter 
from config import get_cfg_defaults


def train_epoch(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion, 
    clip: int, 
    epoch: int, 
    writer: SummaryWriter, 
    device: torch.device,
    scaler: GradScaler,
    running_loss_step: int = 10,
):
    model.train()
    epoch_loss = 0 
    running_loss = 0
    c = 0
    for src, tgt in tqdm.tqdm(dataloader):
        src = src.to(device)
        tgt = tgt.to(device) 

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(src, tgt)
            # output => [tgt_len - 1, batch_size, output_dim]
            # tgt => [tgt_len, batch_size]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            # output => [batch_size * (tgt_len - 1)]
            # tgt => [batch_size * (tgt_len - 1), output_dim]

            loss = criterion(output, tgt)

        epoch_loss += loss.item()
        running_loss += loss.item()
                
        scaler.scale(loss).backward()
        #loss.backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        if c % running_loss_step == (running_loss_step - 1):
            writer.add_scalar(
                "train - running loss", 
                running_loss / running_loss_step, 
                epoch * len(dataloader) + c
            )
            running_loss = 0
        c += 1


    return epoch_loss/len(dataloader)


def eval_epoch(
    model, 
    dataloader, 
    criterion,
    device, 
    epoch, 
    writer, 
    running_loss_step=10,
    test=False,
):
    model.eval()
    epoch_loss = 0
    running_loss = 0
    c = 0
    with torch.no_grad():
        for src, tgt in tqdm.tqdm(dataloader):
            src = src.to(device) 
            tgt = tgt.to(device) 
            output = model(src, tgt, 0) # turn off teacher forcing
            # output => [tgt_len, batch_size, output_dim]
            # tgt => [tgt_len, batch_size]
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)
            # output => [batch_size * (tgt_len - 1)]
            # tgt => [batch_size * (tgt_len - 1), output_dim]

            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            running_loss += loss.item()

            
            # Running loss to tensorboard
            if (c % running_loss_step == (running_loss_step - 1)) and test==False:
                writer.add_scalar(
                    "val - running loss", 
                    running_loss / running_loss_step, 
                    epoch * len(dataloader) + c
                )
                running_loss = 0
            c += 1

    return epoch_loss/len(dataloader)


def test(
    model: torch.nn.Module, 
    criterion, 
    cfg: CfgNode, 
    device: torch.device, 
    collater: Collater
) -> float:
    test_iter = get_iter(cfg, split="test")
    test_dataloader = DataLoader(
        test_iter, 
        batch_size=cfg.TEST.BATCH_SIZE, 
        collate_fn=collater,
    )
    test_loss = eval_epoch(model, test_dataloader, criterion, device, None, None, test=True)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
    
    return test_loss


def pred_test_samples(
    model: torch.nn.Module, 
    test_dataloader: DataLoader, 
    cfg: CfgNode, 
    vocab_transform: dict, 
    device: torch.device, 
    num_samples: int = 7, 
    max_length: int = 1000,
) -> str:
    """Create a few translations with the current model using the test dataset to check
    the translation quality.

    Args:
        model (torch.nn.Module): Currently trained Seq2Seq model.
        test_dataloader ([type]): Test Dataloader for the chosen dataset.
        cfg ([type]): Experiment configuration file.
        vocab_transform (dict): Vocabulary transformation dictionary.
        device ([type]): GPU or CPU.
        num_samples (int, optional): Number of test sentences to predict. Defaults to 5.
        max_length (int, optional): Maximum length of the predicted output. Defaults to 1000.

    Returns:
        str: Source sentence, target translation and predicted translation.
    """
    c = 1
    out_str = ""
    model.eval()
    with torch.no_grad():
        for src, tgt in test_dataloader:
            out_str += "===" * 30
            out_str += "\n"

            src_ = src.to(device)
            #tgt_filler = tgt.to(device)
            tgt_filler = torch.zeros((max_length, 1), dtype=torch.long).to(device)
            pred = model(src_, tgt_filler, 0)

            pred = pred.argmax(2).squeeze(1)
            pred = [t.item() for t in pred]
            
            if cfg.DATASET.EOS_IDX in pred:
                eos = pred.index(cfg.DATASET.EOS_IDX)
                pred = pred[1:eos]
            else: 
                pred = pred[1:max_length]

            pred = vocab_transform[cfg.DATASET.TGT_LANGUAGE].lookup_tokens(pred)
            
            src = list(src.squeeze(1).numpy())
            tgt = list(tgt.squeeze(1).numpy())
            
            src = vocab_transform[cfg.DATASET.SRC_LANGUAGE].lookup_tokens(src)[1:-1]
            tgt = vocab_transform[cfg.DATASET.TGT_LANGUAGE].lookup_tokens(tgt)[1:-1]

            out_str += "SRC:\t{}\n".format(" ".join(src))
            out_str += "TGT:\t{}\n".format(" ".join(tgt))
            out_str += "---" * 30
            out_str += "\n"
            out_str += "PRED:\t{}\n".format(" ".join(pred))

            c += 1
            if c >= num_samples:
                return out_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        help="path to config.yaml file which should be merged into standard config",
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory",
        default=None,
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        help="path to checkpoint to resume from",
        default=None,
    )
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.config:
        cfg.merge_from_file(args.config)
    
    _, vocab_transform, text_transform = prepare_data(cfg=cfg)
    
    output_dir = args.out_dir if args.out_dir != None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    exp_dir = os.path.join(output_dir, "{}_{}".format(
        datetime.now().strftime("%Y-%m-%d"), 
        cfg.DATASET.NAME,
    ))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp: 
        fp.write(cfg.dump())
    
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, ".runs"))
    
    if args.resume_from:
        model = build_model(cfg, vocab_transform, device, args.resume_from)
        print("Load model from: {}".format(args.resume_from))
    else: 
        model = build_model(cfg, vocab_transform, device)
    print("Number of parameters: {}".format(count_params(model)))

    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.OPTIMIZER.LR, 
            betas=cfg.get("OPTIMIZER.BETAS") if cfg.get("OPTIMIZER.BETAS") else (0.9, 0.999), 
            eps=cfg.get("OPTIMIZER.EPS") if cfg.get("OPTIMZER.EPS") else 1e-08, 
            weight_decay=cfg.get("OPTIMIZER.WEIGHT_DECAY") if cfg.get("OPTIMIZER.WEIGHT_DECAY") else 0,
        )
    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=cfg.OPTIMIZER.LR,
            momentum=cfg.get("OPTIMIZER.MOMENTUM") if cfg.get("OPTIMIZER.MOMENTUM") else 0,
        )
    elif cfg.OPTIMIZER.NAME == "ADADELTA":
        optimizer = torch.optim.Adadelta(
            model.parameters(), 
            lr=cfg.OPTIMIZER.LR,
            rho=cfg.get("OPTIMIZER.RHO") if cfg.get("OPTIMIZER.RHO") else 0.9,
            eps=cfg.get("OPTIMIZER.EPS") if cfg.get("OPTIMIZER.EPS") else 0.000001,
            weight_decay=cfg.get("OPTIMIZER.WEIGHT_DECAY") if cfg.get("OPTIMIZER.WEIGHT_DECAY") else 0,
        )
    else: 
        print("{} isn't implemented.".format(cfg.OPTIMIZER.NAME))
        print("Please choose a valid optimzer ['SGD', 'Adam'].")

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.DATASET.PAD_IDX)
    collater = Collater(cfg, text_transform)

    if cfg.get("LR_SCHEDULER"):
        if cfg.LR_SCHEDULER.NAME == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer, 
                mode=cfg.get("LR_SCHEDULER.MODE") if cfg.get("LR_SCHEDULER.MODE") else 'min', 
                factor=cfg.get("LR_SCHEDULER.FACTOR") if cfg.get("LR_SCHEDULER.FACTOR") else 0.1, 
                threshold=cfg.get("LR_SCHEDULER.THRESHOLD") if cfg.get("LR_SCHEDULER.THRESHOLD") else 0.00001,
                patience=cfg.get("LR_SCHEDULER.PATIENCE") if cfg.get("LR_SCHEDULER.PATIENCE") else 5, 
            )
        elif cfg.LR_SCHEDULER.NAME == "StepLR":
            scheduler = StepLR(
                optimizer=optimizer,
                step_size=cfg.LR_SCHEDULER.STEP_SIZE,
                gamma=cfg.get("LR_SCHEDULER.GAMMA") if cfg.get("LR_SCHEDULER.GAMMA") else 0.1,
                last_epoch=cfg.get("LR_SCHEDULER.LAST_EPOCH") if cfg.get("LR_SCHEDULER.LAST_EPOCH") else -1,
                verbose=cfg.get("LR_SCHEDULER.VERBOSE") if cfg.get("LR_SCHEDULER.VERBOSE") else False,
            )
        else: 
            print("Learning Rate Scheduler: {} is not implemented.".format(cfg.get("LR_SCHEDULER.NAME", "Name missing")))
            raise NotImplementedError

    scaler = GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(cfg.TRAIN.N_EPOCHS):
        start_time = time.time()
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        train_iter = get_iter(cfg, split="train")
        train_dataloader = DataLoader(
            train_iter, 
            batch_size=cfg.TRAIN.BATCH_SIZE, 
            collate_fn=collater,
            pin_memory=True,
        )
        print("Number of training mini-batches: {}".format(len(train_dataloader))) 
        
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            criterion, 
            cfg.OPTIMIZER.CLIP,
            epoch,
            writer,
            device,
            scaler,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)

        val_iter = get_iter(cfg, split="valid")
        val_dataloader = DataLoader(
            val_iter, 
            batch_size=cfg.VAL.BATCH_SIZE, 
            collate_fn=collater,
        )
        print("Number of val mini-batches: {}".format(len(val_dataloader)))

        val_loss = eval_epoch(
            model, 
            val_dataloader, 
            criterion, 
            device, 
            epoch, 
            writer,
        )

        scheduler.step(val_loss)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # check out a few test example translations with the current model
        test_iter = get_iter(cfg, split="test")
        test_dataloader = DataLoader(test_iter, batch_size=1, collate_fn=collater)
        test_preds = pred_test_samples(model, test_dataloader, cfg, vocab_transform, device)
        print(test_preds)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss 
            torch.save(model.state_dict(), os.path.join(exp_dir, "best.pth"))
        
        print("Epoch: {} | Time: {}m {}s".format(epoch+1, epoch_mins, epoch_secs))
        print("\tTrain loss: {} | Train perplexity: {}".format(train_loss, math.exp(train_loss)))
        print("\tVal loss: {} | Val perplexity: {}".format(val_loss, math.exp(val_loss)))


    print("Best model from epoch {} is loaded for testing.".format(best_epoch + 1))
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best.pth")))
    test(model, criterion, cfg, device, collater)

    writer.close()

if __name__ == "__main__":
    main()