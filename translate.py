import argparse 
import torch 
from config import get_cfg_defaults
from rnn_attn import build_model
from dataset import prepare_data
import matplotlib.pyplot as plt 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", 
        type=str, 
        help="Path to your model .pth file", 
    )
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to config.yaml, if not specified the default config.py is used.", 
        default="config.py",
    )
    parser.add_argument(
        "sentence", 
        type=str, 
        help="Sentence to translate", 
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        help="Maximum output sentence length.", 
        default=100,
    )
    parser.add_argument(
        "--attn_out",
        type=str,
        help="attention plot filepath",
        default="attention.png",
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="tuple of integers to describe the plot size",
        default=(10,10),
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config != None:
        cfg.merge_from_file(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_transform, vocab_transform, text_transform = prepare_data(cfg)
    
    model = build_model(cfg, vocab_transform, device, args.model)
    model.eval()

    text_tensor = text_transform[cfg.DATASET.SRC_LANGUAGE](args.sentence)
    text_tensor = text_tensor.unsqueeze(1).to(device)

    #tgt_filler = torch.zeros((args.max_length, 1), dtype=torch.long).to(device)

    with torch.no_grad():
        out, attn = model.inference(text_tensor, args.max_length, cfg)

    pred_tokens = out.argmax(2).squeeze(1)
    pred_tokens = [t.item() for t in pred_tokens]
    if cfg.DATASET.EOS_IDX in pred_tokens:
        eos = pred_tokens.index(cfg.DATASET.EOS_IDX) # find the first <eos> token occurence
        pred_tokens_ = pred_tokens[1:eos]
    else: 
        pred_tokens_ = pred_tokens[1:]

    translation = vocab_transform[cfg.DATASET.TGT_LANGUAGE].lookup_tokens(pred_tokens_)
    attn = attn.squeeze(0).cpu().numpy()
    attn = attn[1:eos+1, 1:] # [tgt, src]
    src_tokens = token_transform[cfg.DATASET.SRC_LANGUAGE](args.sentence)
    tgt_tokens = token_transform[cfg.DATASET.TGT_LANGUAGE](" ".join(translation))
    
    src_tokens += ["<eos>"]
    tgt_tokens += ["<eos>"]
    
    print("--"*50)
    print("Input[{}]:\t{}".format(cfg.DATASET.SRC_LANGUAGE, " ".join(src_tokens)))
    print("--"*50)
    print("Output[{}]:\t{}".format(cfg.DATASET.TGT_LANGUAGE, " ".join(tgt_tokens)))

    # Save the attention matrix 
    fig, ax = plt.subplots(1,1, figsize=args.figsize)
    
    img = ax.matshow(attn)
    ax.set_xticks([i for i in range(len(src_tokens))])
    ax.set_yticks([i for i in range(len(tgt_tokens))])
    
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(tgt_tokens)
    
    fig.savefig(args.attn_out)


if __name__ == "__main__":
    main()
