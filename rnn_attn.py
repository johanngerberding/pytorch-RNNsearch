import random 
from yacs.config import CfgNode
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# https://github.com/bentrevett/pytorch-seq2seq

class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        enc_hidden_dim: int, 
        dec_hidden_dim: int, 
        dropout: float, 
        bidirectional: bool = True,
    ):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(
            embed_dim, 
            enc_hidden_dim, 
            bidirectional=bidirectional,
        )
        if bidirectional:
            self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        else: 
            self.fc = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src => [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded => [src_len, batch_size, embed_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs => [src_len, batch_size, hidden_dim * num_directions]
        # hidden => [n_layers * num_directions, batch_size, hidden_dim]
        # hidden is stacked [forward1, backward1, forward2, backward2, ...]
        # outputs are always from the last layer
        # hidden[-2, :, :] is the last of the forwards RNN
        # hidden[-1, :, :] is the last of the backwards RNN 
        # the initial decoder hidden is the final hidden state of the backwards and forwards
        # encoder RNNs fed through linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # hidden => [batch_size, dec_hidden_dim]

        return outputs, hidden 


class Attention(nn.Module):
    def __init__(
        self, 
        enc_hidden_dim: int, 
        dec_hidden_dim: int, 
        bidirectional: bool = True,
    ):
        super(Attention, self).__init__()
        if bidirectional:
            self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        else: 
            self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden => [batch_size, dec_hidden_dim]
        # encoder_outputs => [src_len, batch_size, enc_hidden_dim * num_directions]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden state src_len times 
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden => [batch_size, src_len, dec_hidden_dim]
        # encoder_outputs => [batch_size, src_len, enc_hidden_dim * num_directions]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy => [batch_size, src_len, dec_hidden_dim]
        attention = self.v(energy).squeeze(2)
        # attention =>
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self, 
        output_dim: int, 
        embed_dim: int, 
        enc_hidden_dim: int, 
        dec_hidden_dim: int, 
        dropout: float, 
        attention: torch.nn.Module,
        bidirectional: bool = True,
    ):
        super(Decoder, self).__init__()

        self.output_dim = output_dim 
        self.attention = attention 
        self.embedding = nn.Embedding(output_dim, embed_dim)
        
        if bidirectional:
            self.rnn = nn.GRU((enc_hidden_dim * 2) + embed_dim, dec_hidden_dim)
            self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embed_dim, output_dim)
        else: 
            self.rnn = nn.GRU(enc_hidden_dim + embed_dim, dec_hidden_dim)
            self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + embed_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, inp, hidden, encoder_outputs) -> tuple:
        # input => [batch_size]
        # hidden => [batch_size, dec_hidden_dim]
        # encoder_outputs => [src_len, batch_size, enc_hidden_dim * num_directions]
        inp = inp.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        # embedded => [1, batch_size, embed_dim]
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        # a => [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        # weighted => [1, batch_size, enc_hidden_dim * num_directions]
        rnn_inp = torch.cat((embedded, weighted), dim=2)
        # rnn_inp => [1, batch_size, enc_hidden_dim * num_directions + embed_dim]
        output, hidden = self.rnn(rnn_inp, hidden.unsqueeze(0))
        # output => [seq_len, batch_size, dec_hidden_dim * num_directions]
        # hidden => [n_layers * num_directions, batch_size, dec_hidden_dim]
        # seq_len, n_layers and num_directions will always be 1 in this decoder
        # output => [1, batch_size, dec_hidden_dim]
        # hidden => [1, batch_size, dec_hidden_dim]
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        pred = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # pred => [batch_size, output_dim]
        return pred, hidden.squeeze(0), a


class Seq2Seq(nn.Module):
    def __init__(
        self, 
        encoder: torch.nn.Module, 
        decoder: torch.nn.Module, 
        device: torch.device
    ):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio = 0.5):
        # src => [src_len, batch_size]
        # tgt => [tgt_len, batch_size]
        # teacher forcing ratio is probability (0.5 -> 50%)
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs 
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        # encoder_outputs = all hidden states of the input sequence (back & forward)
        # hidden = final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to decoder is the <sos> token
        inp = tgt[0, :]

        for t in range(1, tgt_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (preds) and new hidden state
            output, hidden, _ = self.decoder(inp, hidden, encoder_outputs)
            # a => [batch_size, 1, src_len] 
            # place predictions in a tensor holding predictions for each token 
            outputs[t] = output
            # decide if we are going to use teacher forcing
            teacher = random.random() < teacher_forcing_ratio
            # get highest pred token
            top1 = output.argmax(1)

            inp = tgt[t] if teacher else top1

        return outputs
    

    def inference(self, src, max_length: int, cfg):
        batch_size = src.shape[1]
        tgt_vocab_size = self.decoder.output_dim 
        outputs = torch.zeros(max_length, batch_size, tgt_vocab_size).to(self.device)
        # placeholder for attention scores
        attentions = torch.zeros( max_length, batch_size, src.shape[0]).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        # first decoder input is the <bos> token
        inp = torch.tensor([cfg.DATASET.BOS_IDX]).repeat(batch_size).to(self.device)

        for t in range(1, max_length):
            output, hidden, a = self.decoder(inp, hidden, encoder_outputs)
            a = a.permute(1, 0, 2)

            outputs[t] = output
            attentions[t] = a  

            top1 = output.argmax(1)
            inp = top1
        
        attentions = attentions.permute(1, 0, 2)

        return outputs, attentions



def init_weights(model: torch.nn.Module) -> torch.nn.Module:
    print("Initialize weights...")    
    for name, param in model.named_parameters():
        if 'weight' in name: 
            nn.init.normal_(param.data, mean=0, std=0.01)
        else: 
            nn.init.constant_(param.data, 0)
    return model


def build_model(
    cfg: CfgNode, 
    vocab_transform: dict, 
    device: torch.device, 
    checkpoint: str = "",
) -> torch.nn.Module:
    
    input_dim = len(vocab_transform[cfg.DATASET.SRC_LANGUAGE])
    output_dim = len(vocab_transform[cfg.DATASET.TGT_LANGUAGE])

    attention = Attention(
        enc_hidden_dim=cfg.MODEL.ENC_HIDDEN_DIM,
        dec_hidden_dim=cfg.MODEL.DEC_HIDDEN_DIM,
        bidirectional=cfg.MODEL.ENC_BIDIRECTIONAL,
    )

    enc = Encoder(
        input_dim=input_dim,
        embed_dim=cfg.MODEL.ENC_EMBED_DIM,
        enc_hidden_dim=cfg.MODEL.ENC_HIDDEN_DIM,
        dec_hidden_dim=cfg.MODEL.DEC_HIDDEN_DIM,
        dropout=cfg.MODEL.ENC_DROPOUT,
        bidirectional=cfg.MODEL.ENC_BIDIRECTIONAL,
    )

    dec = Decoder(
        output_dim=output_dim,
        embed_dim=cfg.MODEL.DEC_EMBED_DIM,
        enc_hidden_dim=cfg.MODEL.ENC_HIDDEN_DIM,
        dec_hidden_dim=cfg.MODEL.DEC_HIDDEN_DIM,
        dropout=cfg.MODEL.DEC_DROPOUT,
        attention=attention,
        bidirectional=cfg.MODEL.ENC_BIDIRECTIONAL, # yes this is correct (bad design I guess)
    )

    model = Seq2Seq(enc, dec, device).to(device)

    if checkpoint != "":
        print("Load model checkpoint: {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint))
    else: 
        model = init_weights(model)
    
    return model

"""
def main():
    from config import get_cfg_defaults
    from dataset import prepare_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    _, vocab_transform, text_transform = prepare_data(cfg)
    src = torch.randint(500, (25, 4), dtype=torch.long).to(device) # [src_len, batch_size]
    tgt = torch.randint(500, (20, 4), dtype=torch.long).to(device)
    model = build_model(cfg, vocab_transform, device)
    print(model)

    model.eval()
    with torch.no_grad():
        out = model(src, tgt, 0)
    print(out.size())

    with torch.no_grad():
        pred, attn = model.inference(src, 250, cfg)
    
    print(pred.size())
    print(attn.size())

    test = attn[0].cpu().numpy()
    test = test[:25, :]
    print(test)



if __name__ == "__main__":
    main()"""