import torch
import torch.nn as nn

from reformer_pytorch import ReformerLM, Reformer, Autopadder
from charformer_pytorch import GBST
from tokenizer import encode, decode
from tqdm import tqdm

class SIALM(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            n_encoder_layers=12,
            n_decoder_layers=4,
            bucket_size=32,
            num_tokens=258, # idx=256: PAD, idx=257=EOS
            gbst_max_block_size=4,
            gbst_dim=128,
            gbst_downscale_factor=4,
            encoder_max_seq_len=2048,
            decoder_max_seq_len=256,
            ):
        super().__init__()
        self.gbst = GBST(
                num_tokens=num_tokens,
                dim=gbst_dim,
                max_block_size=gbst_max_block_size,
                downsample_factor=gbst_downscale_factor
                )
        self.encoder_pe = nn.Parameter(torch.randn(1, encoder_max_seq_len, d_model))
        self.gbst2dmodel = nn.Linear(gbst_dim, d_model)
        self.encoder = Reformer(
                dim=d_model,
                depth=n_encoder_layers,
                heads=n_heads,
                lsh_dropout=0.1,
                causal=False,
                num_mem_kv=0,
                ff_activation=nn.ReLU,
                one_value_head=True,
                bucket_size=bucket_size,
                full_attn_thres=bucket_size,
                layer_dropout = 0.2,
                )
        self.decoder = ReformerLM(
                num_tokens = num_tokens,
                dim=d_model,
                max_seq_len=decoder_max_seq_len,
                depth=n_decoder_layers,
                heads=n_heads,
                lsh_dropout=0.1,
                causal=True,
                num_mem_kv=0,
                ff_activation=nn.ReLU,
                one_value_head=True,
                bucket_size=bucket_size,
                full_attn_thres=bucket_size,
                weight_tie_embedding=True,
                )
        self.bucket_size = bucket_size
        self.encoder = Autopadder(self.encoder)
        self.decoder = Autopadder(self.decoder)

    def forward(self, src, tgt):
        src, _ = self.gbst(src)
        src = self.gbst2dmodel(src)
        src = src + self.encoder_pe[:, :src.shape[1]]
        mem = self.encoder(src)
        out = self.decoder(tgt, keys=mem)
        return out

    def predict_sentence(self, sentence, max_length=128, show_progress=False):
        with torch.no_grad(): # without gradients
            device = self.parameters().__next__().device
            src = torch.LongTensor([encode(sentence)]).to(device)
            src, _ = self.gbst(src)
            src = self.gbst2dmodel(src)
            src = src + self.encoder_pe[:, :src.shape[1]]
            mem = self.encoder(src)
            tgt = torch.full((1, 1), 256).to(device)
            tgt = torch.argmax(self.decoder(tgt, keys=mem), dim=2)
            it = range(max_length-1)
            it = tqdm(it) if show_progress else it
            for i in it:
                tgt = torch.cat([tgt, torch.argmax(self.decoder(tgt, keys=mem), dim=2)[:, -1:]], dim=1)
                if tgt[0, -1] == 256 or tgt[0, -1] == 257:
                    break
            return decode(list(tgt[0].cpu().numpy()))
