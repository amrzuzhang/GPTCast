import torch
import torch.nn.functional as F
import lightning as L
from typing import Optional, Union
from pathlib import Path
from gptcast.models import VAEGANVQ
from gptcast.models.components import GPT, GPTCastConfig
from gptcast.utils.converters import (
    dbz_to_rainfall,
    rainfall_to_dbz,
    swvl1_norm_to_phys,
    swvl1_phys_to_norm,
)
import numpy as np
import re
from collections import OrderedDict

import einops
from tqdm.auto import tqdm
import math

    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AbstractEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    

class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c


class ForcingContextEncoder(torch.nn.Module):
    """Encode continuous forcing fields into a small prefix of GPT embeddings."""

    def __init__(
        self,
        in_channels: int,
        n_embd: int,
        n_cond_tokens: int = 4,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.n_cond_tokens = int(n_cond_tokens)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, n_embd, kernel_size=1),
            torch.nn.GELU(),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((self.n_cond_tokens, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).squeeze(-1).transpose(1, 2).contiguous()  # (B, Tcond, Cemb)
        return h


class GPTCast(L.LightningModule):
    default_checkpoint_path = Path(__file__).parent.parent.parent.resolve() / "models"

    @classmethod
    def load_from_pretrained(cls, gpt_chkpt: str, first_stage_chkpt: str, device: str = "cpu") -> 'GPTCast':
        first_stage = VAEGANVQ.load_from_pretrained(first_stage_chkpt, device=device).to(device).eval()
        
        ckpt = torch.load(gpt_chkpt, weights_only=False, map_location=device)

        vocab_size, n_embd = ckpt['state_dict']['transformer.tok_emb.weight'].shape
        block_size, n_embd2 = ckpt['state_dict']['transformer.pos_emb'].shape[-2:]
        assert n_embd == n_embd2, "Number of embeddings in token and position embeddings must match ({} != {})".format(n_embd, n_embd2)
        
        n_layer = 0
        cre = re.compile(r"transformer.blocks.(\d+)")
        for k in ckpt['state_dict'].keys():
            if match := cre.search(k):
                found = int(match.group(1))
                n_layer = max(n_layer, found)
        n_layer += 1
        if n_layer != GPTCastConfig.n_layer:
            print(f"Number of layers in checkpoint ({n_layer}) does not match the expected number of layers ({GPTCastConfig.n_layer}).")

        transformer = GPT(vocab_size, block_size, n_layer, GPTCastConfig.n_head, n_embd).to(device).eval()

        # remap state_dict keys and remove prefix
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith("transformer."):
                new_state_dict[k[12:]] = v

        transformer.load_state_dict(new_state_dict, strict=True)

        # gptcast = cls.load_from_checkpoint(gpt_chkpt, transformer=transformer, first_stage=first_stage, strict=False).to(device).eval()
        gptcast = cls(transformer=transformer, first_stage=first_stage).to(device).eval()

        return gptcast

    def __init__(self,
                 transformer: GPT,
                 first_stage: VAEGANVQ,
                #  permuter=None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 first_stage_key: str = "image",
                 cond_stage_key: Optional[str] = None,
                 use_forcing_conditioning: bool = False,
                 forcing_channels: int = 0,
                 forcing_n_cond_tokens: int = 4,
                 forcing_hidden_channels: int = 128,
                 pkeep: float = 1.0,
                 sos_token: int = 0,
                 base_learning_rate: float = 1e-4,
                 log_physical_metrics: bool = False,
                 physical_metric_context_steps: Optional[int] = None,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['first_stage', 'transformer']) #, 'permuter'])
        self.base_learning_rate = base_learning_rate
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.use_forcing_conditioning = bool(use_forcing_conditioning)

        self.first_stage_model = first_stage
        assert(self.first_stage_model.hparams.freeze_weights)
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train

        if self.use_forcing_conditioning:
            if int(forcing_channels) < 1:
                raise ValueError("forcing_channels must be >= 1 when use_forcing_conditioning=True")
            self.be_unconditional = False
            self.cond_stage_key = cond_stage_key or "forcing"
            self.cond_stage_model = None
            self.forcing_context_encoder = ForcingContextEncoder(
                in_channels=int(forcing_channels),
                n_embd=int(transformer.config.n_embd),
                n_cond_tokens=int(forcing_n_cond_tokens),
                hidden_channels=int(forcing_hidden_channels),
            )
        else:
            # force unconditional training
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key if cond_stage_key is None else cond_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)

        # self.permuter = Identity() if permuter is None else permuter

        self.transformer = transformer

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @staticmethod
    def _norm_to_phys(x: torch.Tensor, clip: tuple[float, float], norm: tuple[float, float]) -> torch.Tensor:
        cmin, cmax = float(clip[0]), float(clip[1])
        nmin, nmax = float(norm[0]), float(norm[1])
        out = (x - nmin) / (nmax - nmin + 1e-12)
        out = out * (cmax - cmin) + cmin
        return out

    @staticmethod
    def _split_stacked_sequence(x: torch.Tensor, seq_len: int, stack_seq: Optional[str]) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D BCHW tensor, got shape {tuple(x.shape)}")

        if stack_seq is None:
            if x.shape[1] == seq_len:
                return x
            if seq_len == 1 and x.shape[1] == 1:
                return x
            raise ValueError(
                f"Cannot split unstacked sequence with shape {tuple(x.shape)} for seq_len={seq_len}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"Expected a single-channel stacked tensor for stack_seq={stack_seq!r}, got shape {tuple(x.shape)}"
            )

        bsz, _, height, width = x.shape
        if stack_seq == "v":
            if height % seq_len != 0:
                raise ValueError(f"Height {height} is not divisible by seq_len={seq_len}")
            frame_h = height // seq_len
            return x[:, 0].reshape(bsz, seq_len, frame_h, width)
        if stack_seq == "h":
            if width % seq_len != 0:
                raise ValueError(f"Width {width} is not divisible by seq_len={seq_len}")
            frame_w = width // seq_len
            return x[:, 0].reshape(bsz, height, seq_len, frame_w).permute(0, 2, 1, 3).contiguous()

        raise ValueError(f"Unsupported stack_seq={stack_seq!r}")

    def _teacher_forced_outputs(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        quant_z, z_indices = self.encode_to_z(x)
        cond_embeddings = None
        c_indices = None
        if self.use_forcing_conditioning:
            cond_embeddings = self.encode_forcing(c)
        else:
            _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape, device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        if self.use_forcing_conditioning:
            logits, _ = self.transformer(a_indices[:, :-1], embeddings=cond_embeddings)
            logits = logits[:, cond_embeddings.shape[1] - 1:]
        else:
            cz_indices = torch.cat((c_indices, a_indices), dim=1)
            logits, _ = self.transformer(cz_indices[:, :-1])
            logits = logits[:, c_indices.shape[1]-1:]

        return logits, target, quant_z.shape

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, target, _ = self._teacher_forced_outputs(x, c)
        return logits, target

    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def encode_to_z(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info.view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # if self.downsample_cond_size > -1:
        #     c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def encode_forcing(self, c: torch.Tensor) -> torch.Tensor:
        if not self.use_forcing_conditioning:
            raise RuntimeError("encode_forcing called but use_forcing_conditioning=False")
        return self.forcing_context_encoder(c)

    @torch.no_grad()
    def decode_to_img(self, index: torch.Tensor, zshape: torch.Size) -> torch.Tensor:
        # index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def predict_next_index(
        self,
        context: torch.Tensor,
        temperature: float = 1.,
        top_k: Optional[int] = None,
        cond_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert context.ndim == 2
        logits, _ = self.transformer(context, embeddings=cond_embeddings)
        # we just need the prediction for the last token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_k == 1:
            _, res = torch.topk(probs, k=1, dim=-1)
        else:
            res = torch.multinomial(probs, num_samples=1)

        assert(res.shape[1] == 1)
        return res

    @torch.no_grad()
    def predict_next_frame_indices(
        self,
        input_indices: torch.Tensor,
        c_indices: Optional[torch.Tensor],
        window_size: int,
        temperature: float = 1.,
        top_k: Optional[int] = None,
        show_progress: bool = True,
        pbar_position: int = 0,
        cond_embeddings: Optional[torch.Tensor] = None,
        ):
        idx_batch, idx_step, idx_h, idx_w = input_indices.shape  # b s h w

        predicted_indices = (torch.ones((idx_batch, idx_h, idx_w), dtype=input_indices.dtype, device=self.device) * -1)
        if show_progress:
            progressbar = tqdm(total=idx_h*idx_w, leave=True, desc="token", position=pbar_position)
        for i in range(0, idx_h):
            half_window = window_size // 2
            if i <= half_window:
                local_i = i
            elif idx_h - i < half_window:
                local_i = window_size - (idx_h - i)
            else:
                local_i = half_window
            for j in range(0, idx_w):
                if j <= half_window:
                    local_j = j
                elif idx_w - j < half_window:
                    local_j = window_size - (idx_w - j)
                else:
                    local_j = half_window

                i_start = i - local_i
                i_end = i_start + window_size
                j_start = j - local_j
                j_end = j_start + window_size
                # print(
                #     f"rows: {i_start}-{i_end}, cols: {j_start}-{j_end}, abs_target_pos: {i}-{j}, local_target_pos: {local_i}-{local_j}")

                past_patches = input_indices[:, :, i_start:i_end, j_start:j_end]
                past_tokens = past_patches.reshape(past_patches.shape[0], -1)

                predicted_patch = predicted_indices[:, i_start:i_end, j_start:j_end]
                predicted_tokens = predicted_patch.reshape(predicted_patch.shape[0], -1)[:,
                                :local_i * predicted_patch.shape[1] + local_j]

                if c_indices is not None:
                    conditioning = c_indices.reshape(c_indices.shape[0], -1)
                    full_context = torch.cat((conditioning, past_tokens, predicted_tokens), dim=1)
                else:
                    full_context = torch.cat((past_tokens, predicted_tokens), dim=1)
                res = self.predict_next_index(full_context, temperature, top_k, cond_embeddings=cond_embeddings)
                predicted_indices[:, i, j] = res

                if show_progress:
                    progressbar.update()

        return predicted_indices
        # return {
        #     'predicted_indices': ,
        #     'predicted_quant_shape': torch.Size([idx_batch, quant_input_shape[1], idx_h, idx_w])
        # }

    @torch.no_grad()
    def predict_sequence(
        self,
        seq: torch.Tensor,
        forcing_seq: Optional[torch.Tensor] = None,
        steps: int = 1,
        window_size: Optional[int] = 16,
        padding_value: float =-1.,
        future: bool = True,
        temperature: float = 1.,
        top_k: Optional[int] = None,
        ae_precision: torch.dtype = torch.float32,
        gpt_precision: torch.dtype = torch.bfloat16,
        verbosity: int = 0,
        pbar_position: int = 0
        ) -> dict[str, torch.Tensor]:
        """
        Predict future frames for a given sequence. If future is True, the model
        will predict the next `steps` frames. If future is False, the model will
        predict the last `steps` frames of the sequence (i.e. the last `steps` frames
        are not used for prediction). The pretrained models support a maximum input sequence length of 7 steps.

        Args:
            seq (torch.Tensor): Input sequence of shape (h w s). The values should be in the range [-1, 1] where -1
                corresponds to 0 DBz and 1 corresponds to 60DBZ (or the maximum reflectivity value).
            steps (int): Number of steps to predict
            window_size (Optional[int]): GPT spatial window size (this is model dependent)
            padding_value (float): Value used for padding the input sequence
            future (bool): If True, predict the next `steps` frames, otherwise predict the last `steps` frames
            temperature (float): Temperature used for sampling the next token
            top_k (Optional[int]): If not None, sample from the top k tokens
            ae_precision (torch.dtype): Precision used for the autoencoder
            gpt_precision (torch.dtype): Precision used for the GPT model
            verbosity (int): Verbosity level
            pbar_position (int): Position of the progress bar
        """
        assert not self.transformer.training
        assert len(seq.shape) == 3
        # assert isinstance(self.first_stage_model, VQModel)
        seq = seq.to(device=self.device)
        seq = einops.rearrange(seq, 'h w s -> s h w')

        if future:
            assert steps >= 1
            input_sequence = seq
            target_sequence = None
        else:
            assert 1 <= steps < seq.shape[0]
            input_sequence = seq[:-steps]
            target_sequence = seq[-steps:]

        num_down = self.first_stage_model.encoder.num_resolutions-1
        patch_size = 2**num_down
        if window_size is None:
            # try to infer GPT spatial window size from trasnformer config
            # we assume that the transformer was trained with a temporal window size of 8
            # and derive the spatial window size from the block size
            window_size = int(math.sqrt(self.transformer.config.block_size // 8))
            print(f"Using window size of {window_size}x{window_size} tokens")

        window_size_pixel = window_size*patch_size
        # print(window_size_pixel)
        in_steps, h, w = input_sequence.shape
        assert (h >= window_size_pixel) and \
               (w >= window_size_pixel), f"Window size x patch size ({window_size}x{patch_size}={window_size_pixel})" \
                                         f" cannot be bigger than image height/width"
        # print(f"Patch size is {patch_size}x{patch_size} pixels")
        bottom_pad = (patch_size - h % patch_size) % patch_size
        right_pad = (patch_size - w % patch_size) % patch_size
        if bottom_pad + right_pad != 0:
            # print(f"Input tensor height/width is not a multiple of {patch_size}, padding tensor with {padding_value}")
            # print(f"Original size: {in_steps}steps x {h}h x {w}w,"
            #       f" padded size: {in_steps}steps x {h+bottom_pad}h x {w+right_pad}w")
            input_sequence = F.pad(input_sequence, (0, right_pad, 0, bottom_pad), value=padding_value)
            # in_steps, h, w = sequence.shape

        x = einops.rearrange(input_sequence, 's h w -> (s h) w')[None, None, ...]
        x = x.to(memory_format=torch.contiguous_format).float()
        c = x

        with torch.autocast(self.device.type, enabled=True if ae_precision!=torch.float32 else False, dtype=ae_precision):
            quant_input, input_indices = self.encode_to_z(x)
            x_rec = self.decode_to_img(input_indices, quant_input.shape).squeeze()
            x_rec = x_rec.reshape(in_steps, x_rec.shape[0]//in_steps, x_rec.shape[1])

        c_indices = None
        cond_embeddings = None
        if self.use_forcing_conditioning:
            if forcing_seq is None:
                raise ValueError("forcing_seq must be provided when use_forcing_conditioning=True")
            forcing_x = forcing_seq.to(device=self.device)
            if forcing_x.ndim != 3:
                raise ValueError("forcing_seq must have shape (h, w, c)")
            forcing_x = einops.rearrange(forcing_x, 'h w c -> 1 c h w').to(memory_format=torch.contiguous_format).float()
            with torch.autocast(self.device.type, enabled=True if gpt_precision!=torch.float32 else False, dtype=gpt_precision):
                cond_embeddings = self.encode_forcing(forcing_x)
        else:
            _, c_indices = self.encode_to_c(c)
        quant_shape = quant_input.shape
        indices = input_indices.reshape(quant_shape[0], in_steps, quant_shape[2] // in_steps, quant_shape[3])
        ind_b, ind_s, ind_h, ind_w = indices.shape
        # print(quant_input.shape, input_indices.shape, c_indices.shape, indices.shape)

        predicted_seq = list()
        disabe_step_progress = verbosity < 1
        show_token_progress = verbosity > 1
        for i in tqdm(range(steps), total=steps, desc="Timestep", disable=disabe_step_progress):
            with torch.autocast(self.device.type, enabled=True if gpt_precision!=torch.float32 else False, dtype=gpt_precision):
                predicted_indices = self.predict_next_frame_indices(indices[:, -in_steps:], c_indices, window_size,
                                                                    temperature=temperature, top_k=top_k,
                                                                    show_progress=show_token_progress,
                                                                    cond_embeddings=cond_embeddings,)    
            with torch.autocast(self.device.type, enabled=True if ae_precision!=torch.float32 else False, dtype=ae_precision):
                predicted_image = self.decode_to_img(
                    predicted_indices.reshape(predicted_indices.shape[0], -1),
                    torch.Size([ind_b, quant_shape[1], ind_h, ind_w])
                )
            predicted_seq.append(predicted_image)

            # append predicted_indices and remove first frame
            indices = torch.cat((indices, predicted_indices[:, None, ...]), dim=1)

        # pred_image = decoded.squeeze().numpy().clip(-1, 1)
        predicted_seq = torch.cat(predicted_seq, dim=1)
        result = {
            'input_indices': indices[:, :in_steps],
            'input_sequence_pad': input_sequence,
            'input_sequence_nopad': input_sequence[..., :h, :w],
            'input_reconstruction': x_rec[..., :h, :w],
            'pred_indices': indices[:, -in_steps:],
            'pred_sequence_pad': predicted_seq,
            'pred_sequence_nopad': predicted_seq[..., :h, :w],
        }
        if target_sequence is not None:
            result['target_sequence'] = target_sequence

        return result

    @torch.no_grad()
    def forecast(
        self,
        input_sequence: Union[np.ndarray, np.ma.MaskedArray],
        steps: int = 1,
        units: str = "mm/h",
        mask: Optional[np.ndarray] = None,
        verbosity: int = 1,
        ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Forecast future frames for a given sequence. The model will predict the next `steps` frames.

        Args:
            x (np.ndarray or np.ma.MaskedArray): Input sequence of shape (s h w). Accepts both dbz and mm/h units.
                                                 Values are converted to dbz internally and clipped to 0-60 dbz.
                                                 That is the rage of the model. The model can leverage up to 7 input steps
                                                 of context, if the input sequence is longer, only the last 7 steps are used.
            steps (int): Number of steps to predict
            units (str): Units of the output. Can be either 'mm/h' or 'dbz'.
            mask (np.ndarray): 2D Mask to apply to all frames. Should have the same shape height x width as the input sequence.
                               If the input sequence is already a masked array, the mask will be added to the existing mask.
            verbosity (int): Verbosity level for the prediction process. 0: no output, 1: timestep output, 2: token output
        """
        assert len(input_sequence.shape) == 3
        assert units in ["mm/h", 'dbz'], "Only 'mm/h' and 'dbz' units are supported"
        assert steps >= 1
        if mask is not None:
            assert mask.shape == input_sequence.shape[1:], "Mask shape should match the input sequence shape"

        if input_sequence.shape[0] > 7:
            input_sequence = input_sequence[-7:]
            print("Input sequence is longer than 7 steps, only the last 7 steps are used.")

        # separate mask and data
        if isinstance(input_sequence, np.ma.MaskedArray):
            x_m = input_sequence.mask
            x = input_sequence.data
        else:
            x = input_sequence
            x_m = np.zeros_like(input_sequence, dtype=bool)

        # repeat mask for all input steps and add to input mask
        if mask is not None:
            mask = np.broadcast_to(mask, (input_sequence.shape[0], *mask.shape))
            x_m = np.logical_or(x_m, mask)

        # create output mask
        input_mask_sum = x_m.sum(axis=0).astype(bool)
        output_mask = np.broadcast_to(input_mask_sum, (steps, *input_mask_sum.shape))

        # if input is in mm/h convert back to pseudo dbz
        if units == "mm/h":
            x = x.clip(0) # this should not be necessary, but just in case
            x = rainfall_to_dbz(x)
        
        x = x.clip(0, 60) # limit the range to 0-60 dbz
        x = (x / 30.) -1 # rescale to -1, 1

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = einops.rearrange(x, 's h w -> h w s')

        with torch.no_grad():
            result = self.predict_sequence(x, steps=steps, future=True, window_size=None, verbosity=verbosity)
        y = result['pred_sequence_nopad'].cpu().numpy().squeeze().clip(-1,1)
        # rescale to 0-60 dbz
        y = (y + 1) * 30

        # convert back to mm/h if necessary
        if units == "mm/h":
            y = dbz_to_rainfall(y)

        # set dtype to input dtype
        y = y.astype(input_sequence.dtype)

        # apply output mask if it is not all false or if the input is a masked array
        if output_mask.any() or isinstance(input_sequence, np.ma.MaskedArray):
            y = np.ma.masked_array(y, mask=output_mask)
        
        return y

    @torch.no_grad()
    def forecast_swvl1(
        self,
        input_sequence: Union[np.ndarray, np.ma.MaskedArray],
        *,
        steps: int = 1,
        normalized: bool = True,
        forcing_sequence: Optional[Union[np.ndarray, np.ma.MaskedArray]] = None,
        mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 1,
        verbosity: int = 1,
        clip: tuple[float, float] = (0.0, 0.8),
        norm: tuple[float, float] = (-1.0, 1.0),
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """Forecast future ERA5-Land SWVL1 fields.

        Args:
            input_sequence: Context sequence with shape `(S, H, W)`.
            steps: Number of forecast steps.
            normalized: If `True`, `input_sequence` is assumed to already be in
                model space. If `False`, values are interpreted as physical
                SWVL1 in `m3/m3`.
            mask: Optional static land/ocean mask with shape `(H, W)`.
            temperature: Sampling temperature for autoregressive decoding.
            top_k: If `1`, use greedy decoding. If `None`, sample from the full
                token distribution.
            verbosity: Progress verbosity passed to `predict_sequence`.
            clip: Physical clipping range used during training.
            norm: Normalized range used during training.
        """
        assert len(input_sequence.shape) == 3, "Input must have shape (steps, height, width)"
        assert steps >= 1
        if mask is not None:
            assert mask.shape == input_sequence.shape[1:], "Mask shape should match the input sequence shape"

        if input_sequence.shape[0] > 7:
            input_sequence = input_sequence[-7:]
            print("Input sequence is longer than 7 steps, only the last 7 steps are used.")

        if isinstance(input_sequence, np.ma.MaskedArray):
            x_m = input_sequence.mask
            x = input_sequence.data
        else:
            x = input_sequence
            x_m = np.zeros_like(input_sequence, dtype=bool)

        input_dtype = input_sequence.dtype

        if mask is not None:
            mask = np.broadcast_to(mask, (input_sequence.shape[0], *mask.shape))
            x_m = np.logical_or(x_m, mask)

        output_mask = np.broadcast_to(x_m.sum(axis=0).astype(bool), (steps, *x_m.shape[1:]))

        if not normalized:
            x = swvl1_phys_to_norm(x, clip=clip, norm=norm)

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = einops.rearrange(x, 's h w -> h w s')

        forcing_tensor = None
        if forcing_sequence is not None:
            if isinstance(forcing_sequence, np.ma.MaskedArray):
                forcing_tensor = torch.tensor(forcing_sequence.data, dtype=torch.float32).to(self.device)
            else:
                forcing_tensor = torch.tensor(np.asarray(forcing_sequence), dtype=torch.float32).to(self.device)

        result = self.predict_sequence(
            x,
            forcing_seq=forcing_tensor,
            steps=steps,
            future=True,
            window_size=None,
            temperature=temperature,
            top_k=top_k,
            verbosity=verbosity,
        )

        y = result['pred_sequence_nopad'].cpu().numpy().squeeze().clip(norm[0], norm[1])
        if steps == 1 and y.ndim == 2:
            y = y[None, ...]
        if not normalized:
            y = swvl1_norm_to_phys(y, clip=clip, norm=norm)

        y = y.astype(input_dtype)
        if output_mask.any() or isinstance(input_sequence, np.ma.MaskedArray):
            y = np.ma.masked_array(y, mask=output_mask)
        return y

    def get_input(self, key: str, batch: dict) -> torch.Tensor:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_mask(self, batch: dict, key: str) -> Optional[torch.Tensor]:
        if key is None or key not in batch:
            return None
        mask = batch[key]
        if len(mask.shape) == 3:
            mask = mask[:, None, ...]
        return mask.to(memory_format=torch.contiguous_format).bool()

    def get_xc(self, batch: dict, N: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def _get_physical_metric_settings(
        self,
    ) -> tuple[str, tuple[float, float], tuple[float, float], int, Optional[str]]:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "hparams"):
            raise RuntimeError("Physical metrics require an instantiated datamodule with hparams.")

        clip_and_normalize = getattr(datamodule.hparams, "clip_and_normalize", None)
        if clip_and_normalize is None or len(clip_and_normalize) != 4:
            raise RuntimeError(
                "Physical metrics require data.clip_and_normalize=(clip_min, clip_max, norm_min, norm_max)."
            )

        clip = (float(clip_and_normalize[0]), float(clip_and_normalize[1]))
        norm = (float(clip_and_normalize[2]), float(clip_and_normalize[3]))
        seq_len = int(getattr(datamodule.hparams, "seq_len", 1))
        stack_seq = getattr(datamodule.hparams, "stack_seq", None)
        image_variable_key = str(getattr(datamodule.hparams, "image_variable_key", self.first_stage_key))
        return image_variable_key, clip, norm, seq_len, stack_seq

    def _compute_teacher_forced_physical_metrics(
        self,
        batch: dict,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not bool(self.hparams.log_physical_metrics):
            return None

        image_variable_key, clip, norm, seq_len, stack_seq = self._get_physical_metric_settings()
        x, c = self.get_xc(batch)
        batch_size = int(x.shape[0])

        with torch.no_grad():
            logits, _, z_shape = self._teacher_forced_outputs(x, c)
            pred_indices = logits.argmax(dim=-1)
            pred_img = self.decode_to_img(pred_indices.reshape(pred_indices.shape[0], -1), z_shape)

        target_seq = self._split_stacked_sequence(x.detach(), seq_len=seq_len, stack_seq=stack_seq)
        pred_seq = self._split_stacked_sequence(pred_img.detach(), seq_len=seq_len, stack_seq=stack_seq)

        mask_seq = None
        if "mask" in batch:
            mask = self.get_mask(batch, "mask")
            mask_seq = self._split_stacked_sequence(mask.float(), seq_len=seq_len, stack_seq=stack_seq).bool()

        total_steps = int(target_seq.shape[1])
        context_steps = self.hparams.physical_metric_context_steps
        if context_steps is None or int(context_steps) < 0 or int(context_steps) >= total_steps:
            eval_start = 0
        else:
            eval_start = int(context_steps)

        pred_eval = pred_seq[:, eval_start:]
        target_eval = target_seq[:, eval_start:]
        if mask_seq is not None:
            mask_eval = mask_seq[:, eval_start:]
        else:
            mask_eval = None

        pred_phys = self._norm_to_phys(pred_eval, clip=clip, norm=norm)
        target_phys = self._norm_to_phys(target_eval, clip=clip, norm=norm)
        diff = pred_phys - target_phys
        abs_diff = diff.abs()
        sq_diff = diff.pow(2)

        if mask_eval is not None:
            valid = (~mask_eval).to(dtype=pred_phys.dtype)
            reduce_dims = (0, 2, 3)
            denom = valid.sum(dim=reduce_dims).clamp_min(1.0)
            mae = (abs_diff * valid).sum(dim=reduce_dims) / denom
            rmse = torch.sqrt((sq_diff * valid).sum(dim=reduce_dims) / denom)
        else:
            reduce_dims = (0, 2, 3)
            mae = abs_diff.mean(dim=reduce_dims)
            rmse = torch.sqrt(sq_diff.mean(dim=reduce_dims))

        return {
            "variable": image_variable_key,
            "mae": mae,
            "rmse": rmse,
            "batch_size": torch.tensor(batch_size, device=mae.device, dtype=mae.dtype),
        }

    def _log_teacher_forced_physical_metrics(self, split: str, metrics: Optional[dict[str, torch.Tensor]]) -> None:
        if metrics is None:
            return

        mae = metrics["mae"]
        rmse = metrics["rmse"]
        batch_size = int(metrics["batch_size"].item())

        self.log(
            f"{split}/tf_phys_mae_mean",
            mae.mean(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{split}/tf_phys_rmse_mean",
            rmse.mean(),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        for lead_idx, (lead_mae, lead_rmse) in enumerate(zip(mae, rmse), start=1):
            self.log(
                f"{split}/tf_phys_mae_lead_{lead_idx:02d}",
                lead_mae,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                f"{split}/tf_phys_rmse_lead_{lead_idx:02d}",
                lead_rmse,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def shared_step(self, batch: dict) -> torch.Tensor:
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        batch_size = int(batch[self.first_stage_key].shape[0])
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        batch_size = int(batch[self.first_stage_key].shape[0])
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        metrics = self._compute_teacher_forced_physical_metrics(batch)
        self._log_teacher_forced_physical_metrics("val", metrics)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch)
        batch_size = int(batch[self.first_stage_key].shape[0])
        self.log(
            "test/loss",
            loss,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        metrics = self._compute_teacher_forced_physical_metrics(batch)
        self._log_teacher_forced_physical_metrics("test", metrics)
        return loss

    def configure_optimizers(self):
        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices
        self.learning_rate = agb * ngpu * bs * self.base_learning_rate
        return self.transformer.configure_optimizers(self.learning_rate, fused=True)
