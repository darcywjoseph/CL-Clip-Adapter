import torch 
import torch.nn as nn
from typing import Sequence, Callable, Union
import clip
from model.model import CLIPZeroShotHead

TemplateType = Union[str, Callable[[str], str]]

def get_prompts(cname: str, templates: TemplateType) -> list[str]:
    out = []
    for t in templates:
        if callable(t):
            out.append(t(cname))
        else:
            out.append(t.format(cname))
    return out

torch.no_grad()
def build_clip_zeroshot_head(
    clip_model: nn.Module,
    classnames: list[str],
    device: str,
    templates: Sequence[TemplateType] | None = None,
    use_logit_scale: bool = True,
) -> CLIPZeroShotHead:
    clip_model.eval()

    if templates is None or len(templates) == 0:
        templates = ["a photo of a {}"]

    text_feats = []
    for cname in classnames:
        prompts = get_prompts(cname, templates)
        tokens = clip.tokenize(prompts).to(device)

        emb = clip_model.encode_text(tokens).float()  # (T, D)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        emb = emb.mean(dim=0)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        text_feats.append(emb)

    W = torch.stack(text_feats, dim=0)  # (C, D)

    if use_logit_scale and hasattr(clip_model, "logit_scale"):
        W = W * clip_model.logit_scale.exp().float()

    head = CLIPZeroShotHead(W, normalize=True).to(device)
    for p in head.parameters():
        p.requires_grad = False
    return head
