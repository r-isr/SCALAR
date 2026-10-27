import torch
from timm.models.vision_transformer import VisionTransformer

def forward_with_tokens(self, x, mask_token=None, mask=None, return_norm=False):
    B = x.shape[0]

    # --- timm’s original forward until final norm -----------------
    x = self.patch_embed(x)
    cls = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = self.pos_drop(x + self.pos_embed)

    # optional masking ------------------------------------------------
    if mask is not None and mask_token is not None:
        x[:, 1:][mask.bool()] = mask_token

    for blk in self.blocks:
        x = blk(x)

    tokens = self.norm(x)                       # [B , L+1 , D]

    if not return_norm:
        return self.head(tokens[:, 0])
    else:
        logits = self.head(tokens[:, 0])        # [B , C]
        return logits, tokens                   # tokens includes CLS
# monkey-patch
VisionTransformer.forward = forward_with_tokens
