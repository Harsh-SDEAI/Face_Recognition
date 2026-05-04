"""Run this ONCE from C:\\AlgoTest\\faceeval to patch model_loaders with latest code.

    python patch_models.py

It overwrites arcface.py, magface.py, lvface.py, __init__.py and creates
the lvface_backbones/ folder with the vendored ViT backbone.
"""
import os, textwrap
from pathlib import Path

BASE = Path(__file__).resolve().parent
ML = BASE / "model_loaders"
LBB = ML / "lvface_backbones"
LBB.mkdir(parents=True, exist_ok=True)

def w(path, content):
    Path(path).write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    print(f"  wrote {path}")

print("Patching model_loaders...")

# ── __init__.py ──────────────────────────────────────────────────────────
w(ML / "__init__.py", '''
    """Each loader exposes:
        - class FooEmbedder with .embed(aligned_crop_np) -> np.ndarray (512-d, L2-normalized)
        - load() factory that returns a ready-to-use embedder
    AdaFace additionally returns a (embedding, quality_norm) tuple.
    """
    from . import adaface, arcface, facenet, lvface, magface  # noqa: F401
''')

# ── arcface.py ───────────────────────────────────────────────────────────
w(ML / "arcface.py", '''
    """ArcFace (IR-100 / Glint360K) loader.
    Tries multiple import paths for iresnet100 from InsightFace pip installs.
    """
    from __future__ import annotations
    import cv2, numpy as np, torch
    from torch.nn.functional import normalize
    import config

    def _import_iresnet100():
        errors = []
        for path in (
            "backbones.iresnet",
            "arcface_torch.backbones.iresnet",
            "insightface.recognition.arcface_torch.backbones.iresnet",
        ):
            try:
                mod = __import__(path, fromlist=["iresnet100"])
                return getattr(mod, "iresnet100")
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        try:
            from . import arcface_iresnet
            return getattr(arcface_iresnet, "iresnet100")
        except Exception as exc:
            errors.append(f"local arcface_iresnet: {exc}")
        raise ImportError(
            "Could not import iresnet100 from arcface_torch.  Tried:\\n  - "
            + "\\n  - ".join(errors)
        )

    class ArcFaceEmbedder:
        INPUT_SIZE = 112
        EMBED_DIM = 512
        def __init__(self, device, weights_path=None):
            self.device = device
            iresnet100 = _import_iresnet100()
            self.model = iresnet100(num_features=self.EMBED_DIM)
            state = torch.load(weights_path or config.ARCFACE_WEIGHTS, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(device)
        @torch.no_grad()
        def embed(self, aligned_rgb):
            img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device)
            t = (t / 255.0 - 0.5) / 0.5
            e = self.model(t)
            e = normalize(e, p=2, dim=1)
            return e.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def load(device):
        return ArcFaceEmbedder(device)
''')

# ── magface.py ───────────────────────────────────────────────────────────
w(ML / "magface.py", '''
    """MagFace (IR-100) loader.
    Depends on magface_iresnet.py copied from MagFace repo.
    """
    from __future__ import annotations
    import cv2, numpy as np, torch
    from torch.nn.functional import normalize
    import config

    try:
        from . import magface_iresnet
    except Exception as exc:
        magface_iresnet = None
        _IMPORT_ERR = exc
    else:
        _IMPORT_ERR = None

    class MagFaceEmbedder:
        INPUT_SIZE = 112
        EMBED_DIM = 512
        def __init__(self, device, weights_path=None):
            if magface_iresnet is None:
                raise ImportError(
                    "magface_iresnet.py not found in model_loaders/.  Copy "
                    f"models/iresnet.py from the MagFace repo.  Original: {_IMPORT_ERR}"
                )
            self.device = device
            self.model = magface_iresnet.iresnet100(num_classes=self.EMBED_DIM)
            ckpt = torch.load(weights_path or config.MAGFACE_WEIGHTS, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            state = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(device)
        @torch.no_grad()
        def embed(self, aligned_rgb):
            img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device)
            t = (t / 255.0 - 0.5) / 0.5
            e = self.model(t)
            e = normalize(e, p=2, dim=1)
            return e.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def load(device):
        return MagFaceEmbedder(device)
''')

# ── lvface.py ────────────────────────────────────────────────────────────
w(ML / "lvface.py", '''
    """LVFace (ByteDance, ICCV 2025) loader - PyTorch backend.
    The ViT backbone code is vendored under lvface_backbones/.
    Weights (.pt) auto-download from Hugging Face Hub on first run.
    """
    from __future__ import annotations
    import shutil
    from pathlib import Path
    import cv2, numpy as np, torch
    from torch.nn.functional import normalize
    import config
    from .lvface_backbones import get_model

    def _download_weights_if_missing():
        target = Path(config.LVFACE_WEIGHTS)
        if target.exists():
            return target
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=config.LVFACE_HF_REPO,
            filename=config.LVFACE_HF_FILENAME,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, target)
        return target

    class LVFaceEmbedder:
        INPUT_SIZE = 112
        EMBED_DIM = 512
        def __init__(self, device):
            self.device = device
            self.model = get_model(config.LVFACE_MODEL_NAME, fp16=False)
            weights = _download_weights_if_missing()
            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(device)
        @torch.no_grad()
        def embed(self, aligned_rgb):
            img = cv2.resize(aligned_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device)
            t = (t / 255.0 - 0.5) / 0.5
            e = self.model(t)
            if isinstance(e, (tuple, list)):
                e = e[0]
            e = normalize(e, p=2, dim=1)
            return e.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def load(device):
        return LVFaceEmbedder(device)
''')

# ── lvface_backbones/__init__.py ─────────────────────────────────────────
w(LBB / "__init__.py", '''
    """LVFace ViT backbone factory - vendored from bytedance/LVFace (MIT)."""
    from .vit import VisionTransformer

    def get_model(name, **kwargs):
        num_features = kwargs.get("num_features", 512)
        configs = {
            "vit_t":                dict(embed_dim=256, depth=12, drop_path_rate=0.1,  mask_ratio=0.1,  using_checkpoint=False),
            "vit_t_dp005_mask0":    dict(embed_dim=256, depth=12, drop_path_rate=0.05, mask_ratio=0.0,  using_checkpoint=False),
            "vit_s":                dict(embed_dim=512, depth=12, drop_path_rate=0.1,  mask_ratio=0.1,  using_checkpoint=False),
            "vit_s_dp005_mask_0":   dict(embed_dim=512, depth=12, drop_path_rate=0.05, mask_ratio=0.0,  using_checkpoint=False),
            "vit_b":                dict(embed_dim=512, depth=24, drop_path_rate=0.1,  mask_ratio=0.1,  using_checkpoint=True),
            "vit_b_dp005_mask_005": dict(embed_dim=512, depth=24, drop_path_rate=0.05, mask_ratio=0.05, using_checkpoint=True),
            "vit_l_dp005_mask_005": dict(embed_dim=768, depth=24, drop_path_rate=0.05, mask_ratio=0.05, using_checkpoint=True),
            "vit_h":                dict(embed_dim=1024,depth=48, drop_path_rate=0.1,  mask_ratio=0,    using_checkpoint=True),
        }
        if name not in configs:
            raise ValueError(f"Unknown LVFace backbone: {name}. Options: {list(configs.keys())}")
        c = configs[name]
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, num_heads=8,
            norm_layer="ln", **c)
''')

# ── lvface_backbones/vit.py ──────────────────────────────────────────────
# This one is long; read from the repo copy if it exists, otherwise embed.
VIT_SRC = r'''
# LVFace ViT backbone, vendored from https://github.com/bytedance/LVFace
# Original code from InsightFace project, MIT License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class VITBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)
    def forward(self, x):
        return self.bn(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        with torch.cuda.amp.autocast(True):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2,-1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1,2).reshape(B, N, C)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x); x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU6, norm_layer="ln", patch_n=144):
        super().__init__()
        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.extra_gflops = (num_heads * patch_n * (dim//num_heads) * patch_n * 2) / (1000**3)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1]//patch_size[1]) * (img_size[0]//patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        return self.proj(x).flatten(2).transpose(1, 2)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=112, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 hybrid_backbone=None, norm_layer="ln", mask_ratio=0.1, using_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_channels=in_channels, embed_dim=embed_dim)
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        patch_n = (img_size // patch_size) ** 2
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, num_patches=num_patches, patch_n=patch_n)
            for i in range(depth)])
        self.extra_gflops = sum(b.extra_gflops for b in self.blocks)
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * num_patches, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, eps=2e-5))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    def random_masking(self, x, mask_ratio=0.1):
        N, L, D = x.size()
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.training and self.mask_ratio > 0:
            x, _, ids_restore = self.random_masking(x)
        for func in self.blocks:
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(func, x)
            else:
                x = func(x)
        x = self.norm(x.float())
        if self.training and self.mask_ratio > 0:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]-x.shape[1], 1)
            x_ = torch.cat([x[:,:,:], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
            x = x_
        return torch.reshape(x, (B, self.num_patches * self.embed_dim))
    def forward(self, x):
        x = self.forward_features(x)
        x = self.feature(x)
        return x
'''.lstrip("\n")

(LBB / "vit.py").write_text(VIT_SRC, encoding="utf-8")
print(f"  wrote {LBB / 'vit.py'}")

print("\nDone!  All model_loaders patched.  Now run:")
print("  python generate_embeddings.py")
