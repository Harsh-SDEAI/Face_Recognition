"""LVFace backbones - ViT only.

Vendored slim version of `backbones/__init__.py` from bytedance/LVFace.
The original supports many architectures (iresnet, mobilefacenet, poolformer,
ViT) but the released LVFace weights all use ViT, so we only ship `vit.py`
here.  Add other configs if you ever swap in non-ViT weights.
"""
from .vit import VisionTransformer


def get_model(name: str, **kwargs):
    num_features = kwargs.get("num_features", 512)

    if name == "vit_t":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    if name == "vit_t_dp005_mask0":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    if name == "vit_s":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    if name == "vit_s_dp005_mask_0":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    if name == "vit_b":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    if name == "vit_b_dp005_mask_005":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    if name == "vit_l_dp005_mask_005":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    if name == "vit_h":
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True)

    raise ValueError(f"Unknown LVFace backbone name: {name}")
