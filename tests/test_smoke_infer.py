import torch

from bench.models import build_model_variants
from bench.utils import set_global_seed


def test_smoke_inference_shapes_and_finite():
    if not torch.cuda.is_available():
        return

    set_global_seed(123)
    device = torch.device("cuda")

    models, _ = build_model_variants(
        device=device,
        enable_compile=True,
        try_load_extension=True,
        build_extension_if_missing=False,
    )

    x = torch.randn(2, 1, 32, 32, device=device, dtype=torch.float32)

    for name, model in models.items():
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 10), f"unexpected output shape in mode={name}: {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"non-finite output in mode={name}"


def test_smoke_inference_cudagraph_modes():
    if not torch.cuda.is_available():
        return

    set_global_seed(124)
    device = torch.device("cuda")

    models, _ = build_model_variants(
        device=device,
        enable_compile=False,
        enable_cudagraph=True,
        try_load_extension=True,
        build_extension_if_missing=False,
    )

    x = torch.randn(2, 1, 32, 32, device=device, dtype=torch.float32)

    for name, model in models.items():
        if not name.endswith("_graph"):
            continue

        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 10), f"unexpected output shape in mode={name}: {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"non-finite output in mode={name}"
