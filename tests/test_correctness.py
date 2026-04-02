import torch

from bench.config import ABS_ERR_TOL, REL_ERR_TOL
from bench.models import build_model_variants
from bench.utils import max_abs_error, relative_error, set_global_seed


def test_mode_correctness_against_eager():
    if not torch.cuda.is_available():
        return

    set_global_seed(42)
    device = torch.device("cuda")

    models, _ = build_model_variants(
        device=device,
        enable_compile=True,
        try_load_extension=True,
        build_extension_if_missing=False,
    )

    x = torch.randn(4, 1, 32, 32, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref = models["eager"](x)

    for mode in ("compile", "fused"):
        with torch.no_grad():
            out = models[mode](x)

        abs_err = max_abs_error(ref, out)
        rel_err = relative_error(ref, out)

        assert abs_err <= ABS_ERR_TOL, f"{mode} abs err too high: {abs_err}"
        assert rel_err <= REL_ERR_TOL, f"{mode} rel err too high: {rel_err}"


def test_fused_second_block_correctness_against_eager():
    if not torch.cuda.is_available():
        return

    set_global_seed(43)
    device = torch.device("cuda")

    models, _ = build_model_variants(
        device=device,
        enable_compile=False,
        try_load_extension=True,
        build_extension_if_missing=False,
        fuse_second_block=True,
    )

    x = torch.randn(4, 1, 32, 32, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref = models["eager"](x)
        out = models["fused"](x)

    abs_err = max_abs_error(ref, out)
    rel_err = relative_error(ref, out)

    assert abs_err <= ABS_ERR_TOL, f"fused(second block) abs err too high: {abs_err}"
    assert rel_err <= REL_ERR_TOL, f"fused(second block) rel err too high: {rel_err}"
