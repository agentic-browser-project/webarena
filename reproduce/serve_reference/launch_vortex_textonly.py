"""Thin launcher: parse standard sglang CLI args, force enable_multimodal=False
(text-only), then launch the HTTP server. Edits no library code.

IMPORTANT: this file must NOT live in /home/cc — Python puts the script's own dir on
sys.path[0], and /home/cc/vortex_torch (the repo) would then shadow the real vortex_torch
package as an empty namespace package (vortex_torch.__file__=None, flow never imported).
Living in /home/cc/webarena/sr_compare/wa_exp (no vortex_torch sibling) makes the editable install resolve."""
import os
import sys

# Belt-and-suspenders: drop any sys.path entry that contains a vortex_torch *directory*
# without an __init__.py (namespace-shadow guard), then import the real package + registry.
sys.path = [p for p in sys.path
            if not (p and os.path.isdir(os.path.join(p, "vortex_torch"))
                    and not os.path.exists(os.path.join(p, "vortex_torch", "__init__.py")))]
import vortex_torch          # real package (has __init__ -> imports flow)
import vortex_torch.flow     # registers block_sparse_attention in the algorithm registry

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    # Force text-only so model_config.is_multimodal becomes False and the
    # vortex flashinfer backend's `assert not is_multimodal` passes.
    server_args.enable_multimodal = False
    print(f"[textonly-launcher] enable_multimodal set to {server_args.enable_multimodal}", flush=True)

    from sglang.srt.entrypoints.http_server import launch_server

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
