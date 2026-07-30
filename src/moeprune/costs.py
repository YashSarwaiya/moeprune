"""Cost model for the rented-hardware sweep.

Two corrections to the method sketch in CLAUDE.md, both favorable:

1. 8xH100 (640 GB VRAM) cannot hold the 1.56 TB checkpoint at all. Full-model
   GPU inference needs ~8xB200 or 16xH200. BUT:

2. Calibration needs NO GPU. Router logging is a teacher-forced prefill over
   prompt+reference text — no generation. llama.cpp on a big-RAM/NVMe CPU box
   streams the weights; ~100k calibration tokens in large prefill chunks means
   only a few dozen full sweeps over the weight file. Checkpoint surgery
   (shard rewriting) is likewise pure CPU/disk work. And the unpruned baseline
   comes free from Moonshot's published benchmark numbers.

   So GPUs are rented only to EVALUATE PRUNED variants — and the deep-pruned
   points fit on small cheap nodes. The curve's expensive end is the SHALLOW
   end (25% pruning barely shrinks the model), which is also its least
   interesting end scientifically.

All prices are mid-2026 on-demand marketplace rates; edit Assumptions to taste.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ModelSpec, SweepSpec, pruned_checkpoint_bytes


@dataclass
class GpuNode:
    name: str
    vram_bytes: float
    usd_per_hour: float


DEFAULT_NODES = (
    GpuNode("4xH100-80GB", 0.32e12, 7.6),
    GpuNode("8xH100-80GB", 0.64e12, 15.2),
    GpuNode("8xH200-141GB", 1.13e12, 20.0),
    GpuNode("8xB200-192GB", 1.54e12, 32.0),
    GpuNode("16xH200-141GB", 2.26e12, 40.0),
)


@dataclass
class Assumptions:
    cpu_box_usd_per_hour: float = 3.0     # ~2 TB RAM or fast NVMe RAID, EPYC
    download_hours: float = 6.0           # 1.56 TB from HF with hf_transfer
    calib_hours: float = 8.0              # prefill-only routing pass, CPU box
    surgery_hours: float = 4.0            # read 1.56 TB once, write all variants
    vram_headroom: float = 1.25           # KV cache + activations margin
    eval_problems: int = 2300             # HumanEval+ / MBPP+ / LCB + OOD set
    tokens_per_problem: float = 3400.0    # prompt + thinking(low) + answer
    eval_tok_per_sec: float = 1200.0      # batched decode, conservative
    eval_setup_hours: float = 1.0         # load + warmup per point


def _cheapest_fit(bytes_needed: float, a: Assumptions,
                  nodes=DEFAULT_NODES) -> GpuNode | None:
    need = bytes_needed * a.vram_headroom
    fitting = [n for n in nodes if n.vram_bytes >= need]
    return min(fitting, key=lambda n: n.usd_per_hour) if fitting else None


def estimate(model: ModelSpec, sweep: SweepSpec,
             a: Assumptions = Assumptions()) -> dict:
    eval_hours = (a.eval_problems * a.tokens_per_problem
                  / a.eval_tok_per_sec / 3600.0) + a.eval_setup_hours
    points = []
    for r in sweep.ratios:
        size = pruned_checkpoint_bytes(model, r)
        node = _cheapest_fit(size, a)
        points.append({
            "ratio": r,
            "size_gb": size / 1e9,
            "node": node.name if node else "DOES NOT FIT LISTED NODES",
            "hours": eval_hours,
            "usd": eval_hours * node.usd_per_hour if node else float("nan"),
        })
    cpu_hours = a.download_hours + a.calib_hours + a.surgery_hours
    cpu_usd = cpu_hours * a.cpu_box_usd_per_hour
    eval_usd = sum(p["usd"] for p in points)
    return {
        "cpu_phase": {"hours": cpu_hours, "usd": cpu_usd},
        "eval_points": points,
        "total_usd": cpu_usd + eval_usd,
        "eval_hours_per_point": eval_hours,
    }


def format_table(est: dict) -> str:
    lines = [
        "PHASE 0-2 (CPU box: download + routing logs + shard surgery)",
        f"  {est['cpu_phase']['hours']:.0f} h  ->  ${est['cpu_phase']['usd']:.0f}",
        "",
        "PHASE 3 (GPU evals, one rental per sweep point)",
        f"  {'ratio':>6}  {'size':>8}  {'node':<28}  {'hours':>5}  {'USD':>6}",
    ]
    for p in est["eval_points"]:
        lines.append(f"  {p['ratio']:>5.0%}  {p['size_gb']:>6.0f}GB  "
                     f"{p['node']:<28}  {p['hours']:>5.1f}  {p['usd']:>6.0f}")
    lines += [
        "",
        f"TOTAL: ${est['total_usd']:.0f}"
        "   (baseline evals: use Moonshot's published numbers, $0)",
        "Optional recovery pass (expert-wise KD at the best knee point): "
        "$1-3k, decide after seeing the curve.",
    ]
    return "\n".join(lines)
