from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    lr_root: Path
    gt_root: Path
    frame_match: str = "exact"
    sequence_source: str = "pred"
    description: str = ""
    notes: str = ""

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["lr_root"] = str(self.lr_root)
        data["gt_root"] = str(self.gt_root)
        data["lr_exists"] = self.lr_root.exists()
        data["gt_exists"] = self.gt_root.exists()
        return data


def resolve_project_data_root(workdir: str | Path) -> Path:
    return Path(workdir).resolve()


def get_gt_ready_benchmarks(workdir: str | Path) -> dict[str, BenchmarkDataset]:
    root = resolve_project_data_root(workdir)

    benchmarks = [
        BenchmarkDataset(
            name="reds_sample",
            lr_root=root / "data/official/reds_sample/lr_sequences",
            gt_root=root / "data/official/reds_sample/gt_sequences",
            description="Mandatory REDS sample benchmark with paired LR/GT sequences.",
            notes="Ready for direct Part 2 inference and full-metric evaluation.",
        ),
        BenchmarkDataset(
            name="vimeo_lr",
            lr_root=root / "data/official/vimeo_lr/lr_sequences",
            gt_root=root / "data/benchmarks/vimeo90k/original/sequences",
            description="Mandatory Vimeo LR benchmark. GT is resolved from the full Vimeo-90K original sequences.",
            notes="Predicted subset can be evaluated directly against the full Vimeo GT tree using sequence_source=pred.",
        ),
        BenchmarkDataset(
            name="wild_video_benchmark_112x64",
            lr_root=root / "data/official/wild_video_benchmark_112x64/lr_sequences",
            gt_root=root / "data/official/wild_video_benchmark_112x64/gt_sequences",
            description="Wild video benchmark prepared from extracted frames with LR 112x64 and GT 448x256.",
            notes="Sequence names are normalized to 1..7 and frame names use 8-digit PNG format.",
        ),
        BenchmarkDataset(
            name="vimeo90k_full_test",
            lr_root=root / "results/eval_assets/vimeo90k_full_test_lr",
            gt_root=root / "data/benchmarks/vimeo90k/original/sequences",
            description="Prepared full Vimeo-90K test LR tree aligned with the original GT sequences.",
            notes="This is the most complete GT-bearing Vimeo evaluation set currently ready in the project.",
        ),
    ]
    return {item.name: item for item in benchmarks}


def get_benchmark_or_raise(workdir: str | Path, name: str) -> BenchmarkDataset:
    benchmarks = get_gt_ready_benchmarks(workdir)
    if name not in benchmarks:
        valid = ", ".join(sorted(benchmarks))
        raise KeyError(f"Unknown benchmark dataset '{name}'. Available: {valid}")
    return benchmarks[name]
