import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


RESULT_FIELDS = (
    "pickedItem",
    "scored",
    "droppedItem",
    "shoveAttempted",
    "shoveSucceeded",
    "forcedItemDrop",
    "done",
)


def collect_log_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))

    return [input_path]


def parse_rows(input_path: Path) -> tuple[list[dict], list[dict]]:
    causal_rows: list[dict] = []
    summaries: list[dict] = []
    for log_file in collect_log_files(input_path):
        with log_file.open("r", encoding="utf-8-sig") as handle:
            for raw_line in handle:
                line = raw_line.lstrip("\ufeff").strip()
                if not line:
                    continue

                payload = json.loads(line)
                if payload.get("logType") == "CausalStep":
                    causal_rows.append(payload)
                elif payload.get("logType") == "RoundSummary":
                    summaries.append(payload)

    return causal_rows, summaries


def split_tags(row: dict) -> list[str]:
    raw_tags = row.get("causalTags", "")
    return [tag for tag in raw_tags.split(",") if tag]


def write_tag_stats(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accumulators: dict[str, dict] = defaultdict(lambda: {"count": 0, "reward": 0.0, "results": Counter()})

    for row in rows:
        result = row.get("result", {})
        for tag in split_tags(row):
            stats = accumulators[tag]
            stats["count"] += 1
            stats["reward"] += float(result.get("reward", 0.0))
            for field in RESULT_FIELDS:
                if result.get(field, False):
                    stats["results"][field] += 1

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["tag", "count", "avg_reward", *[f"{field}_rate" for field in RESULT_FIELDS]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tag, stats in sorted(accumulators.items()):
            count = max(stats["count"], 1)
            writer.writerow(
                {
                    "tag": tag,
                    "count": stats["count"],
                    "avg_reward": stats["reward"] / count,
                    **{f"{field}_rate": stats["results"][field] / count for field in RESULT_FIELDS},
                }
            )


def write_causal_edges(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        result = row.get("result", {})
        active_results = [field for field in RESULT_FIELDS if result.get(field, False)]
        for tag in split_tags(row):
            for result_field in active_results:
                edge_counts[(tag, result_field)] += 1

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cause_tag", "result", "count"])
        writer.writeheader()
        for (tag, result_field), count in sorted(edge_counts.items()):
            writer.writerow({"cause_tag": tag, "result": result_field, "count": count})


def write_summary(rows: list[dict], summaries: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tag_counts = Counter(tag for row in rows for tag in split_tags(row))
    total_reward = sum(float(row.get("result", {}).get("reward", 0.0)) for row in rows)
    lines = [
        "# Causal Log Summary",
        "",
        f"- Causal steps: {len(rows)}",
        f"- Round summaries: {len(summaries)}",
        f"- Total reward delta in causal rows: {total_reward:.3f}",
        "",
        "## Top Tags",
    ]
    for tag, count in tag_counts.most_common(12):
        lines.append(f"- {tag}: {count}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Deep AI Arena causal logs.")
    parser.add_argument("--input", required=True, help="Path to a jsonl log file or directory")
    parser.add_argument("--output-dir", default="experiments/causal", help="Directory for analysis outputs")
    args = parser.parse_args()

    rows, summaries = parse_rows(Path(args.input))
    if not rows:
        raise RuntimeError(f"No CausalStep rows found in {args.input}")

    output_dir = Path(args.output_dir)
    write_tag_stats(rows, output_dir / "tag_stats.csv")
    write_causal_edges(rows, output_dir / "causal_edges.csv")
    write_summary(rows, summaries, output_dir / "causal_summary.md")
    print(f"Analyzed {len(rows)} causal rows")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
