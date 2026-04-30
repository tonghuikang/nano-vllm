import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONCURRENCIES = [1, 4, 16, 64, 256, 1024]
PREFIXES = [1, 4096, 32768]


def _parse_summary(path):
    rows = {}
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != 8 or not cells[0].replace(" ", "").isdigit():
            continue
        prefix = int(cells[0])
        values = [float(cell.replace(",", "")) for cell in cells[1:7]]
        rows[prefix] = dict(zip(CONCURRENCIES, values))
    return rows


def _table_after(text, marker):
    start = text.index(marker)
    lines = text[start:].splitlines()
    table = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            table.append(line)
            in_table = True
        elif in_table:
            break
    return table


def _parse_markdown_values(text, marker, ratio=False):
    rows = {}
    for line in _table_after(text, marker):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7 or not cells[0].replace(",", "").isdigit():
            continue
        prefix = int(cells[0].replace(",", ""))
        raw_values = cells[1:7]
        if ratio:
            values = [float(value.rstrip("x×")) for value in raw_values]
        else:
            values = [float(value.replace(",", "")) for value in raw_values]
        rows[prefix] = dict(zip(CONCURRENCIES, values))
    return rows


class BenchmarkDocsTest(unittest.TestCase):
    def test_readme_tables_match_checked_in_benchmark_summaries(self):
        readme = (ROOT / "README.md").read_text()
        nano = _parse_summary(ROOT / "bench" / "bench_nanovllm.txt")
        vllm = _parse_summary(ROOT / "bench" / "bench_vllm.txt")

        self.assertEqual(_parse_markdown_values(readme, "**Nano-vLLM**"), nano)
        self.assertEqual(_parse_markdown_values(readme, "**vLLM**"), vllm)

        ratios = _parse_markdown_values(readme, "**Speed ratio", ratio=True)
        expected_ratios = {
            prefix: {
                concurrency: round(vllm[prefix][concurrency] / nano[prefix][concurrency], 2)
                for concurrency in CONCURRENCIES
            }
            for prefix in PREFIXES
        }
        self.assertEqual(ratios, expected_ratios)
        self.assertLessEqual(max(value for row in ratios.values() for value in row.values()), 1.10)

    def test_full_sweep_doc_matches_checked_in_benchmark_summaries(self):
        full_sweep = (ROOT / "bench" / "FULL_SWEEP_20260429.md").read_text()
        nano = _parse_summary(ROOT / "bench" / "bench_nanovllm.txt")
        vllm = _parse_summary(ROOT / "bench" / "bench_vllm.txt")

        self.assertEqual(_parse_markdown_values(full_sweep, "## nano-vLLM"), nano)
        self.assertEqual(_parse_markdown_values(full_sweep, "## vLLM"), vllm)

        expected_min = min(
            nano[prefix][concurrency] / vllm[prefix][concurrency]
            for prefix in PREFIXES
            for concurrency in CONCURRENCIES
        )
        match = re.search(r"lowest ratio is ([0-9.]+)x", full_sweep)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), expected_min, places=3)
        self.assertGreaterEqual(expected_min, 0.95)


if __name__ == "__main__":
    unittest.main()
