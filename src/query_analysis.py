"""
Query Analysis and Visualization Tool

Runs queries through the current chatbot pipeline and summarizes intent quality,
fallback behavior, confidence/margin thresholds, response source, and rankings.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError:  # Visualization is optional; text/CSV/JSON analysis still works.
    plt = None

try:
    from . import chatbot as chatbot_module
except ImportError:
    import chatbot as chatbot_module


@dataclass
class QueryRecord:
    timestamp: str
    query: str
    detected_intent: str
    response: str
    is_fallback: bool
    confidence: float
    margin: float
    source: str
    rankings: list[tuple[str, float]] = field(default_factory=list)
    top_ranked_intent: str | None = None
    top_ranked_score: float | None = None
    fallback_reasons: list[str] = field(default_factory=list)


class AnalysisMemory(chatbot_module.ChatSessionMemory):
    """In-memory session used by analysis so persistent user data is not changed."""

    def _load_persistent(self):
        return None

    def _save_persistent(self):
        return None


class QueryAnalyzer:
    def __init__(
        self,
        *,
        confidence_threshold: float | None = None,
        margin_threshold: float | None = None,
        allow_llm_fallback: bool = False,
    ):
        self.confidence_threshold = (
            chatbot_module.CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.margin_threshold = (
            chatbot_module.MARGIN_THRESHOLD
            if margin_threshold is None
            else margin_threshold
        )
        self.allow_llm_fallback = allow_llm_fallback

        self.query_responses: list[QueryRecord] = []
        self.intent_counts = Counter()
        self.top_intent_counts = Counter()
        self.source_counts = Counter()
        self.fallback_by_top_intent = defaultdict(int)
        self.low_confidence_count = 0
        self.low_margin_count = 0

    @property
    def fallback_count(self):
        return sum(1 for entry in self.query_responses if entry.is_fallback)

    @property
    def success_count(self):
        return len(self.query_responses) - self.fallback_count

    def add_query_response(
        self,
        query: str,
        detected_intent: str,
        response: str,
        *,
        is_fallback: bool = False,
        confidence: float = 0.0,
        margin: float = 0.0,
        source: str = "local",
        rankings: list[tuple[str, float]] | None = None,
    ):
        """Track one query/response result from the current chatbot pipeline."""
        rankings = rankings or []
        top_ranked_intent = rankings[0][0] if rankings else detected_intent
        top_ranked_score = float(rankings[0][1]) if rankings else confidence

        fallback_reasons = []
        if confidence < self.confidence_threshold:
            fallback_reasons.append("low_confidence")
        if margin < self.margin_threshold:
            fallback_reasons.append("low_margin")
        if detected_intent == chatbot_module.FALLBACK_TAG:
            fallback_reasons.append("fallback_intent")
        if source in {"local_fallback", "gemini"}:
            fallback_reasons.append(f"source:{source}")

        entry = QueryRecord(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            query=query,
            detected_intent=detected_intent,
            response=response,
            is_fallback=is_fallback,
            confidence=float(confidence),
            margin=float(margin),
            source=source,
            rankings=[(intent, float(score)) for intent, score in rankings],
            top_ranked_intent=top_ranked_intent,
            top_ranked_score=top_ranked_score,
            fallback_reasons=fallback_reasons,
        )
        self.query_responses.append(entry)

        self.intent_counts[detected_intent] += 1
        self.top_intent_counts[top_ranked_intent] += 1
        self.source_counts[source] += 1

        if confidence < self.confidence_threshold:
            self.low_confidence_count += 1
        if margin < self.margin_threshold:
            self.low_margin_count += 1
        if is_fallback:
            self.fallback_by_top_intent[top_ranked_intent] += 1

        return entry

    def analyze_query(self, query: str, memory: AnalysisMemory | None = None):
        """Run one query through chat_once and record its current metadata."""
        memory = memory or AnalysisMemory()

        original_llm_fallback = chatbot_module.llm_generate_fallback_response
        if not self.allow_llm_fallback:
            chatbot_module.llm_generate_fallback_response = lambda *_args, **_kwargs: (
                None
            )

        try:
            result = chatbot_module.chat_once(query, memory)
        finally:
            chatbot_module.llm_generate_fallback_response = original_llm_fallback

        source = result.get("source") or "local"
        detected_intent = result["intent"]
        is_fallback = detected_intent == chatbot_module.FALLBACK_TAG or source in {
            "local_fallback",
            "gemini",
        }

        return self.add_query_response(
            query,
            detected_intent,
            result["response"],
            is_fallback=is_fallback,
            confidence=float(result.get("confidence", 0.0)),
            margin=float(result.get("margin", 0.0)),
            source=source,
            rankings=result.get("rankings", []),
        )

    def analyze_queries(self, queries: list[str], *, shared_memory: bool = True):
        """Analyze multiple queries, optionally preserving conversation context."""
        memory = AnalysisMemory() if shared_memory else None
        for query in queries:
            self.analyze_query(query, memory if shared_memory else None)

    def get_fallback_rate_by_top_intent(self):
        """Calculate fallback rate for each top-ranked model intent."""
        rates = {}
        for intent, count in self.top_intent_counts.items():
            fallback = self.fallback_by_top_intent.get(intent, 0)
            rate = (fallback / count * 100) if count > 0 else 0.0
            rates[intent] = {
                "total_queries": count,
                "fallback_count": fallback,
                "fallback_rate": rate,
            }
        return rates

    def load_sample_data(self):
        """Run representative sample queries through the real chatbot."""
        print("Loading sample data through current chatbot...")
        sample_queries = [
            "hi",
            "hello there",
            "how are you",
            "my name is Aditya",
            "what is your name",
            "what time is it",
            "help",
            "/help",
            "thank you",
            "thanks a lot",
            "bye",
            "see you",
            "remind me to drink water tomorrow morning",
            "recommend a technology podcast",
            "can you suggest a funny podcast for my commute",
            "what's the weather in London",
            "how do I learn programming",
            "tell me a joke about databases",
            "what did we just talk about",
            "yes",
            "no",
        ]
        self.analyze_queries(sample_queries, shared_memory=True)

    def _as_dicts(self):
        return [asdict(entry) for entry in self.query_responses]

    def save_json(self, output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._as_dicts(), indent=2), encoding="utf-8")
        print(f"Saved JSON analysis to {path}")

    def save_csv(self, output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self._as_dicts()
        if not rows:
            print("No data to save.")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = row.copy()
                row["rankings"] = json.dumps(row["rankings"])
                row["fallback_reasons"] = json.dumps(row["fallback_reasons"])
                writer.writerow(row)
        print(f"Saved CSV analysis to {path}")

    def visualize(self, *, save_path: str | Path | None = None, show: bool = True):
        """Create visualizations of current chatbot confidence/fallback behavior."""
        if not self.query_responses:
            print("No data to visualize!")
            return
        if plt is None:
            print("matplotlib is not installed, so visualizations are unavailable.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Chatbot Query Analysis", fontsize=16, fontweight="bold")

        ax1 = axes[0, 0]
        labels = list(self.source_counts.keys())
        sizes = [self.source_counts[label] for label in labels]
        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Responses by Source", fontweight="bold")

        ax2 = axes[0, 1]
        intents = [intent for intent, _count in self.intent_counts.most_common()]
        counts = [self.intent_counts[intent] for intent in intents]
        colors = [
            "#e74c3c" if intent == chatbot_module.FALLBACK_TAG else "#3498db"
            for intent in intents
        ]
        bars = ax2.barh(intents, counts, color=colors)
        ax2.set_xlabel("Number of Queries", fontweight="bold")
        ax2.set_title("Final Detected Intent", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width, bar.get_y() + bar.get_height() / 2, f" {int(width)}", va="center"
            )

        ax3 = axes[1, 0]
        rates = self.get_fallback_rate_by_top_intent()
        intents_sorted = sorted(
            rates.keys(), key=lambda name: rates[name]["fallback_rate"], reverse=True
        )
        fallback_rates = [rates[intent]["fallback_rate"] for intent in intents_sorted]
        rate_colors = [
            "#e74c3c" if rate > 50 else "#f39c12" if rate > 0 else "#2ecc71"
            for rate in fallback_rates
        ]
        ax3.barh(intents_sorted, fallback_rates, color=rate_colors)
        ax3.set_xlabel("Fallback Rate (%)", fontweight="bold")
        ax3.set_title("Fallback Rate by Top-Ranked Intent", fontweight="bold")
        ax3.set_xlim(0, 100)
        ax3.grid(axis="x", alpha=0.3)

        ax4 = axes[1, 1]
        non_fallback = [
            entry for entry in self.query_responses if not entry.is_fallback
        ]
        fallback = [entry for entry in self.query_responses if entry.is_fallback]
        ax4.scatter(
            [entry.confidence for entry in non_fallback],
            [entry.margin for entry in non_fallback],
            label="accepted",
            color="#2ecc71",
            alpha=0.75,
        )
        ax4.scatter(
            [entry.confidence for entry in fallback],
            [entry.margin for entry in fallback],
            label="fallback",
            color="#e74c3c",
            alpha=0.75,
        )
        ax4.axvline(
            self.confidence_threshold,
            color="#34495e",
            linestyle="--",
            label="confidence threshold",
        )
        ax4.axhline(
            self.margin_threshold,
            color="#8e44ad",
            linestyle="--",
            label="margin threshold",
        )
        ax4.set_xlabel("Confidence", fontweight="bold")
        ax4.set_ylabel("Margin", fontweight="bold")
        ax4.set_title("Confidence vs Margin", fontweight="bold")
        ax4.set_xlim(0, 1.05)
        ax4.set_ylim(0, 1.05)
        ax4.grid(alpha=0.3)
        ax4.legend(loc="upper right")

        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def print_summary(self):
        """Print a detailed text summary of the analysis."""
        total = len(self.query_responses)
        print("\n" + "=" * 72)
        print("QUERY ANALYSIS SUMMARY")
        print("=" * 72)
        print(f"Total Queries: {total}")
        print(f"Confidence Threshold: {self.confidence_threshold:.2f}")
        print(f"Margin Threshold: {self.margin_threshold:.2f}")

        if total == 0:
            print("No queries analyzed yet.")
            print("=" * 72 + "\n")
            return

        confidences = [entry.confidence for entry in self.query_responses]
        margins = [entry.margin for entry in self.query_responses]
        print(f"Successful Local/Handled Responses: {self.success_count}")
        print(f"Fallback Responses: {self.fallback_count}")
        print(f"Overall Success Rate: {(self.success_count / total * 100):.1f}%")
        print(f"Overall Fallback Rate: {(self.fallback_count / total * 100):.1f}%")
        print(f"Low Confidence Count: {self.low_confidence_count}")
        print(f"Low Margin Count: {self.low_margin_count}")
        print(f"Average Confidence: {mean(confidences):.3f}")
        print(f"Average Margin: {mean(margins):.3f}")

        print("\n" + "-" * 72)
        print("RESPONSES BY SOURCE:")
        print("-" * 72)
        for source, count in self.source_counts.most_common():
            print(f"{source}: {count} ({count / total * 100:.1f}%)")

        print("\n" + "-" * 72)
        print("FINAL INTENT COUNTS:")
        print("-" * 72)
        for intent, count in self.intent_counts.most_common():
            print(f"{intent}: {count}")

        print("\n" + "-" * 72)
        print("FALLBACK RATE BY TOP-RANKED MODEL INTENT:")
        print("-" * 72)
        rates = self.get_fallback_rate_by_top_intent()
        for intent in sorted(
            rates.keys(), key=lambda name: rates[name]["fallback_rate"], reverse=True
        ):
            data = rates[intent]
            print(
                f"{intent}: {data['fallback_count']}/{data['total_queries']} "
                f"({data['fallback_rate']:.1f}%)"
            )

        print("\n" + "-" * 72)
        print("QUERY DETAILS:")
        print("-" * 72)
        for entry in self.query_responses:
            reasons = (
                ", ".join(entry.fallback_reasons) if entry.fallback_reasons else "none"
            )
            print(
                f"{entry.query!r} -> intent={entry.detected_intent}, "
                f"top={entry.top_ranked_intent}, confidence={entry.confidence:.3f}, "
                f"margin={entry.margin:.3f}, source={entry.source}, fallback={entry.is_fallback}, "
                f"reasons={reasons}"
            )
        print("\n" + "=" * 72 + "\n")


def load_queries_from_file(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            return [str(item) for item in data["queries"]]
        raise ValueError(
            "JSON query file must be a list or an object with a 'queries' list."
        )

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description="Analyze chatbot query handling.")
    parser.add_argument(
        "--queries-file", help="Text file with one query per line, or JSON list/object."
    )
    parser.add_argument(
        "--json", dest="json_path", help="Optional path to save detailed JSON results."
    )
    parser.add_argument(
        "--csv", dest="csv_path", help="Optional path to save detailed CSV results."
    )
    parser.add_argument(
        "--plot", dest="plot_path", help="Optional path to save a PNG visualization."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open the matplotlib window."
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Allow Gemini fallback calls during analysis.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("CHATBOT QUERY ANALYSIS TOOL")
    print("=" * 72 + "\n")

    analyzer = QueryAnalyzer(allow_llm_fallback=args.allow_llm)

    if args.queries_file:
        queries = load_queries_from_file(args.queries_file)
        analyzer.analyze_queries(queries, shared_memory=True)
    else:
        analyzer.load_sample_data()

    analyzer.print_summary()

    if args.json_path:
        analyzer.save_json(args.json_path)
    if args.csv_path:
        analyzer.save_csv(args.csv_path)

    if args.plot_path or not args.no_show:
        print("Generating visualizations...")
        analyzer.visualize(save_path=args.plot_path, show=not args.no_show)


if __name__ == "__main__":
    main()
