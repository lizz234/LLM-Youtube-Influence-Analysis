"""
Master Pipeline Orchestrator
Master's Thesis: Advertising and Influence Analysis via LLM-Generated Graphs from YouTube
Author: Aliza Hamid (UC3M Master's in Big Data Analytics)

Orchestrates sequential, idempotent execution of:
1. Data Extraction (YouTube Data API v3 + Transcripts)
2. Knowledge Graph Extraction (Google Gemini LLM + Pydantic Schema)
3. Network Science & Topological Modeling (NetworkX)
4. Interactive & Publication Visualizations (PyVis + Matplotlib/Seaborn)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DEFAULT_GRAPH_DATA_PATH,
    PROCESSED_GRAPH_DATA_PATH,
    PILOT_GRAPH_DATA_PATH,
    DEFAULT_HTML_OUTPUT_PATH,
    RAW_DATA_DIR,
    SCALE_DIVERSE_CHANNELS
)
from src.data_extraction import YouTubeDataExtractor
from src.kg_extraction import KnowledgeGraphExtractor
from src.network_analysis import NetworkAnalyzer
from src.kg_visualization import KnowledgeGraphVisualizer


def print_banner(step_num: int, title: str):
    """Prints a styled execution banner."""
    print("\n" + "=" * 70)
    print(f"STAGE {step_num}: {title.upper()}")
    print("=" * 70)


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Master Thesis Computational Pipeline Runner (YouTube LLM Knowledge Graph)"
    )
    # Execution Stage Flags
    parser.add_argument("--all", action="store_true", help="Execute the complete end-to-end pipeline")
    parser.add_argument("--data-extraction", action="store_true", help="Execute Stage 1: YouTube Video & Transcript Ingestion")
    parser.add_argument("--kg-json", action="store_true", help="Execute Stage 2: LLM Entity & Relation Extraction")
    parser.add_argument("--network-analysis", action="store_true", help="Execute Stage 3: Network Science & Topological Modeling")
    parser.add_argument("--kg-visualization", action="store_true", help="Execute Stage 4: Interactive HTML & Publication Graphics")

    # Granular Parameter Options
    parser.add_argument("--channels", type=str, default=None, help="Path to channel IDs text file (default: channel_ids.txt)")
    parser.add_argument("--videos", type=str, default=None, help="Path to video URLs/IDs text file")
    parser.add_argument("--limit-videos", type=int, default=10, help="Number of latest videos to sample per channel")
    parser.add_argument("--input-json", type=str, default=None, help="Path to input graph JSON dataset")
    parser.add_argument("--output-html", type=str, default=None, help="Output path for PyVis interactive HTML")
    parser.add_argument("--no-skip-cached", action="store_true", help="Force re-download cached video data")

    args = parser.parse_args()

    # If no flag specified, default to running analysis & visualization if data exists, or show help
    if not any([args.all, args.data_extraction, args.kg_json, args.network_analysis, args.kg_visualization]):
        print("\n[!] No execution flag specified. Defaulting to --all (Full Pipeline Run).")
        args.all = True

    start_time = time.time()
    print("\n" + "#" * 70)
    print("#  MASTER THESIS COMPUTATIONAL PIPELINE")
    print("#  Advertising & Influence Analysis via LLM-Generated Graphs")
    print("#  Researcher: Aliza Hamid | UC3M Big Data Analytics")
    print("#" * 70)

    # -------------------------------------------------------------
    # STAGE 1: DATA EXTRACTION
    # -------------------------------------------------------------
    if args.all or args.data_extraction:
        print_banner(1, "YouTube Data & Transcript Ingestion")
        extractor = YouTubeDataExtractor()

        if args.videos:
            print(f"[*] Extracting videos from file: {args.videos}")
            extractor.process_url_file(args.videos, skip_cached=not args.no_skip_cached)
        else:
            channels_file = Path(args.channels) if args.channels else BASE_DIR / "channel_ids.txt"
            if channels_file.exists():
                print(f"[*] Loading channels list from {channels_file}...")
                with open(channels_file, 'r', encoding='utf-8') as f:
                    c_ids = [line.strip().split('#')[0].strip() for line in f if line.strip() and not line.strip().startswith('#')]
                custom_channels = [{"id": cid, "name": f"Channel_{cid}", "tier": "Custom"} for cid in c_ids]
                extractor.process_channel_batch(custom_channels, limit_per_channel=args.limit_videos, skip_cached=not args.no_skip_cached)
            else:
                print(f"[*] Utilizing default scale-diverse creator catalog ({len(SCALE_DIVERSE_CHANNELS)} channels)...")
                extractor.process_channel_batch(SCALE_DIVERSE_CHANNELS, limit_per_channel=args.limit_videos, skip_cached=not args.no_skip_cached)

    # -------------------------------------------------------------
    # STAGE 2: LLM KNOWLEDGE GRAPH EXTRACTION
    # -------------------------------------------------------------
    if args.all or args.kg_json:
        print_banner(2, "LLM Knowledge Graph Extraction (Google Gemini 2.5)")
        kg_extractor = KnowledgeGraphExtractor()
        extracted_records = kg_extractor.extract_from_raw_cache()
        if extracted_records:
            target_out = Path(args.input_json) if args.input_json else DEFAULT_GRAPH_DATA_PATH
            kg_extractor.save_dataset(extracted_records, target_out)
        else:
            print("[!] No new relationships extracted from raw cache.")

    # -------------------------------------------------------------
    # STAGE 3: NETWORK SCIENCE & TOPOLOGICAL MODELING
    # -------------------------------------------------------------
    if args.all or args.network_analysis:
        print_banner(3, "Network Science & Mathematical Analysis")
        input_data = Path(args.input_json) if args.input_json else DEFAULT_GRAPH_DATA_PATH
        analyzer = NetworkAnalyzer(input_data)
        analyzer.run_full_analysis()

    # -------------------------------------------------------------
    # STAGE 4: VISUALIZATION & PUBLICATION GRAPHICS
    # -------------------------------------------------------------
    if args.all or args.kg_visualization:
        print_banner(4, "Interactive PyVis & Publication Figures")
        input_data = Path(args.input_json) if args.input_json else DEFAULT_GRAPH_DATA_PATH
        analyzer = NetworkAnalyzer(input_data)
        visualizer = KnowledgeGraphVisualizer(analyzer)
        html_out = Path(args.output_html) if args.output_html else DEFAULT_HTML_OUTPUT_PATH
        visualizer.generate_interactive_html(html_out)
        visualizer.generate_publication_figures()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  PIPELINE EXECUTION COMPLETED IN {elapsed:.2f} SECONDS")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
