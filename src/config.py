"""
Configuration Module for Master's Thesis Computational System
Handles environment settings, API credentials, file paths, and default parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = BASE_DIR / "figures"
LIB_DIR = BASE_DIR / "lib"

# Ensure essential directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load Environment Variables
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# LLM Configurations
PRIMARY_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1
MAX_TRANSCRIPT_CHARS = 15000

# File Paths
DEFAULT_GRAPH_DATA_PATH = BASE_DIR / "thesis_graph_data.json"
PROCESSED_GRAPH_DATA_PATH = PROCESSED_DATA_DIR / "thesis_graph_data.json"
PILOT_GRAPH_DATA_PATH = BASE_DIR / "pilot_graph_data.json"
DEFAULT_HTML_OUTPUT_PATH = BASE_DIR / "interactive_thesis_map.html"
METRICS_OUTPUT_PATH = PROCESSED_DATA_DIR / "network_metrics.json"
COMMUNITIES_OUTPUT_PATH = PROCESSED_DATA_DIR / "community_clusters.json"
DIFFUSION_OUTPUT_PATH = PROCESSED_DATA_DIR / "diffusion_results.json"
EVALUATION_OUTPUT_PATH = PROCESSED_DATA_DIR / "baseline_evaluation.json"

# Scale-Diverse Creator Channels
SCALE_DIVERSE_CHANNELS = [
    {
        "id": "UCBJycsmduvYEL83R_U4JriQ",
        "name": "Marques Brownlee",
        "tier": "Mega Hub",
        "subscribers": "> 18M",
        "niche": "Consumer Tech Ecosystem"
    },
    {
        "id": "UCXuqSBlHAE6Xw-yeJA0Tunw",
        "name": "Linus Tech Tips",
        "tier": "Mega Hub",
        "subscribers": "> 15M",
        "niche": "PC Hardware & Tech Culture"
    },
    {
        "id": "UCVYamHliCI9rw1tHR1xbkfw",
        "name": "Dave2D",
        "tier": "Mid-Tier",
        "subscribers": "~ 3.8M",
        "niche": "Laptops & Industrial Design"
    },
    {
        "id": "UCkWQ0gDrqOCarmUKmppD7GQ",
        "name": "JayzTwoCents",
        "tier": "Mid-Tier",
        "subscribers": "~ 4.0M",
        "niche": "Custom PC Building & Cooling"
    },
    {
        "id": "UChIs72whgZI9w6d6FhwGGHA",
        "name": "Gamers Nexus",
        "tier": "Deep Niche",
        "subscribers": "~ 2.2M",
        "niche": "Hardware Benchmarking & Investigation"
    },
    {
        "id": "UCI8iQa1hv7oV_Z8D35vVuSg",
        "name": "Hardware Unboxed",
        "tier": "Deep Niche",
        "subscribers": "~ 1.1M",
        "niche": "GPU/CPU Deep Benchmarking"
    },
    {
        "id": "UCvWWf-LYjaujE50iYai8WgQ",
        "name": "Paul's Hardware",
        "tier": "Mid-Tier",
        "subscribers": "~ 1.4M",
        "niche": "PC Hardware & Guides"
    },
    {
        "id": "UCmqIeAKH-r2kGzH-dWeVj_g",
        "name": "Dawid Does Tech Stuff",
        "tier": "Micro/Specialist",
        "subscribers": "~ 450K",
        "niche": "Budget Hardware & Quirky Engineering"
    }
]

# API Rate Limiting & Robustness
DEFAULT_SLEEP_SECONDS = 5
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
