# Advertising & Influence Analysis via LLM-Generated Knowledge Graphs from YouTube

<div align="center">

**A Scalable Computational Framework for Latent Commercial Relationship Extraction, Topological Centrality Modeling, Modularity-Based Sub-Niche Detection, and Information Diffusion Simulation**

*Master's Thesis — Master in Big Data Analytics*  
**Universidad Carlos III de Madrid (UC3M) — Academic Year 2025/2026**  
**Author / Researcher:** Aliza Hamid

[Key Findings](#-key-empirical-findings) • [Pipeline Architecture](#-computational-architecture) • [Quick Start](#-quick-start--reproduction) • [Repository Structure](#-repository-structure) • [Interactive Graph](#-interactive-web-visualization) • [Citation](#-bibtex-citation)

</div>

---

## Executive Summary

Modern influencer marketing on YouTube operates through narrative disclosures, conversational recommendations, and embedded product demonstrations rather than traditional static ads. Because these commercial ties are spoken within video audio, they remain latent and invisible to conventional metadata scrapers.

This Master's Thesis presents an autonomous, end-to-end computational pipeline that unifies Generative Artificial Intelligence (**Google Gemini 2.5 Flash**), strict **Pydantic schema validation**, and **Network Science (NetworkX & PyVis)** to extract, structure, and analyze creator-brand relationship networks at scale.

```
YouTube Transcripts (ASR) ──► Gemini 2.5 Flash (Pydantic) ──► Directed Bipartite KG ──► Centrality & Diffusion ──► Interactive Vis.js
```

### Key Analytical Capabilities:
* **LLM Zero-Shot Relation Extraction:** Automatically parses video transcripts to extract structured `(Creator, Brand, Relation, Product, Context)` tuples with sentiment classifications (*promotes*, *criticizes*, *mentions*).
* **View-Weighted Directed Bipartite Graph:** Models the commercial ecosystem as $G = (V_C \cup V_B, E)$, weighting connections by empirical video viewership and sentiment multipliers.
* **Topological Centrality Modeling:** Computes In/Out-Degree, Eigenvector Centrality (network-based structural prominence), Betweenness Centrality (structural boundary spanners), and PageRank.
* **Unsupervised Modularity Clustering:** Partitions the network into organic market niches using Louvain community detection ($Q$).
* **Stochastic Information Diffusion:** Simulates promotional cascade propagation using the Independent Cascade Model (ICM) across Monte Carlo iterations.
* **Empirical Benchmarking:** Quantifies extraction precision, recall, and F1-score against a hand-annotated ground-truth test suite and a rule-based regex baseline.

---

## Computational Architecture

```mermaid
flowchart TD
    A[YouTube Channel Catalog\nchannel_ids.txt] --> B[Stage 1: Data Ingestion\nsrc/data_extraction.py]
    B -->|Metadata & Views| C[YouTube Data API v3]
    B -->|ASR Transcripts| D[YouTube Transcript API]
    C & D --> E[Raw Local Cache\ndata/raw/video_id/]
    
    E --> F[Stage 2: LLM Extraction Engine\nsrc/kg_extraction.py]
    F -->|Google Gemini 2.5 Flash\nPydantic JSON Schema| G[Master Knowledge Graph\ndata/processed/thesis_graph_data.json]
    
    G --> H[Stage 3: Network Science Modeling\nsrc/network_analysis.py]
    H -->|Weighted Centralities| I[Topological Hubs & Bridges]
    H -->|Louvain Modularity| J[Community Sub-Niches]
    H -->|Monte Carlo ICM| K[Information Diffusion Cascades]
    H -->|Fidelity Benchmark| L[Baseline Comparison Evaluation]
    
    G & H --> M[Stage 4: Visualization Engine\nsrc/kg_visualization.py]
    M --> N[Interactive Web Graph\ninteractive_thesis_map.html]
    M --> O[Publication Graphics 300 DPI\nfigures/*.png]
```

### Modular Pipeline Stages:
1. **Data Ingestion (`src/data_extraction.py`):** Interfaces with YouTube Data API v3 and `youtube-transcript-api`. Features local disk caching (`data/raw/<video_id>/`) and exponential backoff.
2. **Structured LLM Extraction (`src/kg_extraction.py`):** Sends windowed transcripts to Gemini 2.5 Flash using structured Pydantic schemas (`BrandMention`, `ExtractionResult`) for deterministic JSON output.
3. **Network Science Computation (`src/network_analysis.py`):** Constructs directed bipartite graphs in NetworkX, calculating exposure-weighted degree, eigenvector structural prominence, betweenness bridges, Louvain clusters, and ICM diffusion dynamics.
4. **Interactive Graph Visualization (`src/kg_visualization.py`):** Renders standalone, interactive Vis.js HTML5 networks (`interactive_thesis_map.html`) and 300 DPI publication plots (`figures/`).

---

## Key Empirical Findings

| Metric / Dimension | Finding | Academic & Commercial Implication |
| :--- | :--- | :--- |
| **LLM Extraction Fidelity** | **F1 = 0.902** (Precision: 0.884, Recall: 0.921) vs **F1 = 0.610** for Baseline | **+47.8% relative gain**; robust handling of slang, affiliate mentions, and phonetic transcription errors. |
| **Topological Degree Distribution** | Heavy-tailed distribution with **$\gamma \approx 1.84$** | Follows power-law characteristics of human social graphs; top hubs capture majority of exposure. |
| **Eigenvector Centrality vs Reach** | Mid-Tier creators (Dave2D, JayzTwoCents) exhibit disproportionately high prominence | Mid-tier influencers connect high-value sub-graphs (consumer tech + PC hardware) exceeding raw subscriber scale. |
| **Structural Bridging (Betweenness)** | Technical benchmarkers (Gamers Nexus: $C_B = 0.583$) act as critical gatekeepers | Crucial boundary-spanners linking disparate industrial sub-niches that mega-hubs do not bridge. |
| **Information Diffusion (ICM)** | Distributed Mid-Tier seeding achieves **+28.4% greater reach** than Mega-Hubs | Allocating marketing budgets across multi-creator mid-tier portfolios circumvents local cluster bottlenecks. |

---

## Scale-Diverse Creator Cohort

The empirical evaluation was conducted across a curated cohort representing 7 scale-diverse channels ($|V| = 116, |E| = 150$):

| Creator Channel | Tier | Subscribers | Niche Focus | Network Structural Role |
| :--- | :--- | :--- | :--- | :--- |
| **Marques Brownlee (MKBHD)** | Mega Hub | $> 18.5\text{M}$ | Consumer Tech & Smartphones | Generalist Consumer Anchor |
| **Linus Tech Tips (LTT)** | Mega Hub | $> 15.8\text{M}$ | PC Hardware, DIY & Tech Culture | Enthusiast Ecosystem Anchor |
| **Dave2D** | Mid-Tier | $\sim 3.8\text{M}$ | Laptops & Industrial Design | Mobile Hardware Specialist |
| **JayzTwoCents** | Mid-Tier | $\sim 4.0\text{M}$ | Custom Liquid Cooling & PC Modding | PC Hardware Specialist |
| **Gamers Nexus** | Deep Niche | $\sim 2.2\text{M}$ | Teardowns, Thermals & Investigation | Consumer Advocacy & Benchmark Anchor |
| **Hardware Unboxed** | Deep Niche | $\sim 1.1\text{M}$ | GPU/CPU Quantitative Benchmarks | Silicon Performance Specialist |
| **Paul's Hardware** | Mid-Tier | $\sim 1.4\text{M}$ | PC Building Guides & Market Walkthroughs | Buying Advisor & Component Bridge |

---

## Repository Structure

```text
.
├── pipeline.py                       # Master CLI runner for full pipeline orchestration
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for API credentials
├── channel_ids.txt                   # YouTube channel catalog for ingestion
├── interactive_thesis_map.html       # Standalone interactive Vis.js graph visualizer
│
├── src/                              # Modular Python package
│   ├── __init__.py                   # Package initializer
│   ├── config.py                     # Central configuration paths, multipliers & constants
│   ├── data_extraction.py            # Stage 1: YouTube API and ASR transcript harvester
│   ├── kg_extraction.py              # Stage 2: Gemini 2.5 Flash LLM extraction engine
│   ├── network_analysis.py           # Stage 3: NetworkX topological algorithms & ICM simulation
│   └── kg_visualization.py          # Stage 4: Interactive HTML and figure generator
│
├── data/                             # Data directory
│   └── processed/                    # Processed datasets and analytics artifacts
│       ├── thesis_graph_data.json    # Master knowledge graph dataset (nodes & edges)
│       ├── network_metrics.json      # Centrality metrics (Degree, Eigenvector, Betweenness)
│       ├── community_clusters.json   # Louvain community partitions
│       ├── diffusion_results.json    # Monte Carlo ICM cascade curves
│       └── baseline_evaluation.json  # Benchmark evaluation metrics (Precision, Recall, F1)
│
├── figures/                          # 300 DPI publication-grade figures
│   ├── pipeline_architecture_diagram.png
│   ├── network_topology_overview.png
│   ├── degree_distribution_powerlaw.png
│   ├── centrality_rankings.png
│   ├── diffusion_cascade_curves.png
│   └── extraction_fidelity_benchmark.png
│
└── README.md                         # Project documentation (this file)
```

---

## Quick Start & Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/lizz234/masters-thesis.git
cd masters-thesis
```

### 2. Set Up Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials
Copy the `.env.example` template to `.env` and insert your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
YOUTUBE_API_KEY=your_youtube_data_api_v3_key_here
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Execute the End-to-End Pipeline
```bash
# Run all four stages sequentially:
python pipeline.py --all

# Or run individual modules:
python pipeline.py --data-extraction     # Ingest video metadata and captions
python pipeline.py --kg-json             # Extract relationships via Gemini Flash
python pipeline.py --network-analysis    # Compute metrics & diffusion simulation
python pipeline.py --kg-visualization   # Generate interactive HTML map and plots

# Run with custom parameters:
python pipeline.py --all --limit-videos 5 --channels channel_ids.txt
```

---

## Interactive Web Visualization

To interactively explore the extracted YouTube Knowledge Graph:
1. Open `interactive_thesis_map.html` directly in any modern web browser (Google Chrome, Firefox, Safari, Edge). No local server required.
2. **Features:**
   * **ForceAtlas2 Physics Solver:** Dynamic node layout and physics stabilization.
   * **Entity Color Coding:** Blue nodes indicate Content Creators; green nodes indicate Commercial Brands.
   * **View-Weighted Edge Thickness:** Visual line thickness proportional to log-scaled audience views.
   * **Interactive Filtering:** Search and isolate specific brands, creators, or sponsorship clusters.
   * **Context Tooltips:** Hover over any directed edge to view empirical view counts and the verbatim textual sponsorship disclosure snippet.

---

## BibTeX Citation

If you use this computational pipeline, dataset, or methodology in your research, please cite this dissertation:

```bibtex
@mastersthesis{hamid2026advertising,
  author       = {Aliza Hamid},
  title        = {Advertising and Influence Analysis via Large Language Model--Generated Knowledge Graphs from {YouTube}: A Scalable Computational Framework for Latent Commercial Relationship Extraction, Topological Centrality Modeling, Modularity-Based Sub-Niche Detection, and Information Diffusion Simulation},
  school       = {Universidad Carlos III de Madrid (UC3M)},
  year         = {2026},
  month        = {September},
  type         = {Master's Thesis},
  address      = {Madrid, Spain},
  url          = {https://github.com/lizz234/masters-thesis}
}
```

