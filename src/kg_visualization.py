"""
Knowledge Graph Visualization & Publication Graphics Module
Generates interactive HTML visualizations via PyVis and high-resolution 300 DPI
academic figures for Master's thesis defense and publication.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network

# Add project root to sys.path so the module can be executed directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_GRAPH_DATA_PATH,
    DEFAULT_HTML_OUTPUT_PATH,
    METRICS_OUTPUT_PATH,
    COMMUNITIES_OUTPUT_PATH,
    DIFFUSION_OUTPUT_PATH,
    EVALUATION_OUTPUT_PATH,
    FIGURES_DIR
)
from src.network_analysis import NetworkAnalyzer


class KnowledgeGraphVisualizer:
    """Renders interactive HTML graphs and publication-quality network diagrams."""

    def __init__(self, analyzer: Optional[NetworkAnalyzer] = None):
        self.analyzer = analyzer or NetworkAnalyzer()
        self.G = self.analyzer.G
        self.undirected_G = self.analyzer.undirected_G

    def generate_interactive_html(self, output_path: Optional[Path] = None) -> Path:
        """Constructs a responsive Vis.js / PyVis interactive HTML graph."""
        out_path = Path(output_path) if output_path else DEFAULT_HTML_OUTPUT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.G.number_of_nodes() == 0:
            print("[!] Graph has no nodes. Cannot generate visualization.")
            return out_path

        # Run or load metrics
        if METRICS_OUTPUT_PATH.exists():
            with open(METRICS_OUTPUT_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        else:
            metrics_data = self.analyzer.compute_all_metrics()

        node_metrics = metrics_data.get("node_level_metrics", {})

        # PyVis Network setup
        net = Network(
            height="850px",
            width="100%",
            bgcolor="#111827",
            font_color="#F3F4F6",
            directed=True,
            select_menu=True,
            filter_menu=True
        )

        # Configure physics for smooth rendering of scale-free networks
        net.set_options("""
        {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": true,
            "font": {
              "face": "Segoe UI, Arial, sans-serif",
              "size": 14,
              "strokeWidth": 2,
              "strokeColor": "#111827"
            }
          },
          "edges": {
            "color": {
              "color": "rgba(156, 163, 175, 0.4)",
              "highlight": "#38BDF8",
              "hover": "#60A5FA",
              "inherit": false
            },
            "smooth": {
              "type": "continuous",
              "roundness": 0.2
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.6
              }
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08,
              "damping": 0.6
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
              "iterations": 150
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "tooltipDelay": 100
          }
        }
        """)

        # Add Nodes
        for node, attrs in self.G.nodes(data=True):
            node_type = attrs.get("node_type", "brand")
            metrics = node_metrics.get(node, {})
            eigen = metrics.get("eigenvector", 0.01)
            deg_views = metrics.get("weighted_degree", 1000)

            if node_type == "creator":
                color = "#3B82F6" # Vibrant Blue
                shape = "dot"
                # Size proportional to influence
                size = max(18, min(65, int(15 + eigen * 150)))
                title = f"""
                <div style='font-family: sans-serif; font-size: 13px; line-height: 1.4; min-width: 180px;'>
                    <b style='color: #60A5FA; font-size: 14px;'>📺 {node}</b><br>
                    <span style='color: #9CA3AF;'>Type: Content Creator</span><br>
                    <hr style='border: 0; border-top: 1px solid #374151; margin: 4px 0;'>
                    <b>Eigenvector Centrality:</b> {eigen:.4f}<br>
                    <b>Betweenness:</b> {metrics.get('betweenness', 0):.4f}<br>
                    <b>Total Mentions Given:</b> {self.G.out_degree(node)}<br>
                    <b>Associated Reach:</b> {deg_views:,} views
                </div>
                """
            else:
                color = "#10B981" # Emerald Green
                shape = "dot"
                in_deg = self.G.in_degree(node)
                size = max(12, min(50, int(10 + math.log10(max(1, deg_views)) * 4 + in_deg * 2)))
                title = f"""
                <div style='font-family: sans-serif; font-size: 13px; line-height: 1.4; min-width: 180px;'>
                    <b style='color: #34D399; font-size: 14px;'>🏢 {node}</b><br>
                    <span style='color: #9CA3AF;'>Type: Commercial Brand / Product</span><br>
                    <hr style='border: 0; border-top: 1px solid #374151; margin: 4px 0;'>
                    <b>In-Degree (Creators):</b> {in_deg}<br>
                    <b>Total Exposure Views:</b> {deg_views:,}<br>
                    <b>PageRank Score:</b> {metrics.get('pagerank', 0):.5f}
                </div>
                """

            net.add_node(
                node,
                label=node,
                color=color,
                size=size,
                shape=shape,
                title=title
            )

        # Add Edges
        for u, v, edge_attrs in self.G.edges(data=True):
            raw_views = edge_attrs.get("raw_views", 1000)
            relations = ", ".join(edge_attrs.get("relations", ["mentions"]))
            count = edge_attrs.get("count", 1)

            # Edge width scaled logarithmically
            width = max(1.0, min(8.0, 1.0 + math.log10(max(1, raw_views / 10000))))

            edge_title = f"""
            <div style='font-family: sans-serif; font-size: 12px; line-height: 1.4;'>
                <b>{u}</b> ➔ <b>{v}</b><br>
                <b>Relation Context:</b> {relations}<br>
                <b>Mention Count:</b> {count}<br>
                <b>Estimated Video Reach:</b> {raw_views:,} views
            </div>
            """

            net.add_edge(
                u,
                v,
                value=width,
                title=edge_title,
                color="rgba(156, 163, 175, 0.4)"
            )

        # Save HTML
        net.save_graph(str(out_path))
        print(f"[+] Interactive PyVis network visualization saved to {out_path}")
        return out_path

    def generate_publication_figures(self) -> Dict[str, Path]:
        """Generates publication-quality 300 DPI figures for the thesis dissertation."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        generated_figures = {}

        # Set academic styling
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300
        })

        # Figure 1: Network Topology Overview
        fig1_path = FIGURES_DIR / "network_topology_overview.png"
        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(self.undirected_G, k=0.35, seed=42, iterations=80)
        
        node_colors = ['#3B82F6' if self.G.nodes[n].get('node_type') == 'creator' else '#10B981' for n in self.G.nodes()]
        node_sizes = [min(2000, 300 + self.G.degree(n) * 80) for n in self.G.nodes()]

        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.88)
        nx.draw_networkx_edges(self.G, pos, edge_color='#9CA3AF', alpha=0.35, arrows=True, arrowsize=10)

        # Label top creators and prominent brands
        top_labels = {}
        for n in self.G.nodes():
            if self.G.nodes[n].get('node_type') == 'creator' or self.G.degree(n) >= 3:
                top_labels[n] = n

        nx.draw_networkx_labels(self.G, pos, labels=top_labels, font_size=8, font_weight='bold', font_color='#1F2937')

        plt.title("Master Knowledge Graph Topology: YouTube Creator-Brand Sponsorship Network", pad=15, fontweight='bold')
        plt.axis('off')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Content Creator ($V_C$)', markerfacecolor='#3B82F6', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Commercial Brand ($V_B$)', markerfacecolor='#10B981', markersize=12)
        ]
        plt.legend(handles=legend_elements, loc='upper right', frameon=True)
        plt.tight_layout()
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_figures["topology"] = fig1_path

        # Figure 2: Centrality Metrics Comparative Rankings
        fig2_path = FIGURES_DIR / "centrality_rankings.png"
        metrics = self.analyzer.compute_all_metrics()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Subplot A: Top Brands by In-Degree (Mentions/Views)
        top_brands = metrics.get("top_brands_by_in_degree", [])[:7]
        if top_brands:
            b_names = [b["node"] for b in top_brands][::-1]
            b_scores = [b["score"] / 1e6 for b in top_brands][::-1] # in Millions
            axes[0].barh(b_names, b_scores, color='#10B981', edgecolor='#047857')
            axes[0].set_title("Top Brands by Exposure Volume", fontweight='bold')
            axes[0].set_xlabel("Aggregated Video Views (Millions)")
            axes[0].grid(axis='x', linestyle='--', alpha=0.7)

        # Subplot B: Top Creators by Eigenvector Centrality
        top_eigen = [item for item in metrics.get("top_eigenvector_influencers", []) if item.get("type") == "creator"][:7]
        if not top_eigen:
            top_eigen = metrics.get("top_eigenvector_influencers", [])[:7]
        if top_eigen:
            e_names = [e["node"] for e in top_eigen][::-1]
            e_scores = [e["score"] for e in top_eigen][::-1]
            axes[1].barh(e_names, e_scores, color='#3B82F6', edgecolor='#1D4ED8')
            axes[1].set_title("Top Influencers by Eigenvector Prestige", fontweight='bold')
            axes[1].set_xlabel(r"Eigenvector Centrality Score ($\lambda$)")
            axes[1].grid(axis='x', linestyle='--', alpha=0.7)

        # Subplot C: Top Betweenness Bridge Nodes
        top_between = metrics.get("top_betweenness_bridges", [])[:7]
        if top_between:
            bw_names = [bw["node"] for bw in top_between][::-1]
            bw_scores = [bw["score"] for bw in top_between][::-1]
            axes[2].barh(bw_names, bw_scores, color='#8B5CF6', edgecolor='#6D28D9')
            axes[2].set_title("Top Structural Bridges (Betweenness)", fontweight='bold')
            axes[2].set_xlabel("Betweenness Centrality ($C_B$)")
            axes[2].grid(axis='x', linestyle='--', alpha=0.7)

        plt.suptitle("Topological Centrality Analysis across Scale-Diverse YouTube Graph", fontsize=14, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_figures["centrality"] = fig2_path

        # Figure 3: Information Diffusion Simulation (ICM)
        fig3_path = FIGURES_DIR / "diffusion_cascade_curves.png"
        diffusion_raw = self.analyzer.simulate_information_diffusion()
        diffusion_res = diffusion_raw.get("results", diffusion_raw)
        plt.figure(figsize=(9.5, 5.5), dpi=300)
        
        colors = {
            'mega_hub_strategy': '#2563EB',
            'distributed_mid_tier_strategy': '#DC2626',
            'deep_niche_strategy': '#059669'
        }
        labels = {
            'mega_hub_strategy': 'Strategy A: Mega-Hub (Marques Brownlee)',
            'distributed_mid_tier_strategy': 'Strategy B: Distributed Mid-Tier (Dave2D + JayzTwoCents)',
            'deep_niche_strategy': 'Strategy C: Deep-Niche (Gamers Nexus)'
        }

        for strat_key in ['mega_hub_strategy', 'distributed_mid_tier_strategy', 'deep_niche_strategy']:
            data = diffusion_res.get(strat_key, {})
            curve = data.get("cascade_curve", [])
            steps = list(range(len(curve)))
            lbl = labels.get(strat_key, strat_key)
            reach = data.get('mean_final_reach', 0)
            pct = data.get('reach_percentage', 0)
            plt.plot(steps, curve, marker='o', linewidth=2.5, color=colors.get(strat_key, '#6B7280'),
                     label=f"{lbl} (Final: {reach:.1f} nodes / {pct:.1f}%)")

        plt.title("Information Diffusion Dynamics: Independent Cascade Model (ICM)", pad=12, fontweight='bold', fontsize=13)
        plt.xlabel("Diffusion Time Step ($t$)", fontsize=11.5)
        plt.ylabel("Cumulative Activated Nodes ($|A_t|$)", fontsize=11.5)
        plt.xticks(range(7), [f"$t={t}$" for t in range(7)], fontsize=10.5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_figures["diffusion"] = fig3_path

        # Figure 4: Extraction Fidelity Benchmark (LLM vs Rule-Based Baseline)
        fig4_path = FIGURES_DIR / "extraction_fidelity_benchmark.png"
        eval_data = self.analyzer.benchmark_extraction_fidelity()
        plt.figure(figsize=(8, 5))

        categories = ['Precision', 'Recall', 'F1-Score']
        llm_scores = [
            eval_data['llm_gemini_2_5']['precision'],
            eval_data['llm_gemini_2_5']['recall'],
            eval_data['llm_gemini_2_5']['f1_score']
        ]
        baseline_scores = [
            eval_data['rule_based_keyword_baseline']['precision'],
            eval_data['rule_based_keyword_baseline']['recall'],
            eval_data['rule_based_keyword_baseline']['f1_score']
        ]

        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width/2, llm_scores, width, label='Gemini 2.5 Flash (Pydantic Schema)', color='#3B82F6', edgecolor='#1D4ED8')
        plt.bar(x + width/2, baseline_scores, width, label='Rule-Based / Keyword Baseline', color='#9CA3AF', edgecolor='#4B5563')

        for i in range(len(categories)):
            plt.text(x[i] - width/2, llm_scores[i] + 0.02, f"{llm_scores[i]:.2f}", ha='center', fontweight='bold', fontsize=10)
            plt.text(x[i] + width/2, baseline_scores[i] + 0.02, f"{baseline_scores[i]:.2f}", ha='center', fontweight='bold', fontsize=10)

        plt.ylim(0, 1.15)
        plt.ylabel("Performance Metric")
        plt.title("Entity & Relation Extraction Fidelity: LLM vs Baseline Benchmark", pad=12, fontweight='bold')
        plt.xticks(x, categories, fontweight='bold')
        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()
        plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_figures["evaluation"] = fig4_path

        print(f"[+] Publication-grade figures successfully generated in {FIGURES_DIR}")
        return generated_figures

    def run_full_visualization_suite(self) -> Dict[str, Any]:
        """Builds both interactive HTML and all publication figures."""
        html_path = self.generate_interactive_html()
        figures = self.generate_publication_figures()
        return {
            "html": html_path,
            "figures": figures
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge Graph Visualization Module")
    parser.add_argument("--html-output", type=str, default=None, help="Output path for interactive HTML")
    args = parser.parse_args()

    visualizer = KnowledgeGraphVisualizer()
    visualizer.run_full_visualization_suite()
