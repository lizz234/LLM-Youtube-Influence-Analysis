"""
Knowledge Graph Extraction Module
Extracts latent brand-creator relationships, sponsorship disclosures, and sentiment
from YouTube transcripts and metadata using Google Gemini and Pydantic schemas.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Add project root to sys.path so the module can be executed directly as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    GEMINI_API_KEY,
    PRIMARY_GEMINI_MODEL,
    FALLBACK_GEMINI_MODEL,
    LLM_TEMPERATURE,
    MAX_TRANSCRIPT_CHARS,
    DEFAULT_GRAPH_DATA_PATH,
    PROCESSED_GRAPH_DATA_PATH,
    RAW_DATA_DIR,
    DEFAULT_SLEEP_SECONDS,
    MAX_RETRIES
)


class BrandMention(BaseModel):
    """Pydantic schema for individual extracted brand-creator relationships."""
    creator: str = Field(
        description="The canonical name of the YouTube content creator speaking."
    )
    brand: str = Field(
        description="The specific commercial brand, company, hardware sponsor, or software platform mentioned."
    )
    relation: Literal["promotes", "criticizes", "mentions"] = Field(
        description="The contextual sentiment/relationship: 'promotes' (sponsored, partnered, recommended), 'criticizes' (negative critique/warning), or 'mentions' (neutral/factual reference)."
    )
    product: Optional[str] = Field(
        default=None,
        description="The specific product, device, model, or service name if mentioned (e.g., 'iPhone 16 Pro', 'Framework 16')."
    )
    context_snippet: Optional[str] = Field(
        default=None,
        description="A brief phrase from the transcript confirming the relationship."
    )


class ExtractionResult(BaseModel):
    """Pydantic container for list of extracted brand relationships."""
    mentions: List[BrandMention] = Field(default_factory=list)


class KnowledgeGraphExtractor:
    """Orchestrates LLM entity and relation extraction over video data packages."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = PRIMARY_GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        if not self.api_key:
            print("[!] Warning: GEMINI_API_KEY is not set. LLM extraction will not be possible without an API key.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[!] Failed to initialize Gemini API client: {e}")
                self.client = None

    def construct_prompt(self, creator_name: str, transcript_text: str, video_title: str = "", top_comments: Optional[List[str]] = None) -> str:
        """Constructs an information-dense extraction prompt."""
        truncated_transcript = transcript_text[:MAX_TRANSCRIPT_CHARS]
        comments_context = ""
        if top_comments:
            clean_comments = [c.replace("\n", " ") for c in top_comments[:5]]
            comments_context = "\nTop Audience Comments:\n" + "\n".join([f"- {c}" for c in clean_comments])

        prompt = f"""
You are an expert computational linguist and social network data extractor specializing in digital marketing and YouTube sponsorship analysis.

Analyze the following YouTube video transcript and context. Extract all commercial brand sponsorships, hardware/software product placements, affiliate partnerships, and explicit endorsements made by the content creator.

Channel / Creator: "{creator_name}"
Video Title: "{video_title}"
{comments_context}

Transcript:
\"\"\"{truncated_transcript}\"\"\"

Extraction Guidelines:
1. Canonicalize the Creator: Use the canonical creator name provided ("{creator_name}") for every record.
2. Canonicalize the Brand: Use standardized brand names (e.g., use "Apple" instead of "Apple's new iPad", "Intel" instead of "Intel Core Ultra", "dbrand" instead of "dbrand skin", "Framework" instead of "Framework Laptop").
3. Classify Relation:
   - "promotes": Explicit sponsor, paid promotion, affiliate link, free review sample praised, or strong endorsement.
   - "criticizes": Flawed product, negative review, anti-consumer warning, broken hardware critique.
   - "mentions": Factual benchmark comparison, competitor mention, neutral industry reference.
4. Output strictly according to the requested JSON schema.
"""
        return prompt.strip()

    def extract_from_video_data(self, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts brand-creator relationships from a single video data package."""
        if not self.client:
            print("[!] Gemini client not initialized. Skipping extraction.")
            return []

        vid = video_data.get("video_id", "unknown")
        creator = video_data.get("channel", "Unknown Creator")
        transcript = video_data.get("transcript", "")
        views = video_data.get("views", 0)
        metadata = video_data.get("metadata", {})
        title = metadata.get("title", "")

        if not transcript or len(transcript.strip()) < 50:
            print(f"  [-] Skipping {vid}: Transcript is empty or too short.")
            return []

        prompt = self.construct_prompt(creator, transcript, video_title=title)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractionResult,
                        temperature=LLM_TEMPERATURE
                    )
                )

                # Parse JSON
                parsed_json = json.loads(response.text)
                mentions = parsed_json.get("mentions", [])

                enriched_records = []
                for m in mentions:
                    rec = {
                        "creator": m.get("creator") or creator,
                        "brand": m.get("brand", "").strip(),
                        "relation": m.get("relation", "mentions"),
                        "product": m.get("product"),
                        "context_snippet": m.get("context_snippet"),
                        "video_id": vid,
                        "video_title": title,
                        "views": int(views) if str(views).isdigit() else 0,
                        "published_at": metadata.get("published_at", "")
                    }
                    if rec["brand"]:
                        enriched_records.append(rec)

                print(f"  [+] Success for {vid} ({creator}): Extracted {len(enriched_records)} relationships.")
                return enriched_records

            except Exception as e:
                print(f"  [-] LLM attempt {attempt+1}/{MAX_RETRIES} failed for {vid}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(DEFAULT_SLEEP_SECONDS * (attempt + 1))
        return []

    def extract_from_raw_cache(self, raw_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Scans the local raw video cache directory and extracts relationships for all cached videos."""
        raw_dir = raw_dir or RAW_DATA_DIR
        video_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        print(f"\n=======================================================")
        print(f"STARTING LLM KNOWLEDGE GRAPH EXTRACTION")
        print(f"Found {len(video_dirs)} cached video packages in {raw_dir}")
        print(f"=======================================================\n")

        all_records = []
        for i, vdir in enumerate(video_dirs, 1):
            vid = vdir.name
            meta_path = vdir / "metadata.json"
            trans_path = vdir / "transcript.txt"

            if not meta_path.exists() or not trans_path.exists():
                continue

            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            with open(trans_path, 'r', encoding='utf-8') as f:
                transcript = f.read()

            video_data = {
                "video_id": vid,
                "metadata": metadata,
                "transcript": transcript,
                "channel": metadata.get("channel_title", "Unknown Creator"),
                "views": metadata.get("views", 0)
            }

            print(f"[{i}/{len(video_dirs)}] Processing video {vid} - '{metadata.get('channel_title')}'...")
            records = self.extract_from_video_data(video_data)
            all_records.extend(records)
            time.sleep(3) # Polite delay between LLM calls

        return all_records

    def save_dataset(self, records: List[Dict[str, Any]], output_path: Optional[Path] = None):
        """Saves extracted relationships to standardized JSON locations."""
        out_path = output_path or DEFAULT_GRAPH_DATA_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        # Also write to processed dir if different
        if out_path != PROCESSED_GRAPH_DATA_PATH:
            with open(PROCESSED_GRAPH_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"\n[+] Master Knowledge Graph dataset saved! ({len(records)} relationships written to {out_path})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Knowledge Graph Extraction Module")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    extractor = KnowledgeGraphExtractor()
    records = extractor.extract_from_raw_cache(Path(args.raw_dir) if args.raw_dir else None)
    extractor.save_dataset(records, Path(args.output) if args.output else None)
