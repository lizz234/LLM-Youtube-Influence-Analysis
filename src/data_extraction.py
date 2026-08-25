"""
Data Extraction Module
Fetches YouTube video metadata, engagement statistics, top comments, reactions,
and automated speech recognition (ASR) transcripts with caching and rate limiting.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from src.config import (
    YOUTUBE_API_KEY,
    RAW_DATA_DIR,
    SCALE_DIVERSE_CHANNELS,
    DEFAULT_SLEEP_SECONDS,
    MAX_RETRIES,
    BACKOFF_FACTOR
)


class YouTubeDataExtractor:
    """Manages robust data extraction from YouTube Data API v3 and YouTube Transcript API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or YOUTUBE_API_KEY
        if not self.api_key:
            print("[!] Warning: YOUTUBE_API_KEY is not set. Data fetching will rely strictly on local cache.")
            self.youtube_client = None
        else:
            try:
                self.youtube_client = build('youtube', 'v3', developerKey=self.api_key)
            except Exception as e:
                print(f"[!] Failed to initialize YouTube API client: {e}")
                self.youtube_client = None

    @staticmethod
    def extract_video_id(url_or_id: str) -> str:
        """Extracts standard 11-character video ID from raw ID or URL formats."""
        url_or_id = url_or_id.strip()
        if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
            return url_or_id
        
        # Regex for YouTube URLs
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'youtu\.be\/([0-9A-Za-z_-]{11})',
            r'embed\/([0-9A-Za-z_-]{11})',
            r'shorts\/([0-9A-Za-z_-]{11})'
        ]
        for p in patterns:
            match = re.search(p, url_or_id)
            if match:
                return match.group(1)
        return url_or_id

    def get_video_dir(self, video_id: str) -> Path:
        """Returns the cache directory for a specific video ID."""
        video_dir = RAW_DATA_DIR / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def is_cached(self, video_id: str) -> bool:
        """Checks if metadata and transcript already exist in the video cache directory."""
        vdir = self.get_video_dir(video_id)
        metadata_file = vdir / "metadata.json"
        transcript_file = vdir / "transcript.txt"
        return metadata_file.exists() and transcript_file.exists()

    def fetch_channel_videos(self, channel_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetches the latest videos from a YouTube channel's uploads playlist."""
        if not self.youtube_client:
            print("[!] Cannot fetch channel videos: YouTube API client is not initialized.")
            return []

        print(f"[*] Querying channel: {channel_id} (Limit: {max_results} videos)...")
        for attempt in range(MAX_RETRIES):
            try:
                channel_req = self.youtube_client.channels().list(
                    part="contentDetails,snippet",
                    id=channel_id
                )
                channel_resp = channel_req.execute()
                items = channel_resp.get('items', [])
                if not items:
                    print(f"  [-] No channel found for ID: {channel_id}")
                    return []

                uploads_playlist_id = items[0]['contentDetails']['relatedPlaylists']['uploads']
                channel_title = items[0]['snippet']['title']

                # Fetch playlist items
                playlist_req = self.youtube_client.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=max_results
                )
                playlist_resp = playlist_req.execute()

                videos = []
                for p_item in playlist_resp.get('items', []):
                    vid_id = p_item['snippet']['resourceId']['videoId']
                    title = p_item['snippet']['title']
                    pub_at = p_item['snippet']['publishedAt']
                    videos.append({
                        "video_id": vid_id,
                        "title": title,
                        "channel_id": channel_id,
                        "channel_title": channel_title,
                        "published_at": pub_at
                    })
                print(f"  [+] Found {len(videos)} videos for channel '{channel_title}'.")
                return videos
            except HttpError as he:
                print(f"  [-] HTTP error on attempt {attempt+1}/{MAX_RETRIES}: {he}")
                time.sleep(DEFAULT_SLEEP_SECONDS * (BACKOFF_FACTOR ** attempt))
            except Exception as e:
                print(f"  [-] Error querying channel {channel_id}: {e}")
                time.sleep(DEFAULT_SLEEP_SECONDS)
        return []

    def fetch_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Fetches detailed statistics, title, description, and tags for a video."""
        if not self.youtube_client:
            return {"video_id": video_id, "views": 0, "likes": 0, "comment_count": 0}

        try:
            req = self.youtube_client.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            res = req.execute()
            items = res.get('items', [])
            if not items:
                return {"video_id": video_id, "views": 0, "likes": 0, "comment_count": 0}

            item = items[0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})

            metadata = {
                "video_id": video_id,
                "title": snippet.get('title', ''),
                "channel_id": snippet.get('channelId', ''),
                "channel_title": snippet.get('channelTitle', ''),
                "published_at": snippet.get('publishedAt', ''),
                "description": snippet.get('description', ''),
                "tags": snippet.get('tags', []),
                "views": int(stats.get('viewCount', 0)),
                "likes": int(stats.get('likeCount', 0)),
                "comment_count": int(stats.get('commentCount', 0)),
                "duration": item.get('contentDetails', {}).get('duration', '')
            }
            return metadata
        except Exception as e:
            print(f"  [-] Failed to fetch metadata for {video_id}: {e}")
            return {"video_id": video_id, "views": 0, "likes": 0, "comment_count": 0}

    def fetch_video_transcript(self, video_id: str) -> Optional[str]:
        """Fetches automated captions or manual subtitles for a video."""
        try:
            # Try instance method or static fallback
            try:
                ytt = YouTubeTranscriptApi()
                transcript_data = ytt.fetch(video_id)
                full_text = " ".join([snippet.text for snippet in transcript_data])
            except AttributeError:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([entry['text'] for entry in transcript_data])
            return full_text
        except (TranscriptsDisabled, NoTranscriptFound) as ne:
            print(f"  [-] No transcript available for {video_id} ({ne})")
            return None
        except Exception as e:
            print(f"  [-] Error downloading transcript for {video_id}: {e}")
            return None

    def fetch_video_comments(self, video_id: str, max_comments: int = 20) -> List[Dict[str, Any]]:
        """Fetches top comments and audience reaction signals."""
        if not self.youtube_client:
            return []

        comments = []
        try:
            req = self.youtube_client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_comments, 100),
                textFormat="plainText",
                order="relevance"
            )
            res = req.execute()
            for item in res.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    "author": top_comment.get('authorDisplayName', ''),
                    "text": top_comment.get('textDisplay', ''),
                    "likes": top_comment.get('likeCount', 0),
                    "published_at": top_comment.get('publishedAt', '')
                })
        except HttpError as he:
            # Comments might be disabled
            pass
        except Exception as e:
            pass
        return comments

    def process_single_video(self, video_id: str, channel_hint: Optional[str] = None, skip_cached: bool = True) -> Dict[str, Any]:
        """Extracts and caches metadata, transcript, comments, and reactions for a single video."""
        vid = self.extract_video_id(video_id)
        vdir = self.get_video_dir(vid)

        meta_path = vdir / "metadata.json"
        trans_path = vdir / "transcript.txt"
        comm_path = vdir / "comments.json"
        react_path = vdir / "reactions.json"

        if skip_cached and self.is_cached(vid):
            print(f"  [>] Video {vid} already cached in {vdir}. Skipping API calls.")
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            with open(trans_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            return {
                "video_id": vid,
                "metadata": metadata,
                "transcript": transcript,
                "channel": metadata.get('channel_title', channel_hint or 'Unknown Creator'),
                "views": metadata.get('views', 0)
            }

        print(f"[*] Ingesting video: {vid}...")
        metadata = self.fetch_video_metadata(vid)
        if channel_hint and not metadata.get('channel_title'):
            metadata['channel_title'] = channel_hint

        transcript = self.fetch_video_transcript(vid) or ""
        comments = self.fetch_video_comments(vid, max_comments=20)
        reactions = {
            "views": metadata.get('views', 0),
            "likes": metadata.get('likes', 0),
            "comment_count": metadata.get('comment_count', 0),
            "top_comment_count": len(comments)
        }

        # Cache to disk
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        with open(trans_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        with open(comm_path, 'w', encoding='utf-8') as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
        with open(react_path, 'w', encoding='utf-8') as f:
            json.dump(reactions, f, indent=2, ensure_ascii=False)

        print(f"  [+] Saved data package for {vid} (Transcript length: {len(transcript)} chars, Views: {metadata.get('views', 0):,})")
        return {
            "video_id": vid,
            "metadata": metadata,
            "transcript": transcript,
            "channel": metadata.get('channel_title', channel_hint or 'Unknown Creator'),
            "views": metadata.get('views', 0)
        }

    def process_channel_batch(self, channels: Optional[List[Dict[str, Any]]] = None, limit_per_channel: int = 10, skip_cached: bool = True) -> List[Dict[str, Any]]:
        """Ingests videos across multiple scale-diverse channels."""
        channels = channels or SCALE_DIVERSE_CHANNELS
        all_videos_data = []

        print(f"\n=======================================================")
        print(f"STARTING SCALE-DIVERSE YOUTUBE INGESTION PIPELINE")
        print(f"Channels: {len(channels)} | Limit per channel: {limit_per_channel}")
        print(f"=======================================================\n")

        for ch in channels:
            ch_id = ch["id"]
            ch_name = ch.get("name", "Unknown")
            ch_tier = ch.get("tier", "Standard")
            print(f"\n--- Channel: {ch_name} [{ch_tier}] ({ch_id}) ---")
            
            videos = self.fetch_channel_videos(ch_id, max_results=limit_per_channel)
            for v in videos:
                vid_data = self.process_single_video(v["video_id"], channel_hint=ch_name, skip_cached=skip_cached)
                if vid_data.get("transcript"):
                    all_videos_data.append(vid_data)
                time.sleep(2) # Polite delay between videos

        print(f"\n[+] Ingestion Batch Complete! Successfully processed {len(all_videos_data)} video transcripts.")
        return all_videos_data

    def process_url_file(self, file_path: str, skip_cached: bool = True) -> List[Dict[str, Any]]:
        """Processes videos listed in a text file (one URL or ID per line)."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Input video file not found: {file_path}")

        with open(p, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith('#')]

        print(f"[*] Processing {len(lines)} video items from {file_path}...")
        results = []
        for line in lines:
            # Handle comments after URL
            raw_target = line.split('#')[0].strip()
            if not raw_target:
                continue
            vid_data = self.process_single_video(raw_target, skip_cached=skip_cached)
            if vid_data.get("transcript"):
                results.append(vid_data)
            time.sleep(1)
        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YouTube Data Ingestion Module")
    parser.add_argument("--channels-file", type=str, default=None, help="Path to channel IDs text file")
    parser.add_argument("--videos-file", type=str, default=None, help="Path to video URLs/IDs text file")
    parser.add_argument("--limit", type=int, default=10, help="Max videos per channel")
    parser.add_argument("--no-skip-cached", action="store_true", help="Force re-download cached videos")
    args = parser.parse_args()

    extractor = YouTubeDataExtractor()
    if args.videos_file:
        extractor.process_url_file(args.videos_file, skip_cached=not args.no_skip_cached)
    else:
        extractor.process_channel_batch(limit_per_channel=args.limit, skip_cached=not args.no_skip_cached)
