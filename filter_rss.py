"""
RSS Feed Filter Script
======================
OpenAI / Anthropic の公式RSSフィードから24時間以内の記事を取得し、
Gemini APIを用いてフィルタリングした上で選別済みRSSを出力する。
"""

import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime, formatdate
import re
import sys


# ── 設定 ──────────────────────────────────────────────
FEEDS = [
    {
        "name": "OpenAI-Selection",
        "source_url": "https://openai.com/news/rss.xml",
        "prompt_file": "openai-prompt.txt",
        "output_file": "openai-selection.xml",
        "feed_title": "OpenAI-Selection",
        "feed_link": "https://openai.com/news",
        "feed_description": "OpenAI News から選別された記事フィード",
    },
    {
        "name": "Anthropic-Selection",
        "source_url": "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
        "prompt_file": "anthropic-prompt.txt",
        "output_file": "anthropic-selection.xml",
        "feed_title": "Anthropic-Selection",
        "feed_link": "https://www.anthropic.com/news",
        "feed_description": "Anthropic News から選別された記事フィード",
    },
    {
        "name": "Google-Selection",
        "source_url": "https://blog.google/rss/",
        "prompt_file": "google-prompt.txt",
        "output_file": "google-selection.xml",
        "feed_title": "Google-Selection",
        "feed_link": "https://blog.google/",
        "feed_description": "Google Blog から選別された記事フィード",
    },
]

GEMINI_MODEL = "gemini-2.5-flash"
HOURS_WINDOW = 24


# ── RSS 取得 ──────────────────────────────────────────
def fetch_rss(url: str) -> ET.Element:
    """URLからRSSフィードを取得しXMLルート要素を返す。"""
    print(f"  Fetching RSS: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "rss-filter-bot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return ET.fromstring(raw)


# ── 24時間以内の記事抽出 ───────────────────────────────
def filter_recent_items(root: ET.Element, hours: int = HOURS_WINDOW) -> list[ET.Element]:
    """RSSアイテムから指定時間以内のものだけを返す。"""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    recent = []

    # 名前空間を考慮してitemを探す
    channel = root.find("channel")
    if channel is None:
        print("  Warning: <channel> not found in RSS")
        return recent

    for item in channel.findall("item"):
        pub_date_el = item.find("pubDate")
        if pub_date_el is None or not pub_date_el.text:
            # pubDateがなければ残す（疑わしきは残す）
            recent.append(item)
            continue
        try:
            pub_dt = parsedate_to_datetime(pub_date_el.text)
            # timezone-naive の場合 UTC と仮定
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            if pub_dt >= cutoff:
                recent.append(item)
        except Exception as e:
            # パースに失敗した記事は残す
            print(f"  Warning: Could not parse pubDate '{pub_date_el.text}': {e}")
            recent.append(item)

    return recent


# ── アイテムからテキスト情報を抽出 ──────────────────────
def extract_item_info(item: ET.Element) -> dict:
    """RSS item要素からタイトルとdescriptionを取り出す。"""
    title_el = item.find("title")
    desc_el = item.find("description")
    return {
        "title": (title_el.text or "").strip() if title_el is not None else "",
        "description": (desc_el.text or "").strip() if desc_el is not None else "",
    }


# ── Gemini API 呼び出し ──────────────────────────────
def call_gemini(prompt: str, api_key: str) -> str:
    """Gemini APIにプロンプトを送信し、レスポンステキストを返す。"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,  # 判定の安定性のため低め
        },
    }
    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
    return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ── LLM フィルタリング ────────────────────────────────
def filter_with_llm(items: list[ET.Element], prompt_file: str, api_key: str) -> list[ET.Element]:
    """LLMを使って記事をフィルタリングする。"""
    if not items:
        return items

    # プロンプトファイルを読み込み
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    # 記事リストをプロンプトに追加
    articles_text = ""
    item_info_list = []
    for i, item in enumerate(items):
        info = extract_item_info(item)
        item_info_list.append(info)
        articles_text += f"\n{i + 1}. タイトル: {info['title']}\n   説明: {info['description']}\n"

    full_prompt = base_prompt + articles_text

    print(f"  Sending {len(items)} articles to Gemini API for filtering...")

    try:
        response_text = call_gemini(full_prompt, api_key)

        # JSONを抽出（```json ... ``` で囲まれている場合にも対応）
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match is None:
            print(f"  Warning: Could not extract JSON from LLM response. Keeping all articles.")
            print(f"  LLM Response: {response_text[:500]}")
            return items

        judgments = json.loads(json_match.group())

        # 判定結果を適用
        filtered = []
        for i, item in enumerate(items):
            info = extract_item_info(item)
            # タイトルで判定結果をマッチング
            included = True  # デフォルトは残す
            matched_judgment = None
            for j in judgments:
                if j.get("title", "").strip() == info["title"]:
                    included = j.get("include", True)
                    reason = j.get("reason", "")
                    status = "KEEP" if included else "EXCLUDE"
                    print(f"    [{status}] {info['title']}")
                    if reason:
                        print(f"           Reason: {reason}")
                    matched_judgment = j
                    break
            else:
                # マッチする判定がなければ残す
                print(f"    [KEEP - no match] {info['title']}")

            if included:
                if matched_judgment:
                    translated_title = matched_judgment.get("translated_title", "").strip()
                    if translated_title and translated_title != info["title"]:
                        title_el = item.find("title")
                        if title_el is not None:
                            print(f"    [TRANSLATE] {info['title']} -> {translated_title}")
                            title_el.text = translated_title
                filtered.append(item)

        print(f"  Filtering result: {len(items)} -> {len(filtered)} articles")
        return filtered

    except Exception as e:
        print(f"  Error during LLM filtering: {e}")
        print("  Keeping all articles (fail-safe: 疑わしきは残す)")
        return items


# ── RSS XML 生成 ──────────────────────────────────────
def generate_rss_xml(
    items: list[ET.Element],
    title: str,
    link: str,
    description: str,
    output_file: str,
) -> None:
    """フィルタリング済みアイテムからRSS 2.0 XMLを生成して保存する。"""
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    title_el = ET.SubElement(channel, "title")
    title_el.text = title

    link_el = ET.SubElement(channel, "link")
    link_el.text = link

    desc_el = ET.SubElement(channel, "description")
    desc_el.text = description

    lang_el = ET.SubElement(channel, "language")
    lang_el.text = "en"

    last_build = ET.SubElement(channel, "lastBuildDate")
    last_build.text = formatdate(timeval=None, localtime=False, usegmt=True)

    # アイテムを追加
    for item in items:
        channel.append(item)

    tree = ET.ElementTree(rss)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass  # Python 3.8 以前では ET.indent が使えない

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"  Wrote {len(items)} articles to {output_file}")


# ── メイン処理 ────────────────────────────────────────
def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("Warning: GEMINI_API_KEY not set. LLM filtering will be skipped.")

    for feed_config in FEEDS:
        print(f"\n{'=' * 60}")
        print(f"Processing: {feed_config['name']}")
        print(f"{'=' * 60}")

        # 1. RSS取得
        try:
            root = fetch_rss(feed_config["source_url"])
        except Exception as e:
            print(f"  Error fetching RSS: {e}")
            print(f"  Skipping {feed_config['name']}")
            # エラー時は空のRSSを出力
            generate_rss_xml(
                items=[],
                title=feed_config["feed_title"],
                link=feed_config["feed_link"],
                description=feed_config["feed_description"],
                output_file=feed_config["output_file"],
            )
            continue

        # 2. 24時間以内の記事を抽出
        recent_items = filter_recent_items(root, HOURS_WINDOW)
        print(f"  Found {len(recent_items)} articles within the last {HOURS_WINDOW} hours")

        # 3. LLMフィルタリング
        if recent_items and api_key:
            filtered_items = filter_with_llm(
                recent_items, feed_config["prompt_file"], api_key
            )
        else:
            if not api_key and recent_items:
                print("  Skipping LLM filter (no API key)")
            filtered_items = recent_items

        # 4. RSS XML 出力
        generate_rss_xml(
            items=filtered_items,
            title=feed_config["feed_title"],
            link=feed_config["feed_link"],
            description=feed_config["feed_description"],
            output_file=feed_config["output_file"],
        )

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
