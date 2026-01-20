import string
import random
import math
import time
from collections import Counter
import itertools
import sys
import re
import requests
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

class TrieNode:
    def __init__(self):
        self.children = {}
        self.weight = 0
        self.is_end = False
        self.paper_ids = set()

class WeightedTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, words, paper_id):
        node = self.root
        for w in words:
            if w not in node.children:
                node.children[w] = TrieNode()
            node = node.children[w]
            node.weight += 1
            node.paper_ids.add(paper_id)
        node.is_end = True

    def update_weight(self, words, delta=1):
        node = self.root
        for w in words:
            if w in node.children:
                node = node.children[w]
                node.weight += delta

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w not in stop_words]

def to_dist(words):
    counts = Counter(words)
    total = sum(counts.values())
    return {w: c / total for w, c in counts.items()}

def kl(p, q, eps=1e-12):
    return sum(p[w] * math.log((p[w] + eps) / (q.get(w, eps))) for w in p)

def jensen_shannon(text1, text2):
    p = to_dist(preprocess(text1))
    q = to_dist(preprocess(text2))
    m = {w: 0.5 * (p.get(w,0) + q.get(w,0)) for w in set(p) | set(q)}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def spinner(message="Loading"):
    for char in itertools.cycle("|/-\\"):
        sys.stdout.write(f"\r{message} {char}")
        sys.stdout.flush()
        yield
        time.sleep(0.05)

def fetch_arxiv_paper(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        title_match = re.search(r"<title>(.*?)</title>", resp.text, re.DOTALL)
        summary_match = re.search(r"<summary>(.*?)</summary>", resp.text, re.DOTALL)
        category_match = re.findall(r'<category term="(.*?)"/>', resp.text)
        if summary_match:
            abstract = summary_match.group(1).strip()
            title = title_match.group(1).strip() if title_match else f"arXiv {arxiv_id}"
            categories = " ".join(category_match) if category_match else ""
            return {"id": arxiv_id, "title": title, "abstract": abstract, "categories": categories}
        return None
    except:
        return None

liked_papers = []
arxiv_links = input("Paste arXiv links you like, separated by commas, or press Enter to skip: ").strip()
if arxiv_links:
    for link in arxiv_links.split(","):
        match = re.search(r"arxiv\.org/abs/(\S+)", link.strip())
        if match:
            arxiv_id = match.group(1)
            paper = fetch_arxiv_paper(arxiv_id)
            if paper:
                liked_papers.append(paper)

ds = load_dataset("sleeping-ai/arxiv_metadata", split="train", streaming=True)
trie = WeightedTrie()
papers = []
categories_set = set()

choice = input("Use 1000 papers for speed or whole ArXiv? (1000/all): ").strip().lower()
limit = None if choice == "all" else 1000

spin = spinner("Building Trie")
for idx, paper in enumerate(ds):
    next(spin)
    if not paper["abstract"]:
        continue
    words = preprocess(paper["abstract"])
    trie.insert(words, idx)
    papers.append({"id": paper["id"], "title": paper["title"], "abstract": paper["abstract"], "categories": paper["categories"]})
    categories_set.update(paper["categories"].split())
    if limit and len(papers) >= limit:
        break
sys.stdout.write("\rTrie built!          \n")

# Determine categories based on user favorites
fav_categories = set()
for p in liked_papers:
    fav_categories.update(p["categories"].split())
if fav_categories:
    print(f"Showing papers from your favorite categories: {', '.join(fav_categories)}")
    filtered_papers = [p for p in papers if any(cat in p["categories"].split() for cat in fav_categories)]
else:
    category_input = input(f"Enter category to filter or press Enter for random: ").strip()
    if category_input == "":
        category_input = random.choice(list(categories_set))
    filtered_papers = [p for p in papers if category_input in p["categories"]]

def sample_candidate(last_liked=None):
    if not filtered_papers:
        return None
    if last_liked is None:
        return random.choice(filtered_papers)
    sims = [(p, jensen_shannon(last_liked["abstract"], p["abstract"])) for p in filtered_papers]
    sims.sort(key=lambda x: x[1])
    top_candidates = [p for p, s in sims[:50]]
    return random.choice(top_candidates)

def top_matches(liked_papers):
    sims = []
    for paper in filtered_papers:
        min_js = min(jensen_shannon(paper["abstract"], liked["abstract"]) for liked in liked_papers)
        sims.append((paper, min_js))
    sims.sort(key=lambda x: x[1])
    return [p for p, s in sims[:10]]

def main():
    last_liked = liked_papers[-1] if liked_papers else None
    swipe_count = 0
    while True:
        candidate = sample_candidate(last_liked)
        if candidate is None:
            print("No papers in this category.")
            break
        print("\nTitle:", candidate["title"])
        print("Abstract:", candidate["abstract"])
        action = input("a like, d reject, q quit: ").strip()
        words = preprocess(candidate["abstract"])
        if action == "a":
            trie.update_weight(words, 2)
            last_liked = candidate
            liked_papers.append(candidate)
            swipe_count += 1
        elif action == "d":
            trie.update_weight(words, -1)
            swipe_count += 1
        elif action == "q":
            break

        if swipe_count > 0 and swipe_count % 20 == 0 and liked_papers:
            print("\n=== Top 10 Matched Papers ===")
            matches = top_matches(liked_papers)
            for idx, m in enumerate(matches, 1):
                print(f"{idx}. {m['title']}")
            choice = input("Enter number to match, or Enter to skip: ").strip()
            if choice.isdigit():
                match_idx = int(choice) - 1
                if 0 <= match_idx < len(matches):
                    match = matches[match_idx]
                    print("\nYou matched with:")
                    print("Title:", match["title"])
                    print("Abstract:", match["abstract"])
                    print("ArXiv Link:", f"https://arxiv.org/abs/{match['id']}")
                    break

if __name__ == "__main__":
    main()
