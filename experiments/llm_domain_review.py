"""LLM domain review of statistically admitted (rule, category) pairs.

    export OPENAI_API_KEY=...
    NEGCAND_CATEGORY_NAMES=/path/to/category_names.json \
        python -m experiments.llm_domain_review --dataset kuairec

For every deduplicated (rule, category) pair that passes S+E+H+R in
artifacts/<dataset>/rules.csv, the script sends one request to an OpenAI chat
model (default gpt-4o-mini, temperature 0, JSON output). The request contains:

  * the rule rendered in plain language (no integer codes, no bare column names)
  * the category's published name and an English translation
  * the claim under review (KuaiRec: users in this state tend not to watch
    videos in this category to the end; KuaiRand: they tend not to click them)
  * definitions of the features the rule uses

It contains no statistic: no S, D, E, H, R, tier, count, rate or evaluation
result. A banned-token check on the serialized request enforces this before
any request is sent. The review judges only whether the claim is a plausible
audience statement; statistical support and product specificity are assessed
upstream and are not re-examined here.

Verdicts: BUSINESS_RELEVANT, NOT_RELEVANT or UNRESOLVED (the model was
uncertain, the request failed, or the category has no published name, in
which case no meaning is invented and no request is sent).

Category names. The JSON file maps the category id used in the rules to the
published first-level category name (`first_level_category_name` in KuaiRec's
kuairec_caption_category.csv; KuaiRand shares the same id space). It is not
distributed with the code. Default location: data/<dataset>/category_names.json.

The public experiments in the paper do not use this review.

Output: artifacts/<dataset>/domain_review.csv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from paths import ARTIFACTS, DATA

# Tokens that name statistics or evaluation results. Feature names are masked
# out before the check, so a rule built on a feature such as
# `mean_watch_ratio_train` can still be shown.
BANNED = ["tier1", "tier2", "tier3", "tier4", "qualified", "stage_a", "stage_b",
          "stage_c", "scope=", "evidence=", "e=", "h=", "r=", "s=", "d_median",
          "n_rule", "n_observable", "q_rule", "q_baseline",
          "precision", "coverage", "contamination", "pr-auc", "pr_auc",
          "roc-auc", "roc_auc", "log loss", "logloss", "p-value", "p_value",
          "replicate", "posterior", "percentile", "conversion rate",
          "base rate", "positive rate", "test set", "validation set"]

# Published Kuaishou first-level category names and an English translation of each.
GLOSS = {
    "舞蹈": "dance", "音乐": "music", "游戏": "gaming", "美妆": "make-up and beauty",
    "时尚": "fashion", "明星娱乐": "celebrity and entertainment news", "运动": "sports",
    "颜值": "good-looks / appearance showcase videos", "喜剧": "comedy and skits",
    "旅游": "travel", "生活": "everyday life", "美食": "food and cooking",
    "三农": "rural life and agriculture", "教育": "education and study",
    "艺术": "art", "健康": "health and medicine", "宠物": "pets",
    "汽车": "cars and motoring", "情感": "relationships and emotional advice",
    "二次元": "anime, comics and games (ACG)", "人文": "culture and humanities",
    "财经": "finance, business and investing", "时政资讯": "current affairs and political news",
    "星座命理": "astrology and fortune-telling", "亲子": "parenting and young children",
    "摄影": "photography", "高新数码": "consumer tech and gadgets",
    "民生资讯": "local livelihood news", "科学与法律": "science and law",
    "健身": "fitness and working out", "短剧": "short-form scripted drama",
    "自拍": "selfie videos", "其他": "other / uncategorised", "军事": "military",
    "房产家居": "property and home interiors", "奇人异象": "unusual people and oddities",
    "读书": "books and reading", "影视综": "film, TV and variety shows",
}

PLATFORM = (
    "Kuaishou is one of China's two largest short-video apps. Users scroll a "
    "recommendation feed of short videos, each of which belongs to one of the "
    "platform's content verticals (music, gaming, food, finance, parenting, and "
    "so on). The videos are typically 10-60 seconds long and loop."
)

CLAIM = {
    "kuairec": ("users in this state tend NOT to watch videos in this category "
                "through to their full length (they scroll away part-way)"),
    "kuairand": ("users in this state tend NOT to click on videos in this "
                 "category when the feed shows them one"),
}

PLATFORM_FEATURES = {
    "user_active_degree": 'the platform\'s own activity tier for this account. In this data it takes the values "full_active", "high_active", "middle_active" and "UNKNOWN".',
    "is_lowactive_period": "1 if the account is currently in a low-activity stretch, else 0.",
    "is_live_streamer": "1 if the account live-streams on the platform, else 0.",
    "is_video_author": "1 if the account publishes its own videos, else 0.",
    "follow_user_num": "how many other accounts this user follows.",
    "fans_user_num": "how many followers this user has.",
    "friend_user_num": "how many mutual-follow relationships this user has.",
    "register_days": "how many days ago this account registered.",
}
DERIVED_FEATURES = {
    "n_train_interactions": "how many videos this user was observed watching during the earlier observation window.",
    "n_train_positives": "how many of those videos they watched through to full length.",
    "train_positive_rate": "the share of the videos they watched that they watched through to full length -- an overall completion tendency, not specific to any category.",
    "n_distinct_categories": "how many different content categories this user watched during the observation window.",
    "mean_watch_ratio_train": "this user's average watch-through fraction across the videos they watched -- again an overall tendency, not category-specific.",
}

CRITERIA = """This gate is SEMANTIC ONLY. Whether the pattern is specific to this category
rather than generic has already been tested separately and upstream -- do not
re-litigate it here, and do not reject a state merely because it also describes
broadly lower engagement. Many real audience statements are of exactly that
shape.

Judge it NOT_RELEVANT only when:
  - the attributes in the state have no credible connection to content taste at
    all, so the pairing could not be explained to a colleague; or
  - the pairing is actively implausible -- it contradicts what you would expect
    of this audience and this kind of content.

Judge it BUSINESS_RELEVANT when the state and the category form a coherent
audience story a person could act on, including when the story runs through
general engagement level rather than a niche taste."""

SYSTEM_TMPL = """You are a product analyst on a short-video recommendation team.

%PLATFORM%

You will be shown a USER STATE (a simple rule over account attributes) and a
TARGET CONTENT CATEGORY. The claim under test is stated in the packet.

Decide whether that claim is a SEMANTICALLY PLAUSIBLE audience statement -- the
kind a content or growth team would accept as describing a real taste pattern.

%CRITERIA%

You are given NO statistics of any kind and must not ask for any. Judge only
semantics. Reply with strict JSON:
{"verdict": "BUSINESS_RELEVANT" | "NOT_RELEVANT" | "UNCERTAIN", "reason": "<one sentence>"}"""

PRETTY = {
    "follow_user_num": "the number of accounts this user follows",
    "fans_user_num": "this user's follower count",
    "friend_user_num": "this user's mutual-follow count",
    "register_days": "the account's age in days",
    "is_lowactive_period": "the low-activity-period flag",
    "is_live_streamer": "the live-streamer flag",
    "is_video_author": "the video-author flag",
    "n_train_interactions": "the number of videos this user watched in the observation window",
    "n_train_positives": "the number of videos they watched through to full length",
    "train_positive_rate": "their overall share of videos watched through to full length",
    "n_distinct_categories": "the number of distinct categories they watched",
    "mean_watch_ratio_train": "their average watch-through fraction",
}


def humanise(rule, codes):
    """Rule string -> plain language, with categorical codes replaced by their labels."""
    parts = []
    for p in rule.split(" AND "):
        m = re.match(r"^(.*?)(<=|==)(.*)$", p.strip())
        col, op, thr = m.group(1).strip(), m.group(2), float(m.group(3))
        if col.endswith("_code"):
            base = col[:-5]
            val = codes.get(base, {}).get(str(int(thr)), f"code {int(thr)}")
            parts.append(f'the account\'s platform activity level is "{val}"')
            continue
        pretty = PRETTY.get(col, col)
        v = f"{thr:.3g}"
        parts.append(f"{pretty} is 0" if (op == "==" and thr == 0)
                     else (f"{pretty} is exactly {v}" if op == "==" else f"{pretty} is at most {v}"))
    return ", AND ".join(parts)


def build_packet(rule, codes, cat_name, dataset):
    """Request body for one (rule, category) pair; asserts that no statistic is present."""
    cols = [re.match(r"^(.*?)(<=|==)", p.strip()).group(1).strip()
            for p in rule.split(" AND ")]
    defs = {}
    for c in cols:
        base = c[:-5] if c.endswith("_code") else c
        if base in PLATFORM_FEATURES:
            defs[base] = "Platform profile field. " + PLATFORM_FEATURES[base]
        elif base in DERIVED_FEATURES:
            defs[base] = ("Derived by us from the earlier observation window, not a "
                          "platform field. " + DERIVED_FEATURES[base])
    packet = {
        "user_state": humanise(rule, codes),
        "target_content_category": {
            "name_as_published_by_the_platform": cat_name,
            "english_translation": GLOSS.get(cat_name, "(no translation available)"),
        },
        "claim_under_test": CLAIM[dataset],
        "what_each_attribute_means": defs,
    }
    blob = json.dumps(packet, ensure_ascii=False).lower()
    for fname in sorted(set(PLATFORM_FEATURES) | set(DERIVED_FEATURES),
                        key=len, reverse=True):
        blob = blob.replace(fname.lower(), "<feature>")
    for b in BANNED:
        assert b not in blob, f"LEAKAGE: banned token {b!r} reached the LLM packet"
    return packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kuairec", choices=sorted(CLAIM))
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()

    # The API key is read from the environment only.
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIPPED: OPENAI_API_KEY is not set"); return 2
    system = SYSTEM_TMPL.replace("%PLATFORM%", PLATFORM).replace("%CRITERIA%", CRITERIA)
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    names_path = os.environ.get("NEGCAND_CATEGORY_NAMES") or f"{DATA}/{a.dataset}/category_names.json"
    names = json.load(open(names_path, encoding="utf-8"))
    codes = json.load(open(f"{ARTIFACTS}/{a.dataset}/feature_codes.json"))
    rules = pd.read_csv(f"{ARTIFACTS}/{a.dataset}/rules.csv")
    q = rules[rules.stage_C == True].drop_duplicates(["rule", "category"])
    print(f"{len(q)} statistically admitted (rule, category) pairs to review with {a.model}")

    def judge_one(r):
        cn = names.get(str(int(r.category)))
        if not cn:
            return {"rule": r.rule, "category": int(r.category), "category_name": None,
                    "category_english": "", "user_state": humanise(r.rule, codes),
                    "verdict": "UNRESOLVED",
                    "reason": "no published category name; meaning not invented"}
        pk = build_packet(r.rule, codes, cn, a.dataset)
        try:
            resp = client.chat.completions.create(
                model=a.model, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(pk, ensure_ascii=False, indent=2)}],
                response_format={"type": "json_object"})
            v = json.loads(resp.choices[0].message.content)
        except Exception as e:
            v = {"verdict": "UNCERTAIN", "reason": f"llm error: {e!r}"}
        vd = v.get("verdict", "UNCERTAIN")
        return {"rule": r.rule, "category": int(r.category), "category_name": cn,
                "category_english": GLOSS.get(cn, ""), "user_state": pk["user_state"],
                "verdict": "UNRESOLVED" if vd == "UNCERTAIN" else vd,
                "reason": v.get("reason", "")}

    with ThreadPoolExecutor(max_workers=12) as ex:
        out = list(ex.map(judge_one, list(q.itertuples())))

    d = pd.DataFrame(out)
    p = f"{ARTIFACTS}/{a.dataset}/domain_review.csv"
    d.to_csv(p, index=False)
    print(d.verdict.value_counts().to_dict()); print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
