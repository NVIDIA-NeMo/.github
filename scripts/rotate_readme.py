"""Refresh the org repository catalog and render the monthly profile README.

Reads rotation/config.json (hand-edited) and rotation/state.json (machine-edited),
fetches public repo metadata, picks this month's featured repos and writes
profile/README.md and profile/repositories.json. Standard library only.
"""

import argparse
import datetime
import html
import json
import os
from pathlib import Path
import re
import sys
import urllib.request


ORG = "NVIDIA-NeMo"
CATEGORIES = {
    "nemo-data": "NeMo Data",
    "nemo-training": "NeMo Training",
    "nemo-evaluation": "NeMo Evaluation",
    "nemo-agents": "NeMo Agents",
    "nemo-safety": "NeMo Safety",
}
EXCLUDE_TOPIC = "nemo-exclude-featured"
SEARCH = "https://github.com/search?q=org%3A{org}+topic%3A{topic}+is%3Apublic+archived%3Afalse&type=repositories"


def fetch_repos(token):
    repos, page = [], 1
    while True:
        url = f"https://api.github.com/orgs/{ORG}/repos?type=public&per_page=100&page={page}"
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "nemo-org-profile-rotation"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            batch = json.load(response)
        repos.extend(batch)
        if len(batch) < 100:
            return repos
        page += 1


def to_entry(repo):
    topics = sorted(repo.get("topics", []))
    return {
        "name": repo["name"],
        "full_name": repo["full_name"],
        "description": repo["description"],
        "repository_url": repo["html_url"],
        "readme_api_url": f'https://api.github.com/repos/{repo["full_name"]}/readme',
        "homepage_url": repo.get("homepage") or None,
        "license": repo.get("license"),
        "default_branch": repo["default_branch"],
        "topics": topics,
        "categories": [label for topic, label in CATEGORIES.items() if topic in topics],
        "research_preview": True if "nemo-research-preview" in topics else None,
        "archived": repo["archived"],
        "disabled": repo["disabled"],
        "fork": repo["fork"],
        "excluded_from_featured": EXCLUDE_TOPIC in topics,
    }


def build_catalog(entries, previous):
    catalog = {
        "schema_version": 1,
        "organization": ORG,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_url": f"https://api.github.com/orgs/{ORG}/repos",
        "scope": "All public repositories, including archived repos and repos excluded from promotion.",
        "field_notes": previous["field_notes"],
        "category_topics": CATEGORIES,
        "repositories": sorted(entries, key=lambda item: item["full_name"].lower()),
    }
    # Keep the old timestamp when nothing else changed, so unchanged runs make no commit.
    if {k: v for k, v in catalog.items() if k != "generated_at"} == {k: v for k, v in previous.items() if k != "generated_at"}:
        catalog["generated_at"] = previous["generated_at"]
    return catalog


def is_eligible(entry, config):
    return not (
        entry["archived"] or entry["disabled"] or entry["fork"]
        or entry["excluded_from_featured"] or entry["name"] in config["excluded"]
    )


def select(period, state, by_name, config):
    """Return this period's featured names, updating state in place.

    A new period takes the eligible repos in the next `cards_per_month` positions of
    the order and always advances the cursor by that many, so later batches never
    shift. Any shortfall, from ineligible repos or a mid-month replacement, is filled
    from the positions after the window without moving the cursor. Such a repo is
    featured again in its own slot. A period already in history is reused.
    """
    count = config["cards_per_month"]
    eligible = {name for name, entry in by_name.items() if is_eligible(entry, config)}
    order = state["order"]
    order.extend(sorted(eligible - set(order), key=str.lower))
    if not order:
        raise ValueError("no eligible repositories to feature")

    if period in state["history"]:
        chosen = [n for n in state["history"][period] if n in eligible]
    else:
        start = state["cursor"] % len(order)
        window = [order[(start + step) % len(order)] for step in range(count)]
        chosen = list(dict.fromkeys(n for n in window if n in eligible))
        state["cursor"] = (start + count) % len(order)

    cursor = state["cursor"] % len(order)
    for step in range(len(order)):
        if len(chosen) >= count:
            break
        name = order[(cursor + step) % len(order)]
        if name in eligible and name not in chosen:
            chosen.append(name)
    state["history"][period] = chosen
    return chosen


def display_name(name, config):
    if name in config["display_names"]:
        return config["display_names"][name]
    words = [w if w != w.lower() else w.capitalize() for w in re.split(r"[-_]", name)]
    if words[0].lower() == "nemo":
        words[0] = "NeMo"
        return " ".join(words)
    return "NeMo " + " ".join(words)


def render_card(entry, config):
    url = html.escape(entry["repository_url"], quote=True)
    lines = [
        '    <td width="50%" valign="top">',
        f'      <h3><a href="{url}">{html.escape(display_name(entry["name"], config))}</a></h3>',
    ]
    if entry["description"]:
        lines.append(f'      <p>{html.escape(entry["description"])}</p>')
    chips = [
        f'<a href="{html.escape(SEARCH.format(org=ORG, topic=topic), quote=True)}">'
        f'<img src="assets/categories/{topic}.svg" alt="{label}" height="22"></a>'
        for topic, label in CATEGORIES.items() if label in entry["categories"]
    ]
    if chips:
        lines.append(f'      <p><br>{" ".join(chips)}</p>')
    lines.append("    </td>")
    return "\n".join(lines)


def render_cards(names, by_name, config):
    cells = [render_card(by_name[n], config) for n in names]
    if len(cells) % 2:
        cells.append('    <td width="50%"></td>')
    rows = [f"  <tr>\n{cells[i]}\n{cells[i + 1]}\n  </tr>" for i in range(0, len(cells), 2)]
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def placeholder(values, key):
    if key not in values:
        raise ValueError(f"unknown template placeholder {{{{{key}}}}}")
    return values[key]


def validate(readme, names, config, root):
    for ref in re.findall(r'src="(assets/[^"]+)"', readme):
        if not (root / "profile" / ref).is_file():
            raise ValueError(f"missing asset {ref}")
    for name in config["excluded"]:
        if f"/{ORG}/{name}\"" in readme:
            raise ValueError(f"excluded repo {name} is featured")
    if readme.count("<td width=\"50%\" valign=\"top\">") != len(names):
        raise ValueError("card count does not match selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=datetime.datetime.now(datetime.timezone.utc).date())
    parser.add_argument("--offline", action="store_true", help="use profile/repositories.json instead of the API")
    parser.add_argument("--dry-run", action="store_true", help="print the selection without writing files")
    args = parser.parse_args()
    root = args.root

    today = datetime.datetime.now(datetime.timezone.utc).date()
    if not args.dry_run and (args.date.year, args.date.month) != (today.year, today.month):
        raise ValueError(f"--date {args.date} is not in the current UTC month; use --dry-run to preview other months")

    config = json.loads((root / "rotation/config.json").read_text())
    state = json.loads((root / "rotation/state.json").read_text())
    previous = json.loads((root / "profile/repositories.json").read_text())

    if args.offline:
        entries = previous["repositories"]
    else:
        entries = [to_entry(r) for r in fetch_repos(os.environ.get("GITHUB_TOKEN"))]
    for entry in entries:
        entry["excluded_from_featured"] = EXCLUDE_TOPIC in entry["topics"] or entry["name"] in config["excluded"]
        override = config.get("license_overrides", {}).get(entry["name"])
        current = entry["license"]
        # Only fill gaps: never replace a license GitHub identified.
        if override and (not current or current["spdx_id"] == "NOASSERTION"):
            entry["license"] = {
                "key": override["key"],
                "name": override["name"],
                "spdx_id": override["spdx_id"],
                "url": f"https://api.github.com/licenses/{override['key']}",
                "source": "manual override",
                "note": override["note"],
            }
    by_name = {e["name"]: e for e in entries}
    catalog = build_catalog(entries, previous)

    period = args.date.strftime("%Y-%m")
    names = select(period, state, by_name, config)
    if not names:
        raise ValueError("selection is empty; refusing to publish an empty README")
    month_year = args.date.strftime("%B %Y")
    cards = render_cards(names, by_name, config)
    template = (root / "profile/README.template.md").read_text()
    values = {"month_year": month_year, "featured_cards_html": cards}
    # One pass, so braces inside repo descriptions are never treated as placeholders.
    readme = re.sub(r"\{\{(\w+)\}\}", lambda m: placeholder(values, m.group(1)), template)
    validate(readme, names, config, root)

    print(f"{month_year}: {', '.join(names)}")
    if args.dry_run:
        return
    outputs = {
        "profile/README.md": readme,
        "profile/repositories.json": json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        "rotation/state.json": json.dumps(state, indent=2) + "\n",
    }
    for relative, text in outputs.items():
        destination = root / relative
        staging = destination.with_suffix(destination.suffix + ".tmp")
        staging.write_text(text)
        staging.replace(destination)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
