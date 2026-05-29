# NeMo OSS hub documentation (Fern)

Hub site for open source [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) GitHub repositories. Routes visitors to each library's documentation using the shared NVIDIA Fern global theme from [fern-components](https://github.com/NVIDIA/fern-components). Commercial NeMo products live outside this catalog — refer to [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/).

**Canonical taxonomy:** [TAXONOMY.md](./TAXONOMY.md) — NeMo OSS, Framework, Platform, stages, and repo kinds.

## Information architecture

This hub follows the NVIDIA canonical doc IA from [`tpl-new-site`](https://gitlab-master.nvidia.com/tech-docs/template-library) (`::tpl site`), adapted for an **ecosystem catalog** rather than a single-product manual:

| Canonical section | Hub page | URL |
| --- | --- | --- |
| **About → overview** | Home (section index) | `/` |
| **About → ecosystem** | Ecosystem | `/about/ecosystem` |
| **About → architecture** | Architecture | `/about/architecture` |
| **About → concepts** | Concepts | `/about/concepts` |
| **About → libraries** | Libraries catalog | `/about/libraries` |
| **About → release-notes → index** | Release notes overview | `/about/release-notes` |
| **About → release-notes → containers** | Container releases | `/about/release-notes/containers` |
| **About → release-notes → known-issues** | Known issues | `/about/release-notes/known-issues` |
| **Get Started** | Hub, quickstart, install | `/get-started`, `/get-started/quickstart`, `/get-started/installation` |
| **Get Started → stage** | Data · Pretraining · RL · Inference · E2E | `/get-started/data`, … |
| **Resources → community** | Community | `/resources/community` |

Per-library docs (Curator, AutoModel, Megatron-Bridge, and so on) stay on their own Fern sites. This hub orients readers and links out — it does not duplicate product manuals.

## Content policy

**Keep on the hub** when it helps a reader **choose** at the umbrella level and is likely to stay valid for a long time:

- Lifecycle stages, pipeline shape, Framework vs Platform, and homonyms (refer to [Concepts](/about/concepts) and [TAXONOMY.md](./TAXONOMY.md))
- AutoModel vs Megatron-Bridge and similar **stable forks**
- Catalogs driven from `repos.ts` / `containers.ts` (not hand-maintained repo lists)
- Container release metadata and cross-component known issues for Framework tags

**Push downstream** to a library's own docs when it is:

- Install steps, API usage, tutorials, or model-specific recipes
- Version-pinned commands, example scripts, or benchmark numbers
- Anything that must track frequent product releases

When in doubt: one paragraph plus a link beats copying content that library teams already own.

## Directory structure

```
fern/
├── fern.config.json
├── docs.yml
├── components/
│   ├── repos.ts                  # canonical org repo list
│   ├── RepoCatalog.tsx           # searchable library catalog UI
│   ├── containers.ts             # NGC container list + Framework recent releases
│   ├── ContainerCatalog.tsx      # searchable container catalog UI
│   └── StageGuide.tsx            # per-stage library cards for Get Started pages
└── docs/pages/
    ├── index.mdx                 # About → overview
    ├── about/
    │   ├── ecosystem.mdx
    │   ├── architecture.mdx
    │   ├── concepts.mdx
    │   ├── libraries.mdx         # searchable repo catalog
    │   └── release-notes/
    │       ├── index.mdx         # release-notes overview
    │       ├── containers.mdx    # NGC container announcements
    │       └── known-issues.mdx  # cross-component container issues
    ├── get-started/
    │   ├── index.mdx             # Get Started hub
    │   ├── quickstart.mdx
    │   ├── installation.mdx
    │   ├── data.mdx              # lifecycle stage guides (+ StageGuide)
    │   ├── pretraining.mdx
    │   ├── rl.mdx
    │   ├── inference.mdx
    │   └── e2e.mdx
    └── resources/
        └── community.mdx         # Resources → community
```

When NVIDIA-NeMo adds or archives a repo, update `components/repos.ts`:
- Set **`stage`** to match the org README lifecycle column (Data · Pretraining · RL · Inference · E2E).
- Set **`kind`** to `library`, `integration`, `reference`, or `infrastructure` — see [TAXONOMY.md](./TAXONOMY.md).
- Add **`tags`** for search facets (modality, technique, role). See `GH-TOPICS.MD` for optional GitHub topic alignment.

When a new **NeMo Framework** NGC container ships:

1. Update `latestTag` and `FRAMEWORK_RECENT_RELEASES` in `components/containers.ts` (keep the last three releases).
2. Add any cross-component known issues to `about/release-notes/known-issues.mdx`.

Add new standalone NGC images to `NEMO_CONTAINERS` in `components/containers.ts` when they publish.

Custom React components (e.g. `RepoCatalog`, `ContainerCatalog`) must be **imported** in MDX — bare JSX tags are not auto-registered:

```mdx
import RepoCatalog from "@/components/RepoCatalog";

<RepoCatalog />
```

## Local development

### Prerequisites

- Node.js 22+
- Fern CLI (`npm install -g fern-api`)

### Preview

```bash
cd fern
fern login   # once, for global theme fetch
fern check
fern docs dev
```

Open [http://localhost:3000](http://localhost:3000).

## Publish

Publishing uses the NVIDIA Fern organization token (`DOCS_FERN_TOKEN` org secret).

```bash
git tag docs/v0.1.0 && git push origin docs/v0.1.0
```

Or run the **Publish Fern Docs** workflow from the Actions tab.

Target URLs (configure in `docs.yml`):

- Preview: `nemo-framework.docs.buildwithfern.com/nemo`
- Production: `docs.nvidia.com/nemo`

## Theme

This site uses `global-theme: nvidia`. Theme assets are owned by the fern-components control repo — do not copy logos, CSS, or footer components here. Update branding in fern-components and re-upload the theme.

**GitHub org link:** `navbar-links` in `docs.yml` points to [github.com/NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) (top-right header button). `footer-links.github` duplicates it in the footer. After publish, verify the header link renders — the NVIDIA global theme owns `navbar-links` and may override child values; if missing, add the org URL to the theme or request a site-specific override from the docs platform team.
