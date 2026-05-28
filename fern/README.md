# NeMo Framework hub documentation (Fern)

Hub site for the [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) GitHub organization. Routes visitors to each library's documentation using the shared NVIDIA Fern global theme from [fern-components](https://github.com/NVIDIA/fern-components).

## Directory structure

```
fern/
├── fern.config.json
├── docs.yml
├── components/
│   ├── repos.ts          # canonical org repo list
│   └── RepoCatalog.tsx   # searchable catalog UI
└── docs/pages/
    ├── index.mdx
    ├── getting-started.mdx
    ├── repositories.mdx  # full org catalog (primary)
    ├── libraries.mdx     # lifecycle summary
    └── community.mdx
```

When NVIDIA-NeMo adds or archives a repo, update `components/repos.ts` to match [the org repository list](https://github.com/orgs/NVIDIA-NeMo/repositories?type=all).

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
