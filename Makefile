# NeMo Open Source Software hub — Fern docs convenience targets.
# CI workflows under .github/workflows/fern-docs-*.yml are the source of truth
# for the published pipeline; these targets mirror the local-developer entry points.
#
# Usage:
#   make docs              # local dev server at http://localhost:3000
#   make docs-check        # validate Fern config + MDX (same checks as CI)
#   make docs-login        # guided dashboard sign-in + fern login
#   make docs-preview      # build a shared preview URL (needs DOCS_FERN_TOKEN)
#   make docs-publish      # trigger the Publish Fern Docs workflow on origin/main
#
# See fern/README.md and fern/agents.md for authoring guidance.

.PHONY: help docs docs-check docs-login docs-preview docs-publish

FERN_DIR := fern
FERN_VERSION := $(shell jq -r .version $(FERN_DIR)/fern.config.json)
FERN ?= npx -y fern-api@$(FERN_VERSION)
PUBLISH_WORKFLOW := Publish Fern Docs

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "NeMo OSS hub — Fern docs Make targets"
	@echo "====================================="
	@echo ""
	@echo "  make docs              Start local Fern dev server (http://localhost:3000)"
	@echo "  make docs-check        Validate Fern config + MDX ('fern check' + 'fern docs md check')"
	@echo "  make docs-login        Guided dashboard sign-in + fern login"
	@echo "  make docs-preview      Build a shared preview URL (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish      Trigger the 'Publish Fern Docs' workflow on origin/main"
	@echo ""
	@echo "First time? Run 'make docs-login' before 'make docs'."
	@echo "Authoring guide: fern/README.md · Agent guide: fern/agents.md"
	@echo ""

docs:
	@echo "Starting Fern dev server (http://localhost:3000)..."
	cd $(FERN_DIR) && $(FERN) docs dev

docs-check:
	@echo "Validating Fern config and MDX..."
	cd $(FERN_DIR) && $(FERN) check
	cd $(FERN_DIR) && $(FERN) docs md check

docs-login:
	@echo ""
	@echo "Fern auth — one-time setup per machine"
	@echo "======================================="
	@echo ""
	@echo "  Step 1: open https://dashboard.buildwithfern.com and sign in with your"
	@echo "          @nvidia.com email (email / magic-link flow, not Google SSO)."
	@echo ""
	@echo "  Step 2: confirm the 'nvidia' organization appears in the sidebar."
	@echo ""
	@echo "  Step 3: complete the CLI login prompt below with the same email."
	@echo ""
	@printf "Have you completed step 1 (dashboard sign-in)? [yes/N]: "
	@read confirm; case "$$confirm" in \
		yes|YES|y|Y) ;; \
		*) echo ""; echo "Bailing. Sign in at the dashboard URL above, then re-run 'make docs-login'."; exit 1 ;; \
	esac
	@echo ""
	$(FERN) login

docs-preview:
	@if [ -z "$$DOCS_FERN_TOKEN" ]; then \
	    echo "DOCS_FERN_TOKEN is not set. Issue a token via 'fern token' on a privileged NVIDIA Fern dashboard account."; \
	    exit 1; \
	fi
	@echo "Building a shared Fern preview URL..."
	cd $(FERN_DIR) && FERN_TOKEN=$$DOCS_FERN_TOKEN $(FERN) generate --docs --preview

docs-publish:
	gh workflow run "$(PUBLISH_WORKFLOW)" --ref main
