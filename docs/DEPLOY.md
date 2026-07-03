# Docs deploy

For maintainers. The public docs site at **`https://agentix-project.github.io/`**
is built from the [`docs/`](.) directory with
[Mintlify](https://mintlify.com) and self-hosted on GitHub Pages.

The [`docs.yml`](../.github/workflows/docs.yml) workflow runs on every
push that touches `docs/**`:

```text
mint validate (strict)  →  mint export (static Next.js site)  →  push to Agentix-Project/Agentix-Project.github.io
```

The one-time setup below wires up that cross-repo push. Once it's done,
day-to-day work is just [editing pages](#adding-a-page) and pushing.

## One-time setup

1. **Create the sibling repo `Agentix-Project/Agentix-Project.github.io`** (public,
   empty). GitHub Pages serves any repo named `<org>.github.io` at
   `<org>.github.io/` automatically.

2. **Enable Pages on it.** Settings → Pages → *Build and deployment*
   *Source*: **Deploy from a branch** → branch `main`, folder `/ (root)`.

3. **Create a deploy token.** Generate a fine-grained personal access
   token (Settings → Developer settings → Fine-grained tokens) with:
   - *Repository access*: only `Agentix-Project/Agentix-Project.github.io`
   - *Permissions*: **Contents: Read and write**

4. **Add the token as a secret on this repo.** Settings → Secrets and
   variables → Actions → *New repository secret*:
   - Name: `DOCS_DEPLOY_TOKEN`
   - Value: the token from step 3

That's it. The next push that touches `docs/**` (or a manual
*Run workflow*) will build and publish.

## Local development

```bash
cd docs
npm install -g mint    # if you don't have it
mint dev               # http://localhost:3000 — hot reloads on save
mint validate          # strict pre-flight (same as CI)
mint broken-links      # external link check
```

## Adding a page

1. Create `docs/<slug>.mdx` (or `docs/<group>/<slug>.mdx`) with frontmatter:
   ```yaml
   ---
   title: My page
   description: One sentence.
   ---
   ```
2. Add the page's path (without `.mdx`) to the right `navigation.groups`
   entry in [`docs/docs.json`](./docs.json).
3. `mint validate` locally to catch typos. Push.

See [Mintlify component reference](https://mintlify.com/docs/components/cards)
for `<Tip>`, `<Tabs>`, `<CardGroup>`, etc.
