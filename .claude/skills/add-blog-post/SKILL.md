---
name: add-blog-post
description: Publish a mini-project markdown file (plus its image) as a Jekyll blog post on this site. Use whenever the user asks to add/publish a post from a source .md (and optional .png), e.g. from ~/Documents/Projects/research/mini-projects.
---

# Add a blog post

Turn a source markdown file (and its image) into a published Jekyll post in this
repo. The user typically points at a file like
`~/Documents/Projects/research/mini-projects/<slug>.md` with a sibling
`<slug>.png`. **Do not read the full source markdown** — a small `head` is
enough to grab the title and find the image reference.

## Conventions (how this site works)

- Posts live in `_posts/` named `YYYY-MM-DD-<slug>.md`.
- Post images live in `assets/posts/<name>.png`.
- Front matter puts the title there (no H1 in the body — it would double up):
  ```
  ---
  layout: post
  title: "<title>"
  date: YYYY-MM-DD
  ---
  ```
- Images are referenced with a `relative_url` filter, not a bare path:
  ```
  ![alt text]({{ "/assets/posts/<name>.png" | relative_url }})
  ```
- Kramdown + MathJax handle `$...$` / `$$...$$` math directly — leave it as-is.

## Steps

1. `head -30` the source `.md` to get the title (usually the leading `# ...`
   line) and confirm the image filename. Check the `.png` exists.
2. Copy the image: `cp <src>.png assets/posts/<name>.png`.
3. Copy the markdown to `_posts/YYYY-MM-DD-<slug>.md`. Use today's date (or a
   date the user specifies). Keep the slug from the source filename unless the
   user wants otherwise.
4. Edit the copied post:
   - Replace the leading `# Title` H1 with the front-matter block above.
   - Fix the image reference from the bare relative path (e.g.
     `](<name>.png)`) to the `relative_url` form.
5. Build to verify: `bundle exec jekyll serve --detach --port 4000`.
   - Sass `slash-div` deprecation warnings from the minima theme are
     pre-existing and expected — ignore them.
   - Confirm the rendered file and image exist and the `<img src>` resolves:
     ```
     ls _site/YYYY/MM/DD/<slug>.html
     grep -o 'src="[^"]*<name>[^"]*"' _site/YYYY/MM/DD/<slug>.html
     ```
6. Give the user the preview link:
   `http://127.0.0.1:4000/YYYY/MM/DD/<slug>.html`
7. When the user says it looks fine, stop the detached server
   (`pkill -f jekyll`) — it runs with auto-regeneration off, so mention that
   edits need a rebuild.

Leave committing to the user unless they ask.
