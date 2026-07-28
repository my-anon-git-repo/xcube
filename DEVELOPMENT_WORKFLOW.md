# Repo Development & Anonymization Workflow

This file exists so that future-you (or anyone with access to this checkout)
can recall *why* this repo is structured the way it is, and how to safely
develop and publish without leaking identity during double-blind review.

It is intentionally **not** pushed to any GitHub remote — it names real
accounts and would defeat the anonymization it describes if it ended up on
the `anon` or `stub` repos.

## The core idea

> **Repo = project. Branch = paper. Stub branch mirrors (by name) the anon
> branch it redirects to.**

Paper abstracts link directly to a GitHub branch
(`https://github.com/debjyotiSRoy/xcube/tree/<branch>`), so the anonymization
has to happen at the repo/branch level, not just "don't put my name in the
code."

## The three layers

### 1. Real dev repo — here, `~/xcube`

This checkout. Full commit history, real author identity. **Never pushed to
GitHub during a review period.** One branch per paper (e.g. `plant`,
`reasoning`). Develop normally: as many commits as you want, real author
info, no special care needed locally.

### 2. True anonymous mirror — remote `anon`

```
git@github-myanon:my-anon-git-repo/xcube.git
```

One branch per paper. Contains the full code, but only ever receives a
**squashed, re-authored snapshot** — never the real commit history. Author
is rewritten to `Anonymous <anonymous@users.noreply.github.com>`.

### 3. Stub / redirect repo — remote `stub`

```
git@github-researchanon:research-anon-487/xcube.git
```

This is the **renamed original account**
(`debjyotiSRoy` → `research-anon-487`) — kept alive purely so that the
already-public paper link (`github.com/debjyotiSRoy/xcube/tree/<branch>`)
keeps resolving, via GitHub's username-rename redirect. Each branch here is
an **orphan commit containing only a README** that points at the matching
`anon` mirror branch. Issues / PRs / Wiki are disabled on this repo — those
are the #1 silent identity leak.

There's also an `old` remote (`git@github.com:debjyotiSRoy/xcube.git`),
i.e. the same renamed account, kept around for reference.

## Day-to-day: developing

```bash
cd ~/xcube
git checkout <paper-branch>   # e.g. reasoning, plant
# edit, add, commit as normal — real author info, no restrictions
```

## Publishing an update to the anonymous mirror

**Never** `git push anon` or `git push stub` directly from this repo.
Use the script in `scripts/publish_anon.sh` — it *is* the disposable
snapshot procedure below, automated with confirmation prompts at every
destructive step:

```bash
cd ~/xcube
./scripts/publish_anon.sh <paper-branch>   # e.g. reasoning, plant
```

It will, with a `[y/N]` confirmation before each step:

1. Wipe and recreate `~/xcube-publish`.
2. `git archive <paper-branch> | tar -x` into it — content only, no
   history.
3. `git init` a brand-new repo there and commit everything as
   `Anonymous <anonymous@users.noreply.github.com>`.
4. Force-push that single commit to `anon` (`my-anon-git-repo/xcube.git`)
   on a branch of the same name.

This overwrites `my-anon-git-repo/xcube:<paper-branch>` with the current
state of your local branch — regardless of how many local commits you made
to get there. Run it any time you want the anon mirror to reflect your
latest local work.

### Under the hood (what the script does, spelled out)

```bash
cd ~
rm -rf xcube-publish
mkdir xcube-publish && cd xcube-publish

git -C ~/xcube archive <paper-branch> | tar -x

git init
git checkout -b <paper-branch>
git config user.name "Anonymous"
git config user.email "anonymous@users.noreply.github.com"

git add -A
git commit -m "Update anonymous snapshot"

git remote add anon git@github-myanon:my-anon-git-repo/xcube.git
git push -f anon <paper-branch>
```

## Pulling the anon mirror back into a local branch

`scripts/sync_anon_branch.sh <paper-branch>` does the reverse: it hard-resets
your **local** branch to exactly match `anon/<paper-branch>`.

```bash
cd ~/xcube
./scripts/sync_anon_branch.sh reasoning
```

It refuses to run if you have uncommitted changes, and — before resetting —
always creates a safety branch named `backup/<paper-branch>-local-<timestamp>`
pointing at your local branch's current HEAD, so nothing is ever lost. (If
you see branches like `backup/reasoning-local-20260217-170805` lying around,
that's this script's safety net, not manual cleanup debt.) Use this only if
you deliberately want local to fall back to whatever was last published —
it is **not** part of the normal publish flow and will discard local commits
that never made it into the anon snapshot (they still exist on the backup
branch it creates).

## Checking where you are: scripts/repo_status.sh

After a long coding session (or right after SSHing into any machine), run:

```bash
cd ~/xcube
./scripts/repo_status.sh [branch]   # defaults to whatever's checked out
```

It's read-only (it only fetches `anon/<branch>`, never touches the working
tree) and answers exactly one question: **do I need to commit, publish, or
am I already caught up?** It reports:

- Whether the working tree has staged/unstaged changes (with a short
  diffstat) — untracked files are just counted, not listed, since this repo
  accumulates a lot of scratch clutter that's rarely meaningful.
- The local branch's latest commit and `anon/<branch>`'s latest commit.
- Whether `anon/<branch>`'s *content* matches local exactly. This is a tree
  comparison, not a commit-ancestry one — `git status`'s "diverged" language
  is meaningless here, since `anon` branches are squashed snapshots sharing
  no history with local; a content diff is the only comparison that
  actually answers "do I need to publish?"
- Because it fetches `anon` live, it also surfaces if a *different* machine
  already published something newer than what's on this one.

Ends with one bottom-line verdict: commit, publish, or nothing to do.

## Multi-machine development (cloud GPU boxes)

Development also happens across several rented cloud GPU machines, not just
the primary machine. The `anon` remote doubles as the safe cross-machine
relay for this — it's the only remote whose credentials are safe to place
on a disposable, less-trusted rented box.

### Why not just push/pull normally between machines?

The real identity's SSH key (tied to the renamed `research-anon-487` /
`debjyotiSRoy` account) and the `old`/`stub` remotes should never appear on
a rented GPU box — if that box is compromised or shared infrastructure, the
blast radius is real identity. The `anon` remote (`github-myanon` key,
`my-anon-git-repo` account) has no such downside: it only ever holds
anonymized snapshots that were already going to be published anyway.

### `scripts/gitconfig.sh` is the cloud-GPU bootstrap script

This is what `gitconfig.sh` is for: run it once on a fresh rented box to
install git, set the **global** git identity to
`Anonymous <anonymous@users.noreply.github.com>`, and generate a fresh SSH
key for that box. The box then only ever talks to GitHub as `Anonymous`,
through the `anon` remote — never with the real identity.

This also explains an oddity found in the local history: a couple of real
dev commits (e.g. `058e685`, `8342619`) show `Anonymous` as author. Those
weren't publish-snapshot commits — they were normal dev commits made *on a
GPU box*, where `Anonymous` is that box's identity by design, not a mistake.

### Example workflows

1. **Primary machine → GPU box, start of a run.** On the primary machine
   (real identity), commit as usual, then
   `./scripts/publish_anon.sh <branch>`. Then, on the GPU box (bootstrapped
   via `gitconfig.sh`, holding only the `myanon` key):

   - **First time on this box** (a true fresh clone, nothing there yet):
     ```bash
     git clone -o anon git@github-myanon:my-anon-git-repo/xcube.git ~/xcube
     cd ~/xcube && git checkout <branch>
     ```
     Two gotchas a plain `git clone` doesn't handle for you: it names the
     remote `origin` (hence `-o anon` above, to match what
     `sync_anon_branch.sh` expects), and it checks out the anon repo's
     *default* branch — currently `plant` — not necessarily the one you
     want, hence the explicit `checkout`. `sync_anon_branch.sh` is a no-op
     right after this clone, since you already exactly match
     `anon/<branch>` — skip it.

   - **Every later boot from the same image/template** (the repo is
     already there, possibly stale): skip the clone entirely and just run
     ```bash
     cd ~/xcube && ./scripts/sync_anon_branch.sh <branch>
     ```
     This is where the script actually earns its keep — it snaps the
     box's existing checkout forward to the latest `anon/<branch>`,
     auto-backing up whatever local state was there first, before kicking
     off training.

2. **GPU box → GPU box, mid-experiment handoff.** Box A tweaks a script
   mid-run, commits (as `Anonymous`), and runs `publish_anon.sh <branch>`.
   Box B — running a different experiment in parallel, with no direct
   network path to box A — runs `sync_anon_branch.sh <branch>` to pick up
   that change before its next iteration. No SSH tunnel between GPU boxes
   required.

3. **Preempted spot instance recovery.** A spot GPU box gets terminated
   mid-session. As long as `publish_anon.sh` ran before losing it, nothing
   committed is lost — spin up a replacement box and `sync_anon_branch.sh`
   pulls the last published state.

### Tradeoff to remember

`anon/<branch>` is always a single squashed commit. Every sync through it
collapses history — fine for "get the current code onto this box," useless
for preserving fine-grained commit history across machines. Keep the real,
fully-historied development on the primary machine's `~/xcube`.

## Adding a brand-new paper

1. `git checkout -b <new-branch>` locally, develop, commit as usual.
2. `./scripts/publish_anon.sh <new-branch>` to publish the snapshot to `anon`.
3. Create the matching stub branch on `stub` (no script for this yet —
   still manual):

```bash
cd ~
rm -rf xcube-stub-<new-branch>
git clone git@github-researchanon:research-anon-487/xcube.git xcube-stub-<new-branch>
cd xcube-stub-<new-branch>

git checkout --orphan <new-branch>
git rm -rf . 2>/dev/null || true

cat > README.md << 'EOF'
# Anonymous Submission: <Paper Name>

This branch is a stub to preserve anonymity during review.

**Anonymous mirror (code + instructions):**
https://github.com/my-anon-git-repo/xcube/tree/<new-branch>
EOF

git add README.md
git -c user.name="Anonymous" \
    -c user.email="anonymous@users.noreply.github.com" \
    commit -m "Anonymous submission stub (<Paper Name>)"

git push -f origin <new-branch>
```

4. In the paper, link to
   `https://github.com/debjyotiSRoy/xcube/tree/<new-branch>` (redirects →
   `research-anon-487` → stub README → click through to the real anon
   mirror).

## SSH identity setup (two GitHub accounts)

```
Host github-researchanon
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_researchanon

Host github-myanon
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_myanon
```

## Hard rules — don't shoot yourself

- **Never push `~/xcube` directly to any GitHub remote** (`anon`, `stub`,
  or `old`) during a review period — it carries real history/author info.
- **Always publish via `scripts/publish_anon.sh <branch>`** (the disposable
  snapshot repo + force-push, automated).
- **Always link papers to a branch**, never the repo root.
- **Stub + mirror** is the pattern for every new paper — it scales to any
  number of submissions (ICLR, ICML, NeurIPS, ACL, ...).

## Quick verification (incognito, after publishing)

1. `https://github.com/debjyotiSRoy/xcube/tree/<branch>` → redirects → stub
   README only. No code, no issues, no history.
2. `https://github.com/my-anon-git-repo/xcube/tree/<branch>` → full
   anonymous code, anonymous commit author.
