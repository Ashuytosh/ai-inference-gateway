# /daily-push — Auto Push to GitHub

Push all current changes to GitHub automatically. Creates a feature
branch, commits everything, merges to main, pushes, and cleans up.
Zero manual steps.

## Instructions

Run these commands in sequence. Do NOT ask me any questions.
Just execute silently and show the summary at the end.

1. First check if there are any changes to commit:

```bash
git status --porcelain
```

If output is empty, say "Nothing to push — no changes found" and stop.

2. Store the branch name ONCE and reuse it everywhere:

```bash
BRANCH_NAME="update/$(date +%Y-%m-%d-%H%M)"
echo $BRANCH_NAME
```

3. Get current branch info:

```bash
git branch --show-current
```

4. If not on main, stash changes, switch to main, pop stash:

```bash
git stash
git checkout main
git stash pop
```

5. Create a new feature branch:

```bash
git checkout -b $BRANCH_NAME
```

6. Stage ALL changes:

```bash
git add -A
```

7. Look at what files changed and generate a smart commit message:

```bash
git diff --cached --stat
```

Based on the files changed, create a descriptive commit message.
Format: "area1 + area2: brief description of changes"
Examples:
- "models + routes: added chat endpoint and request validation"
- "templates + static: built chat UI with streaming support"
- "services: ollama integration with retry logic"
- "specs + config: updated SPEC.md and environment settings"
- "core: middleware logging and custom exceptions"

8. Commit with the generated message:

```bash
git commit -m "THE_GENERATED_MESSAGE"
```

9. Switch to main:

```bash
git checkout main
```

10. Merge the feature branch into main:

```bash
git merge $BRANCH_NAME --no-edit
```

11. Push main to GitHub:

```bash
git push origin main
```

12. Delete the feature branch (cleanup):

```bash
git branch -d $BRANCH_NAME
```

13. Verify everything is clean:

```bash
git status
git log --oneline -3
```

14. Show this exact summary format: