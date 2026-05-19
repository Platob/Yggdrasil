# Databricks Assistant config for `ygg[data,databricks]`

Instruction files and Skills that teach the Databricks Assistant (Genie Code)
how to write idiomatic code against [Yggdrasil](https://github.com/Platob/Yggdrasil)
(`pip install "ygg[data,databricks]"`).

## What's here

| File | Where it goes in the Assistant settings panel |
| --- | --- |
| [`.assistant_workspace_instructions.md`](.assistant_workspace_instructions.md) | **Workspace instructions** — admin-managed, loaded for every user in the workspace. |
| [`user_instructions.md`](user_instructions.md) | **User instructions** — personal file an individual user can paste into "Add instructions file". |
| [`skills/`](skills/) | **Skills folder** — markdown skills the Assistant can route to per task. Each file is a self-contained skill. |

## Installing into a Databricks workspace

The Assistant configuration UI exposes three slots (see the workspace's
**Assistant settings** sidebar):

1. **Workspace instructions** — copy `.assistant_workspace_instructions.md`
   into the workspace root (a workspace admin can do this from the settings
   panel via "Edit" on the `.assistant_workspace_instructions.md` slot).
2. **User instructions** — each user clicks **Add instructions file** and
   pastes the contents of `user_instructions.md`.
3. **Skills folder** — click **Create skills folder**, then upload the
   files from [`skills/`](skills/) into it. Workspace admins can use the
   **Create workspace skills folder** slot to publish them to every user.

The Assistant picks skills by matching the task against each skill's
"When to use" section, so the filename and that section are what matter
most for routing.

## Keeping these files in sync with the library

Skills reference public surface (`yggdrasil.data.cast.convert`,
`DatabricksClient`, `Volume`, `Table`, `SQLEngine`, …). When that surface
changes in `python/src/yggdrasil/`, update the skill that mentions it. The
canonical style/voice rules live in [`../AGENTS.md`](../AGENTS.md); the
skills here are condensed call-site guidance for end users, not a
replacement for `AGENTS.md`.
