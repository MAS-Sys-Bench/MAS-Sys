TASK_PROMPT_TEMPLATE = """\
You are working on resolving a GitHub issue in a real Python project.

## Issue
Repository: {repo}
Instance ID: {instance_id}

{problem_statement}
{hints_block}
## Repository
The repository has been checked out at the following ABSOLUTE path:
  {repo_dir}

You MUST use this absolute path as prefix for all file operations.
For example, to view the top-level directory, use:
  str_replace_editor(command='view', path='{repo_dir}')
To view a specific file, use:
  str_replace_editor(command='view', path='{repo_dir}/path/to/file.py')
For bash commands, cd into the repo first:
  bash(command='cd {repo_dir} && find . -name "*.py" | head -20')

You are at the base commit where the issue exists.

## Instructions
1. Explore the repository to understand the codebase structure
2. Locate the relevant source files related to the issue
3. Make the necessary code changes to fix the issue
4. Verify your changes are correct
5. When done, return the final result in plain text without any more tool calls

IMPORTANT: Only edit the source files needed to fix the issue. Do NOT modify test files. Always use ABSOLUTE paths starting with {repo_dir} when calling str_replace_editor or bash.
"""
