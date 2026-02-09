# Claude Code Hooks Documentation

## Overview

This document explains the hooks configured in `.claude/settings.json` for the Wine Classification ML project. Hooks are automated scripts or prompts that run in response to tool usage events, providing quality gates and automated workflows.

## Hook Configuration Summary

The project uses **3 hooks** across **2 event types**:
- 2 PostToolUse hooks (run after Write/Edit operations)
- 1 PreToolUse hook (run before Bash commands)

---

## 1. PostToolUse Command Hook: Python Quality Checks

### Configuration
```json
{
  "matcher": "Write|Edit",
  "hooks": [
    {
      "type": "command",
      "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/check_python.sh",
      "timeout": 30,
      "statusMessage": "Running ruff and py_compile..."
    }
  ]
}
```

### What It Does

This hook automatically runs **code quality checks** on every Python file that Claude writes or edits. It executes three operations in sequence:

1. **Ruff Check with Auto-Fix** (`ruff check --fix`)
   - Checks for common Python issues (unused imports, undefined names, etc.)
   - Automatically fixes violations when possible
   - Reports any unfixable violations

2. **Ruff Format** (`ruff format`)
   - Formats code according to project style standards
   - Ensures consistent indentation, line length, spacing
   - Similar to Black formatter

3. **Python Compile** (`python -m py_compile`)
   - Verifies syntax correctness
   - Catches syntax errors before runtime
   - Ensures file is valid Python

### When It Fires

**Trigger**: After Claude uses the `Write` or `Edit` tool

**Conditions**:
- File must have `.py` extension
- File must exist on disk
- Only fires for Python files (skips other file types)

**Example Scenarios**:
- ✅ Claude writes `src/01_eda.py` → Hook runs
- ✅ Claude edits `src/02_feature_engineering.py` → Hook runs
- ❌ Claude writes `README.md` → Hook does NOT run (not .py)
- ❌ Claude reads a file → Hook does NOT run (Read tool, not Write/Edit)

### Implementation Details

**Script**: `scripts/check_python.sh`

**Input**: Receives JSON on stdin with `tool_input.file_path`

**Process**:
```bash
1. Extract file path from JSON input
2. Check if file ends with .py
3. Check if file exists
4. Run: uv run ruff check --fix "$FILE_PATH"
5. Run: uv run ruff format "$FILE_PATH"
6. Run: uv run python -m py_compile "$FILE_PATH"
7. Collect errors from all three steps
8. If errors exist, print to stderr and exit code 2
9. If successful, exit code 0
```

**Exit Codes**:
- `0` = Success (all checks passed)
- `2` = Failure (ruff or syntax errors)

### Benefits

- **Automatic code quality**: No need to manually run linters
- **Immediate feedback**: Errors caught right after file creation
- **Consistency**: All files follow same coding standards
- **Time savings**: Eliminates manual formatting step

### Example Output

When hook runs successfully:
```
Running ruff and py_compile...
✓ ruff check passed
✓ ruff format completed
✓ py_compile successful
```

When hook catches errors:
```
Running ruff and py_compile...
Ruff errors:
src/test.py:10:5: F401 'numpy' imported but unused
Syntax error:
  File "src/test.py", line 15
    def bad syntax():
           ^
SyntaxError: invalid syntax
```

---

## 2. PostToolUse Prompt Hook: Test File Creation

### Configuration
```json
{
  "matcher": "Write|Edit",
  "hooks": [
    {
      "type": "prompt",
      "prompt": "If the file you just wrote or edited is a .py file (not a test file), check if a corresponding test file exists in a tests/ directory. If no test file exists, create one with pytest unit tests covering the public functions. Name it test_<filename>.py in the tests/ directory. Skip this for __init__.py files and files that are themselves test files."
    }
  ]
}
```

### What It Does

This hook uses an **LLM-based evaluation** to decide whether a test file should be created for newly written Python files. It:

1. Checks if the file is a `.py` file
2. Determines if it's already a test file
3. Looks for existing test files in `tests/` directory
4. If no test exists, creates a pytest test file with unit tests for public functions
5. Skips `__init__.py` files

### When It Fires

**Trigger**: After Claude uses the `Write` or `Edit` tool

**Conditions**:
- File must be a `.py` file
- File must NOT already be a test file (e.g., not `test_*.py`)
- File must NOT be `__init__.py`
- No corresponding test file exists

**Example Scenarios**:
- ✅ Claude writes `src/eda.py`, no `tests/test_eda.py` exists → Creates test file
- ❌ Claude writes `src/test_utils.py` → Skip (already a test file)
- ❌ Claude writes `src/__init__.py` → Skip (__init__.py)
- ❌ Claude writes `src/model.py`, `tests/test_model.py` exists → Skip (test exists)

### Implementation Details

**Type**: Prompt hook (LLM evaluation, not shell script)

**Process**:
1. Claude evaluates the prompt using an LLM
2. LLM analyzes the file that was just written
3. LLM decides whether to create a test file
4. If yes, LLM generates appropriate pytest unit tests
5. Test file saved to `tests/test_<filename>.py`

**Test File Structure**:
```python
import pytest
from src.module_name import function_name

def test_function_name():
    # Test implementation
    pass
```

### Benefits

- **Test coverage**: Encourages testing for all modules
- **Best practices**: Follows pytest conventions
- **Automated**: No need to manually create test scaffolding
- **Smart**: Skips test files and __init__.py automatically

### Example

**After writing** `src/utils.py`:
```python
def calculate_mean(values: list[float]) -> float:
    return sum(values) / len(values)
```

**Hook creates** `tests/test_utils.py`:
```python
import pytest
from src.utils import calculate_mean

def test_calculate_mean():
    assert calculate_mean([1.0, 2.0, 3.0]) == 2.0

def test_calculate_mean_empty():
    with pytest.raises(ZeroDivisionError):
        calculate_mean([])
```

---

## 3. PreToolUse Command Hook: Block Force Push

### Configuration
```json
{
  "matcher": "Bash",
  "hooks": [
    {
      "type": "command",
      "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/block_force_push.sh"
    }
  ]
}
```

### What It Does

This hook acts as a **safety guard** that prevents destructive git operations. It:

1. Intercepts all Bash commands before execution
2. Checks if command contains `git push --force` or `git push -f`
3. If force push detected, **blocks** the command from executing
4. Returns error message explaining why command was blocked

### When It Fires

**Trigger**: Before Claude uses the `Bash` tool (PreToolUse)

**Conditions**:
- Command must contain both `git` and `push`
- Command must contain `-f` or `--force` flag
- Runs on **every** Bash command (fast check, minimal overhead)

**Example Scenarios**:
- ❌ `git push --force origin main` → **BLOCKED**
- ❌ `git push -f` → **BLOCKED**
- ✅ `git push origin main` → Allowed (no force flag)
- ✅ `ls -la` → Allowed (not a git command)
- ✅ `git pull --force` → Allowed (pull, not push)

### Implementation Details

**Script**: `scripts/block_force_push.sh`

**Input**: Receives JSON on stdin with `tool_input.command`

**Process**:
```bash
1. Read command from JSON input
2. Use grep to check for pattern: 'git\s+push\s+.*(-f|--force)'
3. If pattern matches:
   - Output: {"decision":"block","reason":"Force push is blocked..."}
   - Exit code: 0 (hook executes successfully, command is blocked)
4. If pattern doesn't match:
   - Output: Nothing
   - Exit code: 0 (command allowed to proceed)
```

**Decision Output**:
```json
{
  "decision": "block",
  "reason": "Force push is blocked by project hooks. Use regular git push instead."
}
```

### Why This Is Important

Force pushing can cause data loss:
- Overwrites remote history
- Can delete commits made by others
- Difficult to recover from mistakes
- Particularly dangerous on main/master branches

This hook prevents accidental force pushes while still allowing:
- Regular git push
- Git pull (even with --force)
- Other git operations

### Benefits

- **Prevents data loss**: No accidental history rewrites
- **Team safety**: Protects shared branches
- **Best practices**: Encourages proper git workflows
- **Educational**: Reminds users why force push is dangerous

### Example Output

**Blocked command**:
```
$ git push --force origin main

❌ Force push is blocked by project hooks. Use regular git push instead.
```

**Allowed command**:
```
$ git push origin main

✓ Command executed successfully
```

---

## Hook Execution Flow

### PostToolUse Hooks (Write/Edit)

```
User asks Claude to create a file
    ↓
Claude uses Write tool
    ↓
File is created on disk
    ↓
PostToolUse hooks fire:
    ├─> Command Hook: check_python.sh
    │   ├─> ruff check --fix
    │   ├─> ruff format
    │   └─> py_compile
    │
    └─> Prompt Hook: Test file evaluation
        └─> (May create test file)
    ↓
Results returned to Claude
    ↓
Claude continues or reports errors
```

### PreToolUse Hook (Bash)

```
User asks Claude to run git push --force
    ↓
Claude attempts to use Bash tool
    ↓
PreToolUse hook fires BEFORE execution:
    └─> Command Hook: block_force_push.sh
        └─> Detects force push pattern
        └─> Returns decision="block"
    ↓
Bash tool is BLOCKED (never executes)
    ↓
Claude reports that command was blocked
```

---

## Testing the Hooks

### Test Command Hook

Create a Python file with intentional issues:

```python
# test_hook.py
import numpy  # unused import
def broken syntax():  # syntax error
    pass
```

**Expected**: Hook catches both issues and reports errors

### Test Prompt Hook

Create a Python file without a test:

```python
# src/calculator.py
def add(a: int, b: int) -> int:
    return a + b
```

**Expected**: Hook suggests creating `tests/test_calculator.py`

### Test Force Push Hook

Try to force push:

```bash
git push --force origin main
```

**Expected**: Command is blocked with explanation message

---

## Troubleshooting

### Hook Not Firing

**Problem**: Hooks don't seem to run

**Solutions**:
1. Check `.claude/settings.json` exists at repo root
2. Verify hook scripts are executable: `chmod +x scripts/*.sh`
3. Ensure Claude Code was started from repo root
4. Check logs for hook execution errors

### Hook Failing

**Problem**: Hook runs but reports errors

**Solutions**:
1. Verify `uv` is installed and available
2. Check that required tools are installed: `uv run ruff --version`
3. Ensure `jq` is installed (for JSON parsing in scripts)
4. Review hook script output for specific errors

### Hook Too Slow

**Problem**: Hooks take too long to execute

**Solutions**:
1. Increase timeout in settings.json
2. Check if ruff is running on large files
3. Consider excluding certain files from hook execution

---

## Summary

| Hook | Type | Trigger | Purpose | Script |
|------|------|---------|---------|--------|
| Python Quality Checks | Command | PostToolUse (Write/Edit) | Run ruff and py_compile | check_python.sh |
| Test File Creation | Prompt | PostToolUse (Write/Edit) | Suggest creating pytest tests | (LLM-based) |
| Block Force Push | Command | PreToolUse (Bash) | Prevent destructive git ops | block_force_push.sh |

These hooks ensure:
- ✅ Code quality (formatting, linting, syntax)
- ✅ Test coverage (automated test scaffolding)
- ✅ Safety (prevent dangerous git operations)
- ✅ Consistency (all code follows standards)
- ✅ Automation (no manual quality checks needed)

All hooks are configured specifically for the **Wine Classification ML Project** and enforce the standards defined in `part1_claude_code/CLAUDE.md`.
