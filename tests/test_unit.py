"""
Unit tests for learn-claude-code-openai-sdk agents.

These tests do not require API calls and validate structure and local logic.
"""
import importlib.util
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Import Tests
# =============================================================================

def test_imports():
    """Test that all agent modules can be imported."""
    agents = [
        "openai_v0_bash_agent",
        "openai_v0_bash_agent_mini",
        "openai_v1_basic_agent",
        "openai_v2_todo_agent",
        "openai_v3_subagent",
        "openai_v4_skills_agent",
    ]

    for agent in agents:
        spec = importlib.util.find_spec(agent)
        assert spec is not None, f"Failed to find {agent}"
        print(f"  Found: {agent}")

    print("PASS: test_imports")
    return True


# =============================================================================
# TodoManager Tests
# =============================================================================

def test_todo_manager_basic():
    """Test TodoManager basic operations."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()
    result = tm.update(
        [
            {"content": "Task 1", "status": "pending", "activeForm": "Doing task 1"},
            {"content": "Task 2", "status": "in_progress", "activeForm": "Doing task 2"},
        ]
    )

    assert "Task 1" in result
    assert "Task 2" in result
    assert len(tm.items) == 2

    print("PASS: test_todo_manager_basic")
    return True


def test_todo_manager_constraints():
    """Test TodoManager enforces constraints."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()

    try:
        result = tm.update(
            [
                {"content": "Task 1", "status": "in_progress", "activeForm": "Doing 1"},
                {"content": "Task 2", "status": "in_progress", "activeForm": "Doing 2"},
            ]
        )
        assert "Error" in result or "error" in result.lower()
    except ValueError as e:
        assert "in_progress" in str(e).lower()

    tm2 = TodoManager()
    many_items = [
        {"content": f"Task {i}", "status": "pending", "activeForm": f"Doing {i}"}
        for i in range(25)
    ]
    try:
        tm2.update(many_items)
    except ValueError:
        pass
    assert len(tm2.items) <= 20

    print("PASS: test_todo_manager_constraints")
    return True


# =============================================================================
# Reminder Tests
# =============================================================================

def test_reminder_constants():
    """Test reminder constants are defined correctly."""
    from openai_v2_todo_agent import INITIAL_REMINDER, NAG_REMINDER

    assert "<reminder>" in INITIAL_REMINDER
    assert "</reminder>" in INITIAL_REMINDER
    assert "<reminder>" in NAG_REMINDER
    assert "</reminder>" in NAG_REMINDER
    assert "todo" in NAG_REMINDER.lower()

    print("PASS: test_reminder_constants")
    return True


def test_nag_reminder_in_agent_loop():
    """Test NAG_REMINDER injection is inside agent_loop."""
    from openai_v2_todo_agent import agent_loop

    source = inspect.getsource(agent_loop)

    assert "NAG_REMINDER" in source, "NAG_REMINDER should be in agent_loop"
    assert "rounds_without_todo" in source, "rounds_without_todo check should be in agent_loop"
    assert "system_messages.append" in source, "Should inject reminder into system messages"

    print("PASS: test_nag_reminder_in_agent_loop")
    return True


# =============================================================================
# Configuration Tests
# =============================================================================

def test_env_config():
    """Test environment variable configuration."""
    orig_model = os.environ.get("MODEL_ID")

    try:
        os.environ["MODEL_ID"] = "test-model-123"

        import importlib
        import openai_v1_basic_agent

        importlib.reload(openai_v1_basic_agent)
        assert openai_v1_basic_agent.MODEL == "test-model-123"

        print("PASS: test_env_config")
        return True
    finally:
        if orig_model:
            os.environ["MODEL_ID"] = orig_model
        else:
            os.environ.pop("MODEL_ID", None)


def test_default_model():
    """Test default model when env var not set."""
    orig = os.environ.pop("MODEL_ID", None)

    try:
        import importlib
        import openai_v1_basic_agent

        importlib.reload(openai_v1_basic_agent)
        assert "gpt" in openai_v1_basic_agent.MODEL.lower(), openai_v1_basic_agent.MODEL

        print("PASS: test_default_model")
        return True
    finally:
        if orig:
            os.environ["MODEL_ID"] = orig


# =============================================================================
# Tool Schema Tests
# =============================================================================

def test_tool_schemas():
    """Test v1 tool schemas are valid in OpenAI format."""
    from openai_v1_basic_agent import TOOLS

    wrapped = [t["function"] for t in TOOLS]
    required_tools = {"bash", "read_file", "write_file", "edit_file"}
    tool_names = {t["name"] for t in wrapped}

    assert required_tools.issubset(tool_names), f"Missing tools: {required_tools - tool_names}"

    for tool in wrapped:
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["parameters"].get("type") == "object"

    print("PASS: test_tool_schemas")
    return True


# =============================================================================
# TodoManager Edge Case Tests
# =============================================================================

def test_todo_manager_empty_list():
    """Test TodoManager handles empty list."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()
    result = tm.update([])
    assert "No todos" in result or len(tm.items) == 0

    print("PASS: test_todo_manager_empty_list")
    return True


def test_todo_manager_status_transitions():
    """Test TodoManager status transitions."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()
    tm.update([{"content": "Task", "status": "pending", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "pending"

    tm.update([{"content": "Task", "status": "in_progress", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "in_progress"

    tm.update([{"content": "Task", "status": "completed", "activeForm": "Doing task"}])
    assert tm.items[0]["status"] == "completed"

    print("PASS: test_todo_manager_status_transitions")
    return True


def test_todo_manager_missing_fields():
    """Test TodoManager rejects items with missing fields."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()

    try:
        tm.update([{"status": "pending", "activeForm": "Doing"}])
        assert False, "Should reject missing content"
    except ValueError:
        pass

    try:
        tm.update([{"content": "Task", "status": "pending"}])
        assert False, "Should reject missing activeForm"
    except ValueError:
        pass

    print("PASS: test_todo_manager_missing_fields")
    return True


def test_todo_manager_invalid_status():
    """Test TodoManager rejects invalid status values."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()
    try:
        tm.update([{"content": "Task", "status": "invalid", "activeForm": "Doing"}])
        assert False, "Should reject invalid status"
    except ValueError as e:
        assert "status" in str(e).lower()

    print("PASS: test_todo_manager_invalid_status")
    return True


def test_todo_manager_render_format():
    """Test TodoManager render format."""
    from openai_v2_todo_agent import TodoManager

    tm = TodoManager()
    tm.update(
        [
            {"content": "Task A", "status": "completed", "activeForm": "A"},
            {"content": "Task B", "status": "in_progress", "activeForm": "B"},
            {"content": "Task C", "status": "pending", "activeForm": "C"},
        ]
    )

    result = tm.render()
    assert "[x] Task A" in result
    assert "[>] Task B" in result
    assert "[ ] Task C" in result
    assert "1/3" in result

    print("PASS: test_todo_manager_render_format")
    return True


# =============================================================================
# v3 Subagent Tests
# =============================================================================

def test_v3_agent_types_structure():
    """Test v3 AGENT_TYPES structure."""
    from openai_v3_subagent import AGENT_TYPES

    required_types = {"explore", "code", "plan"}
    assert set(AGENT_TYPES.keys()) == required_types

    for name, config in AGENT_TYPES.items():
        assert "description" in config, f"{name} missing description"
        assert "tools" in config, f"{name} missing tools"
        assert "prompt" in config, f"{name} missing prompt"

    print("PASS: test_v3_agent_types_structure")
    return True


def test_v3_get_tools_for_agent():
    """Test v3 get_tools_for_agent filters correctly."""
    from openai_v3_subagent import BASE_TOOLS, get_tools_for_agent

    explore_tools = get_tools_for_agent("explore")
    explore_names = {t["function"]["name"] for t in explore_tools}
    assert "bash" in explore_names
    assert "read_file" in explore_names
    assert "write_file" not in explore_names
    assert "edit_file" not in explore_names

    code_tools = get_tools_for_agent("code")
    assert len(code_tools) == len(BASE_TOOLS)

    plan_tools = get_tools_for_agent("plan")
    plan_names = {t["function"]["name"] for t in plan_tools}
    assert "write_file" not in plan_names

    print("PASS: test_v3_get_tools_for_agent")
    return True


def test_v3_get_agent_descriptions():
    """Test v3 get_agent_descriptions output."""
    from openai_v3_subagent import get_agent_descriptions

    desc = get_agent_descriptions()
    assert "explore" in desc
    assert "code" in desc
    assert "plan" in desc
    assert "read" in desc.lower()

    print("PASS: test_v3_get_agent_descriptions")
    return True


def test_v3_task_tool_schema():
    """Test v3 Task tool schema."""
    from openai_v3_subagent import AGENT_TYPES, TASK_TOOL

    task_fn = TASK_TOOL["function"]
    assert task_fn["name"] == "Task"
    schema = task_fn["parameters"]
    assert "description" in schema["properties"]
    assert "prompt" in schema["properties"]
    assert "agent_type" in schema["properties"]
    assert set(schema["properties"]["agent_type"]["enum"]) == set(AGENT_TYPES.keys())

    print("PASS: test_v3_task_tool_schema")
    return True


# =============================================================================
# v4 SkillLoader Tests
# =============================================================================

def test_v4_skill_loader_init():
    """Test v4 SkillLoader initialization."""
    import tempfile
    from pathlib import Path

    from openai_v4_skills_agent import SkillLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SkillLoader(Path(tmpdir))
        assert len(loader.skills) == 0

    print("PASS: test_v4_skill_loader_init")
    return True


def test_v4_skill_loader_parse_valid():
    """Test v4 SkillLoader parses valid SKILL.md."""
    import tempfile
    from pathlib import Path

    from openai_v4_skills_agent import SkillLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: test
description: A test skill for testing
---

# Test Skill

This is the body content.
"""
        )

        loader = SkillLoader(Path(tmpdir))
        assert "test" in loader.skills
        assert loader.skills["test"]["description"] == "A test skill for testing"
        assert "body content" in loader.skills["test"]["body"]

    print("PASS: test_v4_skill_loader_parse_valid")
    return True


def test_v4_skill_loader_parse_invalid():
    """Test v4 SkillLoader rejects invalid SKILL.md."""
    import tempfile
    from pathlib import Path

    from openai_v4_skills_agent import SkillLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# No frontmatter\n\nJust content.")

        loader = SkillLoader(Path(tmpdir))
        assert "bad-skill" not in loader.skills

    print("PASS: test_v4_skill_loader_parse_invalid")
    return True


def test_v4_skill_loader_get_content():
    """Test v4 SkillLoader get_skill_content."""
    import tempfile
    from pathlib import Path

    from openai_v4_skills_agent import SkillLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "demo"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: demo
description: Demo skill
---

# Demo Instructions

Step 1: Do this
Step 2: Do that
"""
        )

        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.sh").write_text("#!/bin/bash\necho hello")

        loader = SkillLoader(Path(tmpdir))
        content = loader.get_skill_content("demo")
        assert content is not None
        assert "Demo Instructions" in content
        assert "helper.sh" in content
        assert loader.get_skill_content("nonexistent") is None

    print("PASS: test_v4_skill_loader_get_content")
    return True


def test_v4_skill_loader_list_skills():
    """Test v4 SkillLoader list_skills."""
    import tempfile
    from pathlib import Path

    from openai_v4_skills_agent import SkillLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["alpha", "beta"]:
            skill_dir = Path(tmpdir) / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"""---
name: {name}
description: {name} skill
---

Content for {name}
"""
            )

        loader = SkillLoader(Path(tmpdir))
        skills = loader.list_skills()
        assert "alpha" in skills
        assert "beta" in skills
        assert len(skills) == 2

    print("PASS: test_v4_skill_loader_list_skills")
    return True


def test_v4_skill_tool_schema():
    """Test v4 Skill tool schema."""
    from openai_v4_skills_agent import SKILL_TOOL

    skill_fn = SKILL_TOOL["function"]
    assert skill_fn["name"] == "Skill"
    schema = skill_fn["parameters"]
    assert "skill" in schema["properties"]
    assert "skill" in schema["required"]

    print("PASS: test_v4_skill_tool_schema")
    return True


# =============================================================================
# Path Safety Tests
# =============================================================================

def test_v3_safe_path():
    """Test v3 safe_path prevents path traversal."""
    from openai_v3_subagent import WORKDIR, safe_path

    p = safe_path("test.txt")
    assert str(p).startswith(str(WORKDIR))

    try:
        safe_path("../../../etc/passwd")
        assert False, "Should reject path traversal"
    except ValueError as e:
        assert "escape" in str(e).lower()

    print("PASS: test_v3_safe_path")
    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_imports,
        test_todo_manager_basic,
        test_todo_manager_constraints,
        test_reminder_constants,
        test_nag_reminder_in_agent_loop,
        test_env_config,
        test_default_model,
        test_tool_schemas,
        test_todo_manager_empty_list,
        test_todo_manager_status_transitions,
        test_todo_manager_missing_fields,
        test_todo_manager_invalid_status,
        test_todo_manager_render_format,
        test_v3_agent_types_structure,
        test_v3_get_tools_for_agent,
        test_v3_get_agent_descriptions,
        test_v3_task_tool_schema,
        test_v4_skill_loader_init,
        test_v4_skill_loader_parse_valid,
        test_v4_skill_loader_parse_invalid,
        test_v4_skill_loader_get_content,
        test_v4_skill_loader_list_skills,
        test_v4_skill_tool_schema,
        test_v3_safe_path,
    ]

    failed = []
    for test_fn in tests:
        name = test_fn.__name__
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print("=" * 50)
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - len(failed)}/{len(tests)} passed")
    print("=" * 50)

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All unit tests passed!")
        sys.exit(0)
