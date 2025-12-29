"""
Smoke tests for SFT tasks.

These are lightweight tests that verify basic structure without heavy imports.
"""

import ast
from pathlib import Path

import pytest

# Get project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SFT_DIR = PROJECT_ROOT / "sft"


def test_sft_directory_exists():
    """Verify SFT directory structure."""
    assert SFT_DIR.exists()
    assert (SFT_DIR / "propaganda_detection.py").exists()
    assert (SFT_DIR / "claim_verification.py").exists()
    assert (SFT_DIR / "factcheck_generation.py").exists()


def test_models_directory_exists():
    """Verify models directory structure."""
    models_dir = SFT_DIR / "models"
    assert models_dir.exists()
    assert (models_dir / "external_model.py").exists()
    assert (models_dir / "gpt2.py").exists()
    assert (models_dir / "base_gpt.py").exists()


def test_modules_directory_exists():
    """Verify modules directory structure."""
    modules_dir = SFT_DIR / "modules"
    assert modules_dir.exists()
    assert (modules_dir / "attention.py").exists()
    assert (modules_dir / "gpt2_layer.py").exists()


def test_dataset_downloaders_exist():
    """Verify dataset downloader scripts exist."""
    downloaders_dir = SFT_DIR / "dataset_downloaders"
    assert downloaders_dir.exists()
    assert (downloaders_dir / "download_semeval.py").exists()
    assert (downloaders_dir / "download_liar.py").exists()
    assert (downloaders_dir / "download_fever.py").exists()


def parse_python_file(filepath):
    """Parse a Python file and return the AST."""
    with open(filepath) as f:
        return ast.parse(f.read(), filename=str(filepath))


def get_function_names(module_ast):
    """Extract function names from AST."""
    return [node.name for node in ast.walk(module_ast) if isinstance(node, ast.FunctionDef)]


def get_class_names(module_ast):
    """Extract class names from AST."""
    return [node.name for node in ast.walk(module_ast) if isinstance(node, ast.ClassDef)]


def test_propaganda_detection_structure():
    """Verify propaganda_detection.py has expected structure."""
    filepath = SFT_DIR / "propaganda_detection.py"
    module_ast = parse_python_file(filepath)

    functions = get_function_names(module_ast)
    assert "main" in functions


def test_claim_verification_structure():
    """Verify claim_verification.py has expected structure."""
    filepath = SFT_DIR / "claim_verification.py"
    module_ast = parse_python_file(filepath)

    functions = get_function_names(module_ast)
    assert "main" in functions


def test_factcheck_generation_structure():
    """Verify factcheck_generation.py has expected structure."""
    filepath = SFT_DIR / "factcheck_generation.py"
    module_ast = parse_python_file(filepath)

    functions = get_function_names(module_ast)
    assert "main" in functions


def test_external_model_has_required_classes():
    """Verify external_model.py defines required classes."""
    filepath = SFT_DIR / "models" / "external_model.py"
    module_ast = parse_python_file(filepath)

    classes = get_class_names(module_ast)
    assert "FineTuningConfig" in classes
    assert "BaseModelType" in classes

    functions = get_function_names(module_ast)
    assert "get_fine_tuned_external_gpt2" in functions


def test_datasets_internal_has_loaders():
    """Verify datasets_internal.py has data loading functions."""
    filepath = SFT_DIR / "datasets_internal.py"
    module_ast = parse_python_file(filepath)

    functions = get_function_names(module_ast)
    assert "load_propaganda_data" in functions
    assert "load_claim_data" in functions


def test_lora_config_has_shared_configs():
    """Verify lora_config.py defines shared LoRA configurations."""
    filepath = SFT_DIR / "lora_config.py"
    module_ast = parse_python_file(filepath)

    functions = get_function_names(module_ast)
    assert "get_classification_lora_config" in functions
    assert "get_generation_lora_config" in functions


def test_no_syntax_errors_in_tasks():
    """Verify all task scripts have valid Python syntax."""
    task_files = [
        "propaganda_detection.py",
        "claim_verification.py",
        "factcheck_generation.py",
    ]

    for filename in task_files:
        filepath = SFT_DIR / filename
        try:
            parse_python_file(filepath)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filename}: {e}")


def test_no_syntax_errors_in_models():
    """Verify model files have valid Python syntax."""
    model_files = [
        "external_model.py",
        "gpt2.py",
        "base_gpt.py",
    ]

    for filename in model_files:
        filepath = SFT_DIR / "models" / filename
        try:
            parse_python_file(filepath)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filename}: {e}")


def test_invoke_tasks_file_exists():
    """Verify invoke tasks file exists and has task definitions."""
    tasks_file = PROJECT_ROOT / "tasks.py"
    assert tasks_file.exists()

    module_ast = parse_python_file(tasks_file)
    functions = get_function_names(module_ast)

    # Check for SFT task definitions
    assert "propaganda" in functions
    assert "claim_verification" in functions
    assert "factcheck_gen" in functions


def test_readme_exists_and_mentions_tasks():
    """Verify README documents the tasks."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists()

    content = readme.read_text()
    assert "Propaganda" in content or "propaganda" in content
    assert "LIAR" in content or "liar" in content
    assert "FEVER" in content or "fever" in content


def test_quickstart_exists():
    """Verify QUICKSTART guide exists."""
    quickstart = PROJECT_ROOT / "QUICKSTART.md"
    assert quickstart.exists()


@pytest.mark.parametrize(
    "script_name,expected_args",
    [
        ("propaganda_detection.py", ["--model_name", "--ft_config", "--use_gpu"]),
        ("claim_verification.py", ["--dataset", "--model_name", "--ft_config"]),
        ("factcheck_generation.py", ["--dataset", "--model_name", "--use_gpu"]),
    ],
)
def test_task_scripts_have_expected_arguments(script_name, expected_args):
    """Verify task scripts define expected command-line arguments."""
    filepath = SFT_DIR / script_name
    content = filepath.read_text()

    for arg in expected_args:
        assert arg in content, f"{script_name} missing argument: {arg}"


def test_all_tasks_support_model_name_argument():
    """Verify all tasks support --model_name for multi-model support."""
    task_files = [
        "propaganda_detection.py",
        "claim_verification.py",
        "factcheck_generation.py",
    ]

    for filename in task_files:
        filepath = SFT_DIR / filename
        content = filepath.read_text()
        assert "--model_name" in content, f"{filename} missing --model_name support"
        assert "Qwen" in content or "qwen" in content, f"{filename} should default to Qwen"
