import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3] / 'meta_stackelberg'
LEGACY_ROOTS = {'fl_sandbox', 'meta_sg', 'src'}


def _legacy_imports() -> list[str]:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split('.', 1)[0] in LEGACY_ROOTS:
                    violations.append(f'{path.relative_to(PACKAGE_ROOT)}:{node.lineno}:{name}')
    return violations


def test_canonical_package_exists() -> None:
    assert PACKAGE_ROOT.is_dir()


def test_canonical_package_does_not_import_legacy_packages() -> None:
    assert _legacy_imports() == []


def test_security_layer_does_not_import_oracle_evaluation() -> None:
    violations = []
    for path in sorted((PACKAGE_ROOT / 'security').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name == 'meta_stackelberg.evaluation' or name.startswith(
                    'meta_stackelberg.evaluation.'
                ):
                    violations.append(
                        f'{path.relative_to(PACKAGE_ROOT)}:{node.lineno}:{name}'
                    )
    assert violations == []
