#!/usr/bin/env python3
"""
Documentation synchronization script for Felix Framework.
Updates documentation when code changes are detected.
"""

import os
import sys
import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def extract_docstrings(file_path: Path) -> Dict[str, str]:
    """Extract docstrings from Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        docstrings = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if (ast.get_docstring(node) and
                    hasattr(node, 'lineno')):
                    key = f"{file_path.name}:{node.name}"
                    docstrings[key] = ast.get_docstring(node)

        return docstrings

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}

def extract_api_endpoints(file_path: Path) -> List[Dict[str, str]]:
    """Extract API endpoints from Python files."""
    endpoints = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for FastAPI route decorators
        route_patterns = [
            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
        ]

        for pattern in route_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append({
                    'method': method,
                    'path': path,
                    'file': str(file_path),
                })

    except Exception as e:
        print(f"Error extracting endpoints from {file_path}: {e}")

    return endpoints

def update_api_documentation():
    """Update API documentation based on current endpoints."""
    project_root = Path(__file__).parent.parent
    api_files = []

    # Find API-related files
    for pattern in ['deployment/**/*.py', 'src/**/*api*.py', 'src/**/*router*.py']:
        api_files.extend(project_root.glob(pattern))

    all_endpoints = []
    for file_path in api_files:
        if file_path.is_file():
            endpoints = extract_api_endpoints(file_path)
            all_endpoints.extend(endpoints)

    # Generate API documentation
    api_doc_path = project_root / "docs" / "api" / "endpoints.md"
    api_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(api_doc_path, 'w') as f:
        f.write("# Felix Framework API Endpoints\n\n")
        f.write("This document is automatically generated from code analysis.\n\n")

        if all_endpoints:
            f.write("## Available Endpoints\n\n")
            for endpoint in sorted(all_endpoints, key=lambda x: (x['path'], x['method'])):
                f.write(f"### {endpoint['method']} {endpoint['path']}\n\n")
                f.write(f"- **File**: `{endpoint['file']}`\n")
                f.write(f"- **Method**: {endpoint['method']}\n")
                f.write(f"- **Path**: {endpoint['path']}\n\n")
        else:
            f.write("No API endpoints found.\n\n")

    print(f"Updated API documentation: {api_doc_path}")

def update_class_documentation():
    """Update class and function documentation."""
    project_root = Path(__file__).parent.parent
    src_files = list(project_root.glob("src/**/*.py"))

    all_docstrings = {}
    for file_path in src_files:
        if file_path.is_file():
            docstrings = extract_docstrings(file_path)
            all_docstrings.update(docstrings)

    # Generate class documentation
    class_doc_path = project_root / "docs" / "api" / "classes.md"
    class_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(class_doc_path, 'w') as f:
        f.write("# Felix Framework Classes and Functions\n\n")
        f.write("This document is automatically generated from code docstrings.\n\n")

        if all_docstrings:
            for key, docstring in sorted(all_docstrings.items()):
                f.write(f"## {key}\n\n")
                f.write(f"```\n{docstring}\n```\n\n")
        else:
            f.write("No documented classes or functions found.\n\n")

    print(f"Updated class documentation: {class_doc_path}")

def update_configuration_docs():
    """Update configuration documentation based on actual config files."""
    project_root = Path(__file__).parent.parent
    config_files = list(project_root.glob("config/**/*.json"))

    config_doc_path = project_root / "docs" / "configuration" / "reference.md"
    config_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_doc_path, 'w') as f:
        f.write("# Configuration Reference\n\n")
        f.write("This document describes all configuration options for Felix Framework.\n\n")

        for config_file in sorted(config_files):
            try:
                with open(config_file, 'r') as cf:
                    config_data = json.load(cf)

                f.write(f"## {config_file.name}\n\n")
                f.write(f"**Location**: `{config_file.relative_to(project_root)}`\n\n")
                f.write("```json\n")
                f.write(json.dumps(config_data, indent=2))
                f.write("\n```\n\n")

            except Exception as e:
                f.write(f"Error reading {config_file}: {e}\n\n")

    print(f"Updated configuration documentation: {config_doc_path}")

def update_deployment_docs():
    """Update deployment documentation based on current files."""
    project_root = Path(__file__).parent.parent

    # Check for deployment files
    deployment_files = {
        'Dockerfile': 'Docker containerization',
        'docker-compose.yml': 'Docker Compose configuration',
        'app.py': 'HuggingFace Spaces entry point',
        'requirements.txt': 'Python dependencies',
        '.github/workflows/ci-cd.yml': 'CI/CD pipeline'
    }

    deploy_doc_path = project_root / "docs" / "deployment" / "current_setup.md"
    deploy_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(deploy_doc_path, 'w') as f:
        f.write("# Current Deployment Setup\n\n")
        f.write("This document describes the current deployment configuration.\n\n")

        f.write("## Deployment Files\n\n")
        for filename, description in deployment_files.items():
            file_path = project_root / filename
            if file_path.exists():
                f.write(f"- ✅ **{filename}**: {description}\n")
            else:
                f.write(f"- ❌ **{filename}**: {description} (missing)\n")

        f.write("\n## HuggingFace Spaces Configuration\n\n")
        app_py_path = project_root / "app.py"
        if app_py_path.exists():
            f.write("HuggingFace Spaces entry point configured.\n\n")

            # Extract environment variables from app.py
            try:
                with open(app_py_path, 'r') as app_file:
                    content = app_file.read()
                    env_vars = re.findall(r'os\.environ\.get\(["\']([^"\']+)["\']', content)
                    env_vars.extend(re.findall(r'os\.getenv\(["\']([^"\']+)["\']', content))

                if env_vars:
                    f.write("### Environment Variables\n\n")
                    for var in sorted(set(env_vars)):
                        f.write(f"- `{var}`\n")
                    f.write("\n")

            except Exception as e:
                f.write(f"Error analyzing app.py: {e}\n\n")

        else:
            f.write("No HuggingFace Spaces configuration found.\n\n")

    print(f"Updated deployment documentation: {deploy_doc_path}")

def generate_changelog_entry():
    """Generate changelog entry for recent commits."""
    project_root = Path(__file__).parent.parent

    try:
        # Get recent commits
        result = subprocess.run([
            'git', 'log', '--oneline', '-10'
        ], capture_output=True, text=True, cwd=project_root)

        if result.returncode == 0:
            commits = result.stdout.strip().split('\n')

            changelog_path = project_root / "docs" / "RECENT_CHANGES.md"
            with open(changelog_path, 'w') as f:
                f.write("# Recent Changes\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Latest Commits\n\n")

                for commit in commits:
                    if commit.strip():
                        f.write(f"- {commit}\n")

            print(f"Updated changelog: {changelog_path}")

    except Exception as e:
        print(f"Error generating changelog: {e}")

def main():
    """Update all documentation."""
    print("🔄 Updating Felix Framework documentation...")

    try:
        update_api_documentation()
        update_class_documentation()
        update_configuration_docs()
        update_deployment_docs()
        generate_changelog_entry()

        print("✅ Documentation update completed successfully")

    except Exception as e:
        print(f"❌ Documentation update failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()