#!/usr/bin/env python3
"""
HuggingFace Spaces configuration validator for Felix Framework.
Validates configuration files and deployment readiness.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import ast

def validate_app_py() -> Tuple[bool, List[str]]:
    """Validate app.py for HuggingFace Spaces compatibility."""
    issues = []
    app_py_path = Path(__file__).parent.parent / "app.py"

    if not app_py_path.exists():
        return False, ["app.py not found - required for HF Spaces deployment"]

    try:
        with open(app_py_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for required imports
        required_imports = [
            'gradio', 'spaces', 'torch'
        ]

        for import_name in required_imports:
            if import_name not in content:
                issues.append(f"Missing required import: {import_name}")

        # Check for ZeroGPU decorator usage
        if '@spaces.GPU' not in content:
            issues.append("No @spaces.GPU decorators found - ZeroGPU features not utilized")

        # Check for proper main function
        if 'if __name__ == "__main__"' not in content:
            issues.append("No main execution block found")

        # Check for Gradio app launch
        if 'launch(' not in content and 'queue(' not in content:
            issues.append("No Gradio app launch found")

        # Check environment variable handling
        env_vars = ['HF_TOKEN', 'SPACES_ZERO_GPU']
        for var in env_vars:
            if var not in content:
                issues.append(f"Environment variable {var} not handled")

        # Validate Python syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(f"Syntax error in app.py: {e}")

    except Exception as e:
        issues.append(f"Error reading app.py: {e}")

    return len(issues) == 0, issues

def validate_requirements() -> Tuple[bool, List[str]]:
    """Validate requirements.txt for HF Spaces compatibility."""
    issues = []
    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_path.exists():
        return False, ["requirements.txt not found"]

    try:
        with open(requirements_path, 'r') as f:
            requirements = f.read()

        # Required packages for HF Spaces
        required_packages = [
            'gradio', 'spaces', 'torch', 'transformers',
            'numpy', 'plotly'
        ]

        for package in required_packages:
            if package not in requirements.lower():
                issues.append(f"Missing required package: {package}")

        # Check for problematic packages
        problematic = ['tensorflow', 'jax']  # Memory intensive
        for package in problematic:
            if package in requirements.lower():
                issues.append(f"Potentially problematic package for ZeroGPU: {package}")

        # Check version specifications
        lines = [line.strip() for line in requirements.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('#'):
                continue

            if '==' in line:
                package, version = line.split('==', 1)
                # Check for known incompatible versions
                if package.strip() == 'gradio' and version.strip().startswith('3.'):
                    issues.append("Gradio version 3.x may not be compatible with latest Spaces features")

    except Exception as e:
        issues.append(f"Error reading requirements.txt: {e}")

    return len(issues) == 0, issues

def validate_dockerfile() -> Tuple[bool, List[str]]:
    """Validate Dockerfile for HF Spaces if present."""
    issues = []
    dockerfile_path = Path(__file__).parent.parent / "Dockerfile"

    if not dockerfile_path.exists():
        # Dockerfile is optional for HF Spaces
        return True, []

    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Check for Python base image
        if 'FROM python:' not in content and 'FROM pytorch/' not in content:
            issues.append("No Python base image found in Dockerfile")

        # Check for port exposure
        if 'EXPOSE 7860' not in content:
            issues.append("Port 7860 not exposed (required for HF Spaces)")

        # Check for requirements installation
        if 'pip install' not in content and 'requirements.txt' not in content:
            issues.append("No dependency installation found")

        # Check for proper entrypoint
        if 'CMD' not in content and 'ENTRYPOINT' not in content:
            issues.append("No CMD or ENTRYPOINT specified")

    except Exception as e:
        issues.append(f"Error reading Dockerfile: {e}")

    return len(issues) == 0, issues

def validate_readme_header() -> Tuple[bool, List[str]]:
    """Validate README.md has proper HF Spaces metadata."""
    issues = []
    readme_path = Path(__file__).parent.parent / "README.md"

    if not readme_path.exists():
        return False, ["README.md not found"]

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for YAML frontmatter
        if not content.startswith('---'):
            issues.append("README.md missing YAML frontmatter for HF Spaces")
            return False, issues

        # Extract frontmatter
        parts = content.split('---', 2)
        if len(parts) < 3:
            issues.append("Invalid YAML frontmatter structure")
            return False, issues

        frontmatter = parts[1]

        # Required fields for HF Spaces
        required_fields = ['title', 'emoji', 'colorFrom', 'colorTo', 'sdk']
        for field in required_fields:
            if f'{field}:' not in frontmatter:
                issues.append(f"Missing required field in frontmatter: {field}")

        # Check SDK is specified correctly
        if 'sdk:' in frontmatter:
            if 'gradio' not in frontmatter and 'docker' not in frontmatter:
                issues.append("SDK should be 'gradio' or 'docker' for HF Spaces")

        # Check for app_file if using gradio SDK
        if 'sdk: gradio' in frontmatter and 'app_file:' not in frontmatter:
            issues.append("app_file not specified for Gradio SDK")

    except Exception as e:
        issues.append(f"Error reading README.md: {e}")

    return len(issues) == 0, issues

def validate_environment_variables() -> Tuple[bool, List[str]]:
    """Validate environment variable handling."""
    issues = []

    # Check for .env file (should not be committed)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        issues.append(".env file found - remove from repository before deployment")

    # Check .gitignore includes .env
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    if gitignore_path.exists():
        try:
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()

            if '.env' not in gitignore_content:
                issues.append(".env not in .gitignore - add to prevent accidental commits")

        except Exception as e:
            issues.append(f"Error reading .gitignore: {e}")

    return len(issues) == 0, issues

def validate_file_sizes() -> Tuple[bool, List[str]]:
    """Check for files that might be too large for HF Spaces."""
    issues = []
    size_limit_mb = 100  # HF Spaces file size limit

    project_root = Path(__file__).parent.parent

    large_files = []
    for file_path in project_root.rglob('*'):
        if file_path.is_file():
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > size_limit_mb:
                    large_files.append((str(file_path), size_mb))
            except Exception:
                continue

    if large_files:
        for file_path, size in large_files:
            issues.append(f"Large file detected: {file_path} ({size:.1f} MB)")

    return len(issues) == 0, issues

def validate_gpu_memory_usage() -> Tuple[bool, List[str]]:
    """Validate GPU memory usage patterns in code."""
    issues = []
    project_root = Path(__file__).parent.parent

    python_files = list(project_root.glob("**/*.py"))

    memory_patterns = {
        'torch.cuda.empty_cache()': 'Good: GPU memory cleanup',
        'torch.cuda.memory_allocated()': 'Good: Memory monitoring',
        '.cuda()': 'Check: Explicit GPU allocation',
        'device="cuda"': 'Check: GPU device specification',
        'torch.load': 'Check: Model loading (ensure proper device mapping)'
    }

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for pattern, note in memory_patterns.items():
                if pattern in content:
                    if note.startswith('Check:'):
                        # Could be issue, manual review needed
                        issues.append(f"{file_path}: {note}")

        except Exception:
            continue

    return len(issues) == 0, issues

def generate_deployment_checklist() -> Dict[str, Any]:
    """Generate a comprehensive deployment checklist."""
    checklist = {
        'timestamp': str(datetime.now()),
        'checks': {},
        'overall_status': True,
        'critical_issues': [],
        'warnings': [],
        'recommendations': []
    }

    checks = [
        ('app_py', "app.py validation", validate_app_py),
        ('requirements', "requirements.txt validation", validate_requirements),
        ('dockerfile', "Dockerfile validation", validate_dockerfile),
        ('readme', "README.md validation", validate_readme_header),
        ('environment', "Environment variables", validate_environment_variables),
        ('file_sizes', "File size validation", validate_file_sizes),
        ('gpu_memory', "GPU memory usage", validate_gpu_memory_usage),
    ]

    for check_id, check_name, check_func in checks:
        try:
            passed, issues = check_func()
            checklist['checks'][check_id] = {
                'name': check_name,
                'passed': passed,
                'issues': issues
            }

            if not passed:
                checklist['overall_status'] = False
                if check_id in ['app_py', 'requirements', 'readme']:
                    checklist['critical_issues'].extend(issues)
                else:
                    checklist['warnings'].extend(issues)

        except Exception as e:
            checklist['checks'][check_id] = {
                'name': check_name,
                'passed': False,
                'issues': [f"Check failed with error: {e}"]
            }
            checklist['overall_status'] = False
            checklist['critical_issues'].append(f"{check_name}: {e}")

    # Add recommendations
    if checklist['overall_status']:
        checklist['recommendations'] = [
            "✅ Configuration appears ready for HF Spaces deployment",
            "Consider testing locally with gradio first",
            "Monitor GPU memory usage after deployment",
            "Set up monitoring for deployment health"
        ]
    else:
        checklist['recommendations'] = [
            "🔧 Fix critical issues before deployment",
            "Review HF Spaces documentation for requirements",
            "Test configuration changes locally",
            "Consider gradual deployment with feature flags"
        ]

    return checklist

def main():
    """Run HuggingFace Spaces configuration validation."""
    print("🚀 Validating HuggingFace Spaces configuration for Felix Framework...")

    checklist = generate_deployment_checklist()

    # Print results
    print(f"\n📊 Validation Results:")
    print(f"Overall Status: {'✅ READY' if checklist['overall_status'] else '❌ NEEDS FIXES'}")

    print(f"\n🔍 Individual Checks:")
    for check_id, check_info in checklist['checks'].items():
        status = "✅" if check_info['passed'] else "❌"
        print(f"  {status} {check_info['name']}")
        for issue in check_info['issues']:
            print(f"    - {issue}")

    if checklist['critical_issues']:
        print(f"\n🚨 Critical Issues (must fix):")
        for issue in checklist['critical_issues']:
            print(f"  - {issue}")

    if checklist['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in checklist['warnings']:
            print(f"  - {warning}")

    print(f"\n💡 Recommendations:")
    for rec in checklist['recommendations']:
        print(f"  - {rec}")

    # Save detailed report
    report_path = Path(__file__).parent.parent / "hf-spaces-validation-report.json"
    with open(report_path, 'w') as f:
        json.dump(checklist, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if checklist['overall_status'] else 1)

if __name__ == "__main__":
    from datetime import datetime
    main()