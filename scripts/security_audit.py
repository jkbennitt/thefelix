#!/usr/bin/env python3
"""
Security audit script for Felix Framework.
Performs comprehensive security checks before commits.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Configure logging for security audit."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_bandit_scan() -> Tuple[bool, Dict[str, Any]]:
    """Run Bandit security scanner."""
    logger = logging.getLogger(__name__)

    try:
        result = subprocess.run([
            'bandit', '-r', 'src/', 'deployment/',
            '-f', 'json', '-o', 'bandit-report.json'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            logger.info("✅ Bandit scan completed successfully")
            return True, {}
        else:
            with open('bandit-report.json', 'r') as f:
                report = json.load(f)
            return False, report

    except FileNotFoundError:
        logger.error("❌ Bandit not found. Install with: pip install bandit")
        return False, {}
    except Exception as e:
        logger.error(f"❌ Bandit scan failed: {e}")
        return False, {}

def run_safety_check() -> Tuple[bool, Dict[str, Any]]:
    """Run Safety dependency vulnerability check."""
    logger = logging.getLogger(__name__)

    try:
        result = subprocess.run([
            'safety', 'check', '--json', '--output', 'safety-report.json'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            logger.info("✅ Safety check completed successfully")
            return True, {}
        else:
            with open('safety-report.json', 'r') as f:
                report = json.load(f)
            return False, report

    except FileNotFoundError:
        logger.error("❌ Safety not found. Install with: pip install safety")
        return False, {}
    except Exception as e:
        logger.error(f"❌ Safety check failed: {e}")
        return False, {}

def check_secrets_exposure() -> Tuple[bool, List[str]]:
    """Check for potentially exposed secrets."""
    logger = logging.getLogger(__name__)
    issues = []

    # Patterns that might indicate secrets
    secret_patterns = [
        'password', 'token', 'key', 'secret', 'api_key',
        'access_key', 'private_key', 'auth', 'credential'
    ]

    # Files to scan
    scan_files = []
    project_root = Path(__file__).parent.parent

    for pattern in ['src/**/*.py', 'deployment/**/*.py', '*.py', 'config/**/*.json']:
        scan_files.extend(project_root.glob(pattern))

    for file_path in scan_files:
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()

                    for pattern in secret_patterns:
                        if pattern in content:
                            # Check if it's likely a real secret (not just a variable name)
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and ('=' in line or ':' in line):
                                    # Skip if it's clearly a placeholder or environment variable
                                    if any(placeholder in line for placeholder in [
                                        'os.environ', 'env.get', 'getenv',
                                        'placeholder', 'example', 'your_', 'xxx'
                                    ]):
                                        continue

                                    issues.append(f"{file_path}:{i+1} - Potential secret exposure: {line.strip()}")
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")

    if issues:
        logger.error(f"❌ Found {len(issues)} potential secret exposures")
        return False, issues
    else:
        logger.info("✅ No secret exposure detected")
        return True, []

def check_file_permissions() -> Tuple[bool, List[str]]:
    """Check for overly permissive file permissions."""
    logger = logging.getLogger(__name__)
    issues = []

    project_root = Path(__file__).parent.parent

    # Check critical files for proper permissions
    critical_files = [
        '.env', '.env.local', 'config/production.json',
        'scripts/deploy.sh', 'Dockerfile'
    ]

    for file_name in critical_files:
        file_path = project_root / file_name
        if file_path.exists():
            try:
                # On Windows, permission checking is different
                if os.name == 'nt':
                    # Basic check for Windows - just ensure file exists and is readable
                    if not os.access(file_path, os.R_OK):
                        issues.append(f"{file_path} - File not readable")
                else:
                    # Unix-like systems
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]

                    # Check for world-readable sensitive files
                    if file_name.startswith('.env') or 'config' in file_name:
                        if mode[-1] != '0':  # World should not have any permissions
                            issues.append(f"{file_path} - Overly permissive (mode: {mode})")

            except Exception as e:
                logger.warning(f"Could not check permissions for {file_path}: {e}")

    if issues:
        logger.error(f"❌ Found {len(issues)} permission issues")
        return False, issues
    else:
        logger.info("✅ File permissions look good")
        return True, []

def check_dependency_versions() -> Tuple[bool, List[str]]:
    """Check for outdated dependencies with known vulnerabilities."""
    logger = logging.getLogger(__name__)
    issues = []

    try:
        # Check if pip-audit is available
        result = subprocess.run(['pip-audit', '--version'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            # Run pip-audit
            audit_result = subprocess.run([
                'pip-audit', '--format=json', '--output=audit-report.json'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

            if audit_result.returncode != 0:
                try:
                    with open('audit-report.json', 'r') as f:
                        report = json.load(f)

                    for vuln in report.get('vulnerabilities', []):
                        issues.append(f"Vulnerability in {vuln['package']}: {vuln['id']}")

                except Exception:
                    issues.append("pip-audit found vulnerabilities but report parsing failed")
            else:
                logger.info("✅ No dependency vulnerabilities found")

    except FileNotFoundError:
        logger.info("ℹ️  pip-audit not found. Install with: pip install pip-audit")
        return True, []  # Not a failure if tool isn't available
    except Exception as e:
        logger.error(f"❌ Dependency audit failed: {e}")
        return False, [str(e)]

    if issues:
        logger.error(f"❌ Found {len(issues)} dependency issues")
        return False, issues
    else:
        return True, []

def main():
    """Run comprehensive security audit."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🔍 Starting Felix Framework security audit...")

    all_passed = True
    all_issues = []

    # Run all security checks
    checks = [
        ("Bandit Security Scan", run_bandit_scan),
        ("Safety Dependency Check", run_safety_check),
        ("Secrets Exposure Check", check_secrets_exposure),
        ("File Permissions Check", check_file_permissions),
        ("Dependency Vulnerability Check", check_dependency_versions),
    ]

    for check_name, check_func in checks:
        logger.info(f"Running {check_name}...")
        try:
            passed, issues = check_func()
            if not passed:
                all_passed = False
                all_issues.extend(issues if isinstance(issues, list) else [str(issues)])
        except Exception as e:
            logger.error(f"❌ {check_name} failed with exception: {e}")
            all_passed = False
            all_issues.append(f"{check_name}: {str(e)}")

    # Report results
    if all_passed:
        logger.info("🎉 All security checks passed!")
        sys.exit(0)
    else:
        logger.error("❌ Security audit failed with the following issues:")
        for issue in all_issues:
            logger.error(f"  - {issue}")

        # Create summary report
        with open('security-audit-summary.json', 'w') as f:
            json.dump({
                'timestamp': str(datetime.now()),
                'status': 'FAILED',
                'issues': all_issues,
                'total_issues': len(all_issues)
            }, f, indent=2)

        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()