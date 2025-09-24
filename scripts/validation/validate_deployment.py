#!/usr/bin/env python3
"""
Deployment validation script for Felix Framework.
Validates that all deployment components are properly configured.
"""

import os
import sys
import importlib
import json
from pathlib import Path


def validate_file_structure():
    """Validate that all required deployment files exist."""
    required_files = [
        'Dockerfile',
        'docker-compose.yml',
        'requirements-deployment.txt',
        '.env.example',
        'app_fastapi.py',
        'config/settings.py',
        'deployment/web_service.py',
        'deployment/health_checks.py',
        'deployment/logging_config.py',
        'deployment/security.py',
        '.github/workflows/ci-cd.yml',
        '.github/workflows/security-audit.yml',
        'config/nginx/nginx.conf',
        'config/prometheus.yml',
        'config/redis.conf',
        'DEPLOYMENT_GUIDE.md',
        'DEPLOYMENT_SUMMARY.md'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    return {
        'valid': len(missing_files) == 0,
        'missing_files': missing_files,
        'total_files': len(required_files),
        'found_files': len(required_files) - len(missing_files)
    }


def validate_imports():
    """Validate that deployment modules can be imported."""
    test_imports = [
        'config.settings',
        'deployment.health_checks',
        'deployment.logging_config',
        'deployment.security'
    ]

    import_results = {}

    for module_name in test_imports:
        try:
            importlib.import_module(module_name)
            import_results[module_name] = {'status': 'success', 'error': None}
        except Exception as e:
            import_results[module_name] = {'status': 'failed', 'error': str(e)}

    successful_imports = sum(1 for result in import_results.values() if result['status'] == 'success')

    return {
        'valid': successful_imports == len(test_imports),
        'results': import_results,
        'successful': successful_imports,
        'total': len(test_imports)
    }


def validate_docker_files():
    """Validate Docker configuration files."""
    docker_files = ['Dockerfile', 'docker-compose.yml']
    results = {}

    for file_path in docker_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Basic validation
                if file_path == 'Dockerfile':
                    required_keywords = ['FROM', 'COPY', 'RUN', 'EXPOSE', 'CMD']
                elif file_path == 'docker-compose.yml':
                    required_keywords = ['version:', 'services:', 'felix-app:', 'ports:']

                missing_keywords = [kw for kw in required_keywords if kw not in content]

                results[file_path] = {
                    'exists': True,
                    'valid': len(missing_keywords) == 0,
                    'missing_keywords': missing_keywords,
                    'size_bytes': len(content)
                }
            except Exception as e:
                results[file_path] = {
                    'exists': True,
                    'valid': False,
                    'error': str(e)
                }
        else:
            results[file_path] = {
                'exists': False,
                'valid': False,
                'error': 'File not found'
            }

    return results


def validate_configuration():
    """Validate configuration files."""
    config_files = {
        '.env.example': ['ENVIRONMENT=', 'SECRET_KEY=', 'LLM_ENDPOINT='],
        'config/prometheus.yml': ['global:', 'scrape_configs:', 'felix-framework'],
        'config/nginx/nginx.conf': ['server {', 'proxy_pass', 'felix_app'],
        'config/redis.conf': ['bind', 'port', 'maxmemory']
    }

    results = {}

    for file_path, required_content in config_files.items():
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                missing_content = [item for item in required_content if item not in content]

                results[file_path] = {
                    'exists': True,
                    'valid': len(missing_content) == 0,
                    'missing_content': missing_content,
                    'size_bytes': len(content)
                }
            except Exception as e:
                results[file_path] = {
                    'exists': True,
                    'valid': False,
                    'error': str(e)
                }
        else:
            results[file_path] = {
                'exists': False,
                'valid': False,
                'error': 'File not found'
            }

    return results


def validate_github_actions():
    """Validate GitHub Actions workflows."""
    workflow_files = ['.github/workflows/ci-cd.yml', '.github/workflows/security-audit.yml']
    results = {}

    for file_path in workflow_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # Check for required GitHub Actions keywords
                required_keywords = ['name:', 'on:', 'jobs:', 'runs-on:', 'steps:']
                missing_keywords = [kw for kw in required_keywords if kw not in content]

                results[file_path] = {
                    'exists': True,
                    'valid': len(missing_keywords) == 0,
                    'missing_keywords': missing_keywords,
                    'size_bytes': len(content)
                }
            except Exception as e:
                results[file_path] = {
                    'exists': True,
                    'valid': False,
                    'error': str(e)
                }
        else:
            results[file_path] = {
                'exists': False,
                'valid': False,
                'error': 'File not found'
            }

    return results


def main():
    """Run all validation checks."""
    print("🔍 Validating Felix Framework Deployment Configuration")
    print("=" * 60)

    # Run all validations
    file_structure = validate_file_structure()
    print(f"\n📁 File Structure: {'✅ PASS' if file_structure['valid'] else '❌ FAIL'}")
    print(f"   Found {file_structure['found_files']}/{file_structure['total_files']} required files")

    if file_structure['missing_files']:
        print("   Missing files:")
        for file in file_structure['missing_files']:
            print(f"     - {file}")

    # Skip import validation if running in minimal environment
    try:
        import_validation = validate_imports()
        print(f"\n🐍 Python Imports: {'✅ PASS' if import_validation['valid'] else '❌ FAIL'}")
        print(f"   {import_validation['successful']}/{import_validation['total']} modules imported successfully")

        for module, result in import_validation['results'].items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"     {status} {module}")
            if result['error']:
                print(f"       Error: {result['error']}")
    except Exception as e:
        print(f"\n🐍 Python Imports: ⚠️  SKIPPED (missing dependencies)")

    docker_validation = validate_docker_files()
    print(f"\n🐳 Docker Files:")
    for file, result in docker_validation.items():
        status = "✅ PASS" if result['valid'] else "❌ FAIL"
        print(f"   {status} {file}")
        if not result['valid'] and 'missing_keywords' in result:
            for keyword in result['missing_keywords']:
                print(f"     Missing: {keyword}")

    config_validation = validate_configuration()
    print(f"\n⚙️  Configuration Files:")
    for file, result in config_validation.items():
        status = "✅ PASS" if result['valid'] else "❌ FAIL"
        print(f"   {status} {file}")
        if not result['valid'] and 'missing_content' in result:
            for content in result['missing_content']:
                print(f"     Missing: {content}")

    github_validation = validate_github_actions()
    print(f"\n🚀 GitHub Actions:")
    for file, result in github_validation.items():
        status = "✅ PASS" if result['valid'] else "❌ FAIL"
        print(f"   {status} {file}")

    # Overall status
    all_validations = [
        file_structure['valid'],
        all(result['valid'] for result in docker_validation.values()),
        all(result['valid'] for result in config_validation.values()),
        all(result['valid'] for result in github_validation.values())
    ]

    overall_status = all(all_validations)

    print(f"\n🎯 Overall Status: {'✅ DEPLOYMENT READY' if overall_status else '❌ NEEDS ATTENTION'}")

    if overall_status:
        print("\n🚀 Felix Framework deployment configuration is complete and ready!")
        print("   Next steps:")
        print("   1. Copy .env.example to .env and configure your environment")
        print("   2. Set up GitHub secrets for HF Spaces deployment")
        print("   3. Push to main branch to trigger automated deployment")
        print("   4. Or run locally with: docker-compose up -d")
    else:
        print("\n⚠️  Some validation checks failed. Please review the issues above.")

    return 0 if overall_status else 1


if __name__ == "__main__":
    exit(main())