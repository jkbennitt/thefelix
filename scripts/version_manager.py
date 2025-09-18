#!/usr/bin/env python3
"""
Semantic versioning and release management for Felix Framework.
Handles version tagging, changelog generation, and deployment coordination.
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum

class VersionType(Enum):
    """Version increment types."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"

class ReleaseType(Enum):
    """Release channel types."""
    ALPHA = "alpha"
    BETA = "beta"
    RC = "rc"
    STABLE = "stable"

def load_version_file() -> Dict[str, Any]:
    """Load current version from VERSION.json."""
    version_file = Path(__file__).parent.parent / "VERSION.json"

    if not version_file.exists():
        # Create default version file
        default_version = {
            "version": "0.1.0",
            "version_info": {"major": 0, "minor": 1, "patch": 0, "pre_release": None, "build_metadata": None},
            "release_name": "Initial Release",
            "release_date": datetime.now().strftime("%Y-%m-%d"),
            "changelog": "Initial Felix Framework release",
            "compatibility": {"python": ">=3.11"},
            "deployment": {"hf_spaces_ready": False, "docker_ready": False},
            "features": []
        }
        save_version_file(default_version)
        return default_version

    with open(version_file, 'r') as f:
        return json.load(f)

def save_version_file(version_data: Dict[str, Any]) -> None:
    """Save version data to VERSION.json."""
    version_file = Path(__file__).parent.parent / "VERSION.json"
    with open(version_file, 'w') as f:
        json.dump(version_data, f, indent=2)

def parse_semantic_version(version_str: str) -> Dict[str, Any]:
    """Parse semantic version string into components."""
    # Pattern: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$'
    match = re.match(pattern, version_str)

    if not match:
        raise ValueError(f"Invalid semantic version format: {version_str}")

    return {
        "major": int(match.group(1)),
        "minor": int(match.group(2)),
        "patch": int(match.group(3)),
        "pre_release": match.group(4),
        "build_metadata": match.group(5)
    }

def format_semantic_version(version_info: Dict[str, Any]) -> str:
    """Format version components into semantic version string."""
    version = f"{version_info['major']}.{version_info['minor']}.{version_info['patch']}"

    if version_info.get('pre_release'):
        version += f"-{version_info['pre_release']}"

    if version_info.get('build_metadata'):
        version += f"+{version_info['build_metadata']}"

    return version

def increment_version(current_version: Dict[str, Any], version_type: VersionType,
                     prerelease_type: Optional[ReleaseType] = None) -> Dict[str, Any]:
    """Increment version based on type."""
    version_info = current_version['version_info'].copy()

    if version_type == VersionType.MAJOR:
        version_info['major'] += 1
        version_info['minor'] = 0
        version_info['patch'] = 0
        version_info['pre_release'] = None
    elif version_type == VersionType.MINOR:
        version_info['minor'] += 1
        version_info['patch'] = 0
        version_info['pre_release'] = None
    elif version_type == VersionType.PATCH:
        version_info['patch'] += 1
        version_info['pre_release'] = None
    elif version_type == VersionType.PRERELEASE:
        if not prerelease_type:
            prerelease_type = ReleaseType.ALPHA

        current_pre = version_info.get('pre_release', '')
        if current_pre:
            # Increment existing prerelease
            if prerelease_type.value in current_pre:
                parts = current_pre.split('.')
                if len(parts) == 2 and parts[1].isdigit():
                    parts[1] = str(int(parts[1]) + 1)
                    version_info['pre_release'] = '.'.join(parts)
                else:
                    version_info['pre_release'] = f"{prerelease_type.value}.1"
            else:
                version_info['pre_release'] = f"{prerelease_type.value}.1"
        else:
            # First prerelease
            version_info['pre_release'] = f"{prerelease_type.value}.1"

    # Update build metadata with timestamp
    version_info['build_metadata'] = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create new version data
    new_version = current_version.copy()
    new_version['version_info'] = version_info
    new_version['version'] = format_semantic_version(version_info)
    new_version['release_date'] = datetime.now().strftime("%Y-%m-%d")

    return new_version

def get_commit_messages_since_tag(tag: str) -> List[str]:
    """Get commit messages since the specified tag."""
    try:
        result = subprocess.run([
            'git', 'log', f'{tag}..HEAD', '--oneline', '--no-merges'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        else:
            return []
    except Exception:
        return []

def get_latest_git_tag() -> Optional[str]:
    """Get the latest git tag."""
    try:
        result = subprocess.run([
            'git', 'describe', '--tags', '--abbrev=0'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception:
        return None

def categorize_commits(commits: List[str]) -> Dict[str, List[str]]:
    """Categorize commits by type using conventional commit format."""
    categories = {
        'features': [],
        'fixes': [],
        'docs': [],
        'style': [],
        'refactor': [],
        'perf': [],
        'test': [],
        'chore': [],
        'ci': [],
        'breaking': [],
        'other': []
    }

    # Conventional commit patterns
    patterns = {
        'features': r'^feat(\(.+\))?!?:',
        'fixes': r'^fix(\(.+\))?!?:',
        'docs': r'^docs(\(.+\))?!?:',
        'style': r'^style(\(.+\))?!?:',
        'refactor': r'^refactor(\(.+\))?!?:',
        'perf': r'^perf(\(.+\))?!?:',
        'test': r'^test(\(.+\))?!?:',
        'chore': r'^chore(\(.+\))?!?:',
        'ci': r'^ci(\(.+\))?!?:',
        'breaking': r'!:'
    }

    for commit in commits:
        # Remove hash prefix if present
        if ' ' in commit:
            commit_msg = commit.split(' ', 1)[1]
        else:
            commit_msg = commit

        categorized = False

        # Check for breaking changes first
        if re.search(patterns['breaking'], commit_msg):
            categories['breaking'].append(commit_msg)
            categorized = True

        # Check other categories
        for category, pattern in patterns.items():
            if category != 'breaking' and re.search(pattern, commit_msg, re.IGNORECASE):
                categories[category].append(commit_msg)
                categorized = True
                break

        if not categorized:
            categories['other'].append(commit_msg)

    return categories

def generate_changelog(current_version: Dict[str, Any], new_version: Dict[str, Any]) -> str:
    """Generate changelog for the new version."""
    changelog = f"# Changelog for v{new_version['version']}\n\n"
    changelog += f"**Release Date**: {new_version['release_date']}\n"
    changelog += f"**Release Name**: {new_version.get('release_name', 'Unnamed Release')}\n\n"

    # Get commits since last tag
    latest_tag = get_latest_git_tag()
    if latest_tag:
        commits = get_commit_messages_since_tag(latest_tag)
    else:
        # First release
        commits = get_commit_messages_since_tag("HEAD~100")  # Last 100 commits

    if commits:
        categorized = categorize_commits(commits)

        # Breaking changes first
        if categorized['breaking']:
            changelog += "## 🚨 Breaking Changes\n\n"
            for commit in categorized['breaking']:
                changelog += f"- {commit}\n"
            changelog += "\n"

        # Features
        if categorized['features']:
            changelog += "## ✨ New Features\n\n"
            for commit in categorized['features']:
                changelog += f"- {commit}\n"
            changelog += "\n"

        # Bug fixes
        if categorized['fixes']:
            changelog += "## 🐛 Bug Fixes\n\n"
            for commit in categorized['fixes']:
                changelog += f"- {commit}\n"
            changelog += "\n"

        # Performance improvements
        if categorized['perf']:
            changelog += "## ⚡ Performance Improvements\n\n"
            for commit in categorized['perf']:
                changelog += f"- {commit}\n"
            changelog += "\n"

        # Documentation
        if categorized['docs']:
            changelog += "## 📚 Documentation\n\n"
            for commit in categorized['docs']:
                changelog += f"- {commit}\n"
            changelog += "\n"

        # Other changes
        other_categories = ['refactor', 'style', 'test', 'chore', 'ci', 'other']
        other_commits = []
        for category in other_categories:
            other_commits.extend(categorized[category])

        if other_commits:
            changelog += "## 🔧 Other Changes\n\n"
            for commit in other_commits:
                changelog += f"- {commit}\n"
            changelog += "\n"

    # Add deployment information
    if new_version.get('deployment', {}).get('hf_spaces_ready'):
        changelog += "## 🚀 Deployment\n\n"
        changelog += "- ✅ Ready for HuggingFace Spaces deployment\n"
        if new_version['deployment'].get('zerogpu_optimized'):
            changelog += "- ⚡ ZeroGPU acceleration enabled\n"
        if new_version['deployment'].get('docker_ready'):
            changelog += "- 🐳 Docker containerization ready\n"
        changelog += "\n"

    # Add compatibility information
    if new_version.get('compatibility'):
        changelog += "## 📋 Compatibility\n\n"
        for requirement, version in new_version['compatibility'].items():
            changelog += f"- **{requirement}**: {version}\n"
        changelog += "\n"

    return changelog

def create_git_tag(version: str, changelog: str) -> bool:
    """Create and push git tag for the release."""
    try:
        # Create annotated tag
        tag_name = f"v{version}"
        tag_message = f"Felix Framework {version}\n\n{changelog[:500]}..."

        result = subprocess.run([
            'git', 'tag', '-a', tag_name, '-m', tag_message
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode != 0:
            print(f"Failed to create tag: {result.stderr}")
            return False

        # Push tag
        result = subprocess.run([
            'git', 'push', 'origin', tag_name
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        if result.returncode != 0:
            print(f"Failed to push tag: {result.stderr}")
            return False

        return True

    except Exception as e:
        print(f"Error creating git tag: {e}")
        return False

def save_changelog_file(version: str, changelog: str) -> None:
    """Save changelog to file."""
    changelog_dir = Path(__file__).parent.parent / "docs" / "releases"
    changelog_dir.mkdir(parents=True, exist_ok=True)

    changelog_file = changelog_dir / f"v{version}.md"
    with open(changelog_file, 'w') as f:
        f.write(changelog)

    # Update main changelog
    main_changelog = Path(__file__).parent.parent / "CHANGELOG.md"
    if main_changelog.exists():
        with open(main_changelog, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = "# Felix Framework Changelog\n\n"

    with open(main_changelog, 'w') as f:
        f.write("# Felix Framework Changelog\n\n")
        f.write(changelog)
        f.write("\n---\n\n")
        if "# Felix Framework Changelog" in existing_content:
            # Remove the header from existing content
            existing_content = existing_content.split("\n\n", 1)[1] if "\n\n" in existing_content else ""
        f.write(existing_content)

def bump_version(version_type: str, release_type: str = None, release_name: str = None,
                dry_run: bool = False) -> None:
    """Bump version and create release."""
    print(f"🔄 Felix Framework Version Bump ({version_type})")

    # Load current version
    current_version = load_version_file()
    print(f"📋 Current version: {current_version['version']}")

    # Parse version type
    try:
        bump_type = VersionType(version_type)
    except ValueError:
        print(f"❌ Invalid version type: {version_type}")
        print(f"Valid types: {[t.value for t in VersionType]}")
        sys.exit(1)

    # Parse release type for prereleases
    prerelease_type = None
    if bump_type == VersionType.PRERELEASE and release_type:
        try:
            prerelease_type = ReleaseType(release_type)
        except ValueError:
            print(f"❌ Invalid release type: {release_type}")
            print(f"Valid types: {[t.value for t in ReleaseType]}")
            sys.exit(1)

    # Increment version
    new_version = increment_version(current_version, bump_type, prerelease_type)
    new_version['release_name'] = release_name or f"Release {new_version['version']}"

    print(f"📈 New version: {new_version['version']}")
    print(f"🏷️  Release name: {new_version['release_name']}")

    # Generate changelog
    changelog = generate_changelog(current_version, new_version)

    if dry_run:
        print(f"\n📝 Changelog Preview:")
        print(changelog)
        print(f"\n⚠️  Dry run - no changes made")
        return

    # Save new version
    save_version_file(new_version)
    print(f"✅ Updated VERSION.json")

    # Save changelog
    save_changelog_file(new_version['version'], changelog)
    print(f"✅ Updated changelog files")

    # Create git tag
    if create_git_tag(new_version['version'], changelog):
        print(f"✅ Created git tag v{new_version['version']}")
    else:
        print(f"❌ Failed to create git tag")

    print(f"\n🎉 Version bump completed: v{new_version['version']}")
    print(f"📦 Ready for deployment to HuggingFace Spaces")

def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python version_manager.py <major|minor|patch|prerelease> [release_type] [--release-name NAME] [--dry-run]")
        print("\nExamples:")
        print("  python version_manager.py patch")
        print("  python version_manager.py minor --release-name 'ZeroGPU Integration'")
        print("  python version_manager.py prerelease alpha --dry-run")
        sys.exit(1)

    version_type = sys.argv[1]
    release_type = None
    release_name = None
    dry_run = False

    # Parse arguments
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--release-name" and i + 1 < len(args):
            release_name = args[i + 1]
            i += 2
        elif args[i] == "--dry-run":
            dry_run = True
            i += 1
        elif version_type == "prerelease" and release_type is None:
            release_type = args[i]
            i += 1
        else:
            i += 1

    bump_version(version_type, release_type, release_name, dry_run)

if __name__ == "__main__":
    main()