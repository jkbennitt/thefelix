# Felix Framework Branching Strategy

## Overview

The Felix Framework uses a Git Flow-based branching strategy optimized for research projects with production deployments. This strategy ensures code quality, facilitates collaboration, and enables automated deployments to Hugging Face Spaces.

## Branch Structure

### Main Branches

#### `main` (Production)
- **Purpose**: Production-ready code deployed to Hugging Face Spaces
- **Protection**: Protected with required status checks
- **Deployment**: Automatic deployment to HF Spaces on push
- **Merge Policy**: Fast-forward only, signed commits required
- **Access**: Maintainers only

#### `develop` (Integration)
- **Purpose**: Integration branch for feature development
- **Protection**: Protected with required PR reviews
- **Testing**: Full test suite must pass
- **Merge Policy**: Squash merging from features
- **Access**: All contributors

### Supporting Branches

#### `feature/*` (Feature Development)
- **Purpose**: Individual feature development
- **Branching From**: `develop`
- **Merging To**: `develop`
- **Naming**: `feature/short-description` or `feature/issue-number`
- **Lifecycle**: Deleted after merge
- **Examples**:
  - `feature/zerogpu-integration`
  - `feature/improved-monitoring`
  - `feature/issue-123-performance-boost`

#### `release/*` (Release Preparation)
- **Purpose**: Release preparation and stabilization
- **Branching From**: `develop`
- **Merging To**: `main` and `develop`
- **Naming**: `release/v1.2.3`
- **Protection**: No direct pushes, PR only
- **Process**:
  1. Branch from develop when feature-complete
  2. Bug fixes only (no new features)
  3. Update version numbers and documentation
  4. Merge to main (creates release)
  5. Merge back to develop

#### `hotfix/*` (Emergency Fixes)
- **Purpose**: Critical production fixes
- **Branching From**: `main`
- **Merging To**: `main` and `develop`
- **Naming**: `hotfix/v1.2.4`
- **Priority**: Immediate review and deployment
- **Process**:
  1. Branch from main
  2. Fix critical issue
  3. Update version number
  4. Merge to main (triggers emergency deployment)
  5. Merge back to develop

#### `hf-spaces` (HF Spaces Specific)
- **Purpose**: Hugging Face Spaces specific configurations
- **Branching From**: `main`
- **Merging To**: HF Spaces repository
- **Special**: Contains HF-specific files (app.py, README.md header)
- **Deployment**: Automatic sync to HF Spaces

## Workflow Process

### Feature Development
```bash
# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/new-feature

# Work on feature
git add .
git commit -m "feat: implement new feature functionality"

# Push and create PR
git push origin feature/new-feature
# Create PR to develop branch
```

### Release Process
```bash
# Start release
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# Prepare release
# Update version numbers, documentation
git commit -m "chore: prepare v1.2.0 release"

# Create PR to main
git push origin release/v1.2.0
# Create PR to main branch
# After merge, tag will be created automatically
```

### Hotfix Process
```bash
# Emergency fix
git checkout main
git pull origin main
git checkout -b hotfix/v1.2.1

# Fix critical issue
git commit -m "fix: resolve critical security vulnerability"

# Create PR to main
git push origin hotfix/v1.2.1
# Create PR to main branch (priority review)
```

## Branch Protection Rules

### Main Branch
- Require pull request reviews (2 reviewers)
- Require status checks to pass:
  - CI/CD pipeline
  - Code quality checks
  - Security scans
  - Performance benchmarks
- Require branches to be up to date
- Require signed commits
- Restrict pushes to repository administrators
- Allow force pushes: No
- Allow deletions: No

### Develop Branch
- Require pull request reviews (1 reviewer)
- Require status checks to pass:
  - Unit tests
  - Integration tests
  - Code quality checks
- Require branches to be up to date
- Allow squash merging only
- Dismiss stale reviews

### Feature Branches
- No protection (lightweight for development)
- Automatically delete after merge
- Require linear history

## Commit Message Convention

We follow the Conventional Commits specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `revert`: Revert previous commit

### Examples
```
feat(agents): add support for dynamic agent spawning
fix(communication): resolve spoke connection timeout issue
docs: update deployment guide for HF Spaces
test(performance): add ZeroGPU benchmark tests
ci: improve deployment pipeline reliability
```

## Versioning Strategy

### Semantic Versioning (SemVer)
- `MAJOR.MINOR.PATCH` (e.g., v1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Pre-release Versions
- `v1.2.3-alpha.1`: Alpha releases
- `v1.2.3-beta.1`: Beta releases
- `v1.2.3-rc.1`: Release candidates

### Version Tagging
- Automatic tagging on main branch merge
- GPG signed tags required
- Triggers deployment to HF Spaces
- Creates GitHub release with changelog

## Integration with HF Spaces

### Automatic Deployment
1. Push to `main` triggers deployment
2. Docker image built and tested
3. Deployment to HF Spaces with ZeroGPU
4. Health checks and validation
5. Rollback on failure

### Manual Deployment
```bash
# Deploy specific version
git checkout main
git tag v1.2.3
git push origin v1.2.3
# Triggers release deployment
```

### Environment-Specific Configurations
- `config/production.json`: Production settings
- `config/staging.json`: Staging environment
- `config/development.json`: Local development

## Code Review Guidelines

### Required Reviews
- **Main branch**: 2 approvals (including 1 maintainer)
- **Develop branch**: 1 approval
- **Feature branches**: Optional but recommended

### Review Checklist
- [ ] Code follows project conventions
- [ ] Tests added/updated for changes
- [ ] Documentation updated if needed
- [ ] Performance impact assessed
- [ ] Security implications reviewed
- [ ] Breaking changes documented

### Automated Checks
- Code formatting (Black, isort)
- Linting (flake8, mypy)
- Security scanning (Bandit, Safety)
- Test coverage (>80% required)
- Performance regression tests

## Troubleshooting

### Common Issues

#### Failed CI/CD Pipeline
```bash
# Check specific job failure
gh run list --limit 5
gh run view [run-id]
```

#### Merge Conflicts
```bash
# Rebase feature branch
git checkout feature/branch-name
git rebase develop
# Resolve conflicts and force push
git push --force-with-lease origin feature/branch-name
```

#### Failed Deployment
```bash
# Check HF Spaces logs
huggingface-cli repo info spaces/username/felix-framework
# Rollback to previous version if needed
git revert HEAD
git push origin main
```

### Emergency Procedures

#### Critical Bug in Production
1. Create hotfix branch from main
2. Fix issue with minimal changes
3. Create emergency PR with priority review
4. Merge and deploy (automatic)
5. Monitor deployment and health checks

#### Failed Deployment Rollback
1. Identify last working commit
2. Create revert commit
3. Fast-track through review process
4. Monitor successful deployment

## Monitoring and Metrics

### Branch Health Metrics
- Merge frequency
- CI/CD success rates
- Review turnaround time
- Deployment success rate
- Hotfix frequency

### Automated Reporting
- Weekly branch status report
- Monthly deployment metrics
- Quarterly security audit
- Performance trend analysis

This branching strategy ensures the Felix Framework maintains high quality while enabling rapid, reliable deployments to Hugging Face Spaces with ZeroGPU integration.