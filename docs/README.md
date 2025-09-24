# Felix Framework Documentation Hub

Welcome to the Felix Framework documentation! This comprehensive guide will help you navigate all aspects of the helix-based multi-agent cognitive architecture.

> **Quick Links**: [🚀 Try Live Demo](https://huggingface.co/spaces/jkbennitt/felix-framework) | [📊 Project Summary](./PROJECT_INDEX.md) | [💻 Deploy Your Own](./hf-spaces/guides/deployment-guide.md)

---

## 🚀 Getting Started

### New to Felix?
Start here to understand and use Felix Framework:

- **[Project Summary](./PROJECT_INDEX.md)** - Executive overview, status, and achievements
- **[Live Demo](https://huggingface.co/spaces/jkbennitt/felix-framework)** - Try Felix in your browser (ZeroGPU)
- **[Research Foundation](../RESEARCH_LOG.md)** - Complete research journey and validation

### Quick Start Options
Choose your preferred way to experience Felix:

1. **🌐 Browser Demo** - [HuggingFace Spaces](https://huggingface.co/spaces/jkbennitt/felix-framework) (Zero setup required)
2. **💻 Local Development** - Clone and run locally with Python 3.12+
3. **🚀 Deploy Your Own** - Create your own HF Space with ZeroGPU

---

## 📚 Documentation Sections

### 🌪️ HuggingFace Spaces Deployment
Complete documentation for deploying Felix on HuggingFace Spaces with ZeroGPU:

- **[📖 Deployment Guide](./hf-spaces/guides/deployment-guide.md)** - **START HERE** - Complete step-by-step deployment
- **[⚙️ Configuration Files](./hf-spaces/configuration/)** - Requirements, secrets, metadata
  - [SECRETS_MANAGEMENT.md](./hf-spaces/configuration/SECRETS_MANAGEMENT.md) - Secure token handling
  - [requirements-hf.txt](./hf-spaces/configuration/requirements-hf.txt) - HF-optimized dependencies
  - [space_metadata.yaml](./hf-spaces/configuration/space_metadata.yaml) - YAML frontmatter template
- **[🔧 Troubleshooting](./hf-spaces/troubleshooting/)** - Problem-solving guides
  - [ZEROGPU_HUGGINGFACE_INTEGRATION.md](./hf-spaces/troubleshooting/ZEROGPU_HUGGINGFACE_INTEGRATION.md) - ZeroGPU debugging
- **[📊 Reports & Analysis](./hf-spaces/reports/)** - Performance analysis and optimization
  - [INTEGRATION_TEST_REPORT.md](./hf-spaces/reports/INTEGRATION_TEST_REPORT.md) - ZeroGPU testing results
  - [FRONTEND_OPTIMIZATION_REPORT.md](./hf-spaces/reports/FRONTEND_OPTIMIZATION_REPORT.md) - Frontend performance
  - [deployment_coordination_plan.md](./hf-spaces/reports/deployment_coordination_plan.md) - Comprehensive strategy

### 🏗️ Architecture & Design
Understanding Felix's unique approach to multi-agent coordination:

- **[Core Concepts](./architecture/core/)** - Mathematical foundations
  - [mathematical_model.md](./architecture/core/mathematical_model.md) - Helix geometry and parametric equations
  - [hypothesis_mathematics.md](./architecture/core/hypothesis_mathematics.md) - Statistical validation frameworks
- **[Design Decisions](./architecture/decisions/)** - Architecture Decision Records (ADRs)
  - [ADR-001-technology-stack.md](./architecture/decisions/ADR-001-technology-stack.md) - Technology choices
- **[Geometric Models](./architecture/)** - Visual and mathematical representations
  - [thefelix.md](./architecture/thefelix.md) - Original OpenSCAD prototype
  - [the2ndplan.md](./architecture/the2ndplan.md) - Evolution of the concept

### 📖 User Guides
Learn how to use Felix effectively for different scenarios:

- **[User Guide](./guides/user-guide.md)** - Comprehensive usage documentation
- **[Development Guidelines](../CONTRIBUTING.md)** - Contributing to Felix development
- **LLM Integration** (Legacy - now integrated into HF Spaces deployment)
  - Local LM Studio setup and configuration
  - Multi-model deployment strategies

### 🔬 Research & Validation
Scientific foundation and experimental validation:

- **[Research Log](../RESEARCH_LOG.md)** - Complete research journey with hypothesis testing
- **[Statistical Analysis](../src/comparison/)** - Comparison frameworks and validation tools
- **[Test Suite](../tests/)** - 107+ comprehensive tests validating all components

### 📚 Reference Materials
Technical references and release information:

- **[Release Notes](./reference/RELEASE_NOTES_v0.5.0.md)** - Latest version improvements and features
- **[API Documentation](../src/)** - Complete source code with inline documentation
- **[Performance Benchmarks](../benchmarks/)** - Speed and efficiency comparisons

---

## 🎯 Documentation by Use Case

### For Researchers
- [Project Summary](./PROJECT_INDEX.md) - Research achievements and validation
- [Mathematical Model](./architecture/core/mathematical_model.md) - Formal geometric foundations
- [Statistical Analysis](../src/comparison/) - Hypothesis testing and validation frameworks
- [Research Log](../RESEARCH_LOG.md) - Complete experimental methodology

### For Developers
- [HF Spaces Deployment](./hf-spaces/guides/deployment-guide.md) - Production deployment
- [Development Guidelines](../CONTRIBUTING.md) - Code contribution workflow
- [Architecture Documentation](./architecture/) - Technical implementation details
- [Test Suite](../tests/) - Comprehensive testing examples

### For Users
- [Live Demo](https://huggingface.co/spaces/jkbennitt/felix-framework) - Try Felix now
- [User Guide](./guides/user-guide.md) - How to use Felix effectively
- [Troubleshooting](./hf-spaces/troubleshooting/) - Common issues and solutions

### For DevOps/Deployment
- [Deployment Guide](./hf-spaces/guides/deployment-guide.md) - Complete deployment instructions
- [Configuration Management](./hf-spaces/configuration/) - Secrets and environment setup
- [Performance Optimization](./hf-spaces/reports/) - Optimization strategies and analysis

---

## 🚀 Quick Actions

### Try Felix Right Now
- **[🌐 Live Demo](https://huggingface.co/spaces/jkbennitt/felix-framework)** - Interactive helix-based multi-agent coordination
- **[📱 Mobile Demo](https://huggingface.co/spaces/jkbennitt/felix-framework)** - Responsive design works on all devices

### Deploy Felix
- **[🚀 One-Click Deploy](https://huggingface.co/spaces/jkbennitt/felix-framework?duplicate=true)** - Duplicate to your HF account
- **[📖 Manual Setup](./hf-spaces/guides/deployment-guide.md)** - Step-by-step deployment guide

### Local Development
```bash
# Quick setup
git clone https://github.com/jkbennitt/thefelix.git
cd thefelix
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Validate installation
python tests/validation/validate_felix_framework.py

# Run interactive demo
python examples/blog_writer.py "Your topic here"
```

### Explore the Research
- **[📊 Research Summary](./PROJECT_INDEX.md)** - Key findings and validation results
- **[📈 Performance Analysis](./hf-spaces/reports/)** - Efficiency comparisons and optimizations
- **[🔬 Mathematical Foundation](./architecture/core/mathematical_model.md)** - Geometric model details

---

## 🤝 Contributing

Felix Framework welcomes contributions! Whether you're interested in:

- **🔬 Research Extensions** - New hypotheses and validation studies
- **🏗️ Architecture Improvements** - Performance optimizations and features
- **📚 Documentation** - Tutorials, guides, and examples
- **🐛 Bug Reports** - Issues and improvement suggestions

See our [Contributing Guidelines](../CONTRIBUTING.md) for detailed information.

---

## 📞 Support & Community

- **[📋 Issues](https://github.com/jkbennitt/thefelix/issues)** - Bug reports and feature requests
- **[📖 Documentation](https://github.com/jkbennitt/thefelix/tree/main/docs)** - This comprehensive documentation
- **[🌐 Live Demo](https://huggingface.co/spaces/jkbennitt/felix-framework)** - Try before you deploy

---

**Felix Framework: Where geometry meets artificial intelligence** 🌪️

*Navigate to specific sections using the links above, or start with the [Project Summary](./PROJECT_INDEX.md) for a complete overview of achievements and capabilities.*