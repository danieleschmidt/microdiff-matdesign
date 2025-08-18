#!/bin/bash

# MicroDiff-MatDesign Development Environment Setup
# This script runs after the dev container is created

echo "🚀 Setting up MicroDiff-MatDesign development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
  build-essential \
  git \
  curl \
  wget \
  unzip \
  htop \
  tree \
  jq \
  graphviz \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1

# Set up Python environment
echo "🐍 Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel
echo "📦 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
echo "⚙️  Installing MicroDiff-MatDesign in development mode..."
pip install -e ".[dev,gpu]"

# Install additional development tools
echo "🔨 Installing additional development tools..."
pip install \
  jupyterlab \
  ipywidgets \
  tensorboard \
  wandb \
  mlflow \
  optuna

# Set up pre-commit hooks
echo "🎯 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {\
  data/{microct,parameters,processed}, \
  models/{pretrained,checkpoints}, \
  logs, \
  experiments, \
  notebooks, \
  profiling, \
  backups \
}

# Set up Git configuration (if not already configured)
if ! git config --get user.name > /dev/null; then
  echo "📝 Setting up Git configuration..."
  git config --global user.name "Development User"
  git config --global user.email "dev@microdiff-matdesign.local"
fi

# Configure Git to trust the workspace directory
git config --global --add safe.directory "$(pwd)"

# Set up environment variables
echo "🌍 Setting up environment variables..."
if [ ! -f .env ]; then
  echo "📋 Creating .env from .env.example..."
  cp .env.example .env
  echo "✏️  Please edit .env to configure your environment"
fi

# Initialize Jupyter Lab configuration
echo "📊 Setting up Jupyter Lab..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Set up shell aliases for common tasks
echo "🔧 Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# MicroDiff-MatDesign Development Aliases
alias md-test='python -m pytest tests/ -v'
alias md-test-cov='python -m pytest tests/ --cov=microdiff_matdesign --cov-report=html'
alias md-lint='ruff check . && black --check . && mypy microdiff_matdesign/'
alias md-format='ruff check --fix . && black . && isort .'
alias md-security='bandit -r microdiff_matdesign/ -f json -o bandit-report.json'
alias md-build='python -m build'
alias md-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete'
alias md-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias md-tensorboard='tensorboard --logdir=logs --host=0.0.0.0 --port=6006'
EOF

# Set up ZSH aliases if ZSH is available
if [ -f ~/.zshrc ]; then
  cat >> ~/.zshrc << 'EOF'

# MicroDiff-MatDesign Development Aliases
alias md-test='python -m pytest tests/ -v'
alias md-test-cov='python -m pytest tests/ --cov=microdiff_matdesign --cov-report=html'
alias md-lint='ruff check . && black --check . && mypy microdiff_matdesign/'
alias md-format='ruff check --fix . && black . && isort .'
alias md-security='bandit -r microdiff_matdesign/ -f json -o bandit-report.json'
alias md-build='python -m build'
alias md-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete'
alias md-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias md-tensorboard='tensorboard --logdir=logs --host=0.0.0.0 --port=6006'
EOF
fi

# Create a development quick-start script
cat > dev-quickstart.sh << 'EOF'
#!/bin/bash
# MicroDiff-MatDesign Development Quick Start
echo "🚀 MicroDiff-MatDesign Development Quick Start"
echo "==============================================="
echo ""
echo "Available commands:"
echo "  md-test         - Run tests"
echo "  md-test-cov     - Run tests with coverage"
echo "  md-lint         - Run linting"
echo "  md-format       - Format code"
echo "  md-security     - Run security scan"
echo "  md-build        - Build package"
echo "  md-clean        - Clean cache files"
echo "  md-jupyter      - Start Jupyter Lab"
echo "  md-tensorboard  - Start TensorBoard"
echo ""
echo "Useful make targets:"
echo "  make test       - Run full test suite"
echo "  make lint       - Run all linters"
echo "  make security   - Run security checks"
echo "  make docs       - Build documentation"
echo ""
echo "🔗 Quick links:"
echo "  - README.md for project overview"
echo "  - docs/ for documentation"
echo "  - tests/ for test examples"
echo "  - .env.example for configuration"
echo ""
EOF
chmod +x dev-quickstart.sh

# Run initial code quality check
echo "🔍 Running initial code quality check..."
ruff check . --output-format=github || echo "⚠️  Some linting issues found - run 'md-format' to fix"

# Display success message
echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Edit .env file to configure your environment"
echo "   2. Run './dev-quickstart.sh' to see available commands"
echo "   3. Run 'md-test' to verify everything works"
echo "   4. Start developing! 🚀"
echo ""
