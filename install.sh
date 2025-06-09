#!/bin/bash

# TranscriMatic Installation Script
# This script sets up the complete environment for TranscriMatic

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to detect package manager
detect_package_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "apt"
    elif command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v pacman >/dev/null 2>&1; then
        echo "pacman"
    elif command -v brew >/dev/null 2>&1; then
        echo "brew"
    else
        echo "unknown"
    fi
}

# Function to install system packages
install_system_deps() {
    local PKG_MANAGER=$(detect_package_manager)
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Installing system dependencies on Linux..."
        
        # Get Python version for dev package
        PYTHON_MAJOR_MINOR=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        
        case $PKG_MANAGER in
            apt)
                print_status "Using apt package manager"
                # Update package list
                sudo apt-get update
                
                # Install Python dev headers and build tools
                print_status "Installing Python development headers and build tools..."
                sudo apt-get install -y \
                    python${PYTHON_MAJOR_MINOR}-dev \
                    python${PYTHON_MAJOR_MINOR}-venv \
                    build-essential \
                    g++ \
                    gcc \
                    make \
                    libudev-dev \
                    ffmpeg
                ;;
                
            dnf|yum)
                print_status "Using dnf/yum package manager"
                # Install Python dev headers and build tools
                print_status "Installing Python development headers and build tools..."
                sudo $PKG_MANAGER install -y \
                    python${PYTHON_MAJOR_MINOR/./}-devel \
                    gcc \
                    gcc-c++ \
                    make \
                    systemd-devel \
                    ffmpeg
                ;;
                
            pacman)
                print_status "Using pacman package manager"
                # Install Python dev headers and build tools
                print_status "Installing Python development headers and build tools..."
                sudo pacman -S --noconfirm \
                    python \
                    base-devel \
                    systemd-libs \
                    ffmpeg
                ;;
                
            *)
                print_warning "Unknown package manager. Please install manually:"
                echo "  - Python development headers (python-dev)"
                echo "  - Build tools (gcc, g++, make)"
                echo "  - libudev (for USB monitoring)"
                echo "  - ffmpeg (for audio processing)"
                ;;
        esac
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Installing system dependencies on macOS..."
        
        if [[ "$PKG_MANAGER" == "brew" ]]; then
            print_status "Using Homebrew"
            # Install ffmpeg
            if ! command -v ffmpeg >/dev/null 2>&1; then
                print_status "Installing ffmpeg..."
                brew install ffmpeg
            fi
        else
            print_warning "Homebrew not found. Please install ffmpeg manually."
        fi
    fi
}

# Check if running with sudo when needed
check_sudo() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
            print_warning "This script needs sudo privileges to install system packages."
            print_status "Please enter your password when prompted."
        fi
    fi
}

# Main script starts here
print_status "TranscriMatic Installation Script"
echo ""

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python 3 not found! Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    print_error "Python $REQUIRED_VERSION or higher is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
print_status "Python $PYTHON_VERSION found ✓"

# Check sudo access if on Linux
check_sudo

# Install system dependencies
install_system_deps

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists ✓"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and essential tools
print_status "Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install platform-specific dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Installing macOS-specific dependencies..."
    pip install -r requirements-macos.txt
fi

# Download spaCy Spanish language model
print_status "Downloading Spanish language model for spaCy..."
python -m spacy download es_core_news_lg

# Check for Ollama (for local LLM)
if ! command -v ollama >/dev/null 2>&1; then
    print_warning "Ollama not found. For local LLM support, install from: https://ollama.ai"
    echo "  After installing, run: ollama pull llama3:8b-instruct"
else
    print_status "Ollama found ✓"
    # Check if the Spanish model is available
    if ! ollama list | grep -q "llama3:8b-instruct"; then
        print_warning "Recommended model 'llama3:8b-instruct' not found."
        echo "  To download it, run: ollama pull llama3:8b-instruct"
    fi
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p logs
mkdir -p ~/.transcrimatic

# Create config file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    print_status "Creating config.yaml from template..."
    cp config.example.yaml config.yaml
    print_warning "Please edit config.yaml with your settings before running the application"
else
    print_status "config.yaml already exists ✓"
fi

# Set up environment variables file
if [ ! -f ".env" ]; then
    print_status "Creating .env file for API keys..."
    cat > .env << EOL
# TranscriMatic Environment Variables
# Uncomment and set your API keys if using cloud LLM providers

# Google Gemini API Key
# GEMINI_API_KEY=your-gemini-api-key-here

# OpenAI API Key
# OPENAI_API_KEY=your-openai-api-key-here
# OPENAI_ORG=your-org-id-here  # Optional

EOL
    print_warning "If using cloud LLM providers, edit .env file with your API keys"
fi

# Platform-specific instructions
print_status "Checking platform-specific features..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "macOS detected"
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        print_status "Apple Silicon (M3) detected ✓"
        print_status "Will use Metal Performance Shaders for acceleration"
    fi
    print_warning "For USB monitoring, you may need to grant permissions in System Preferences > Security & Privacy"
fi

# Check for acceleration support
print_status "Checking acceleration support..."

# Check for CUDA (NVIDIA GPUs)
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_status "CUDA support detected ✓"
    ACCELERATION="CUDA"
# Check for MPS (Apple Silicon)
elif python -c "import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)" 2>/dev/null | grep -q "True"; then
    print_status "Metal Performance Shaders (MPS) support detected ✓"
    ACCELERATION="MPS"
# Check for ROCm (AMD GPUs) - AI 9 HX370 has integrated graphics
elif [[ -d "/opt/rocm" ]] || command -v rocm-smi >/dev/null 2>&1; then
    print_status "ROCm detected for AMD GPU support"
    print_warning "Note: PyTorch ROCm support may need manual configuration"
    ACCELERATION="ROCm"
else
    print_warning "No GPU acceleration detected. Transcription will use CPU (slower)"
    if lscpu | grep -q "AMD Ryzen AI"; then
        print_status "AMD Ryzen AI processor detected - CPU performance should still be good"
    fi
    ACCELERATION="CPU"
fi

# Make the main script executable
chmod +x main.py

print_status "Installation complete! ✓"
echo ""
echo "System Information:"
echo "  Platform: $OSTYPE"
echo "  Python: $PYTHON_VERSION"
echo "  Acceleration: $ACCELERATION"
echo ""
echo "Next steps:"
echo "1. Edit config.yaml with your settings"
echo "2. If using cloud LLMs, add API keys to .env file"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run the application: python main.py"
echo ""
print_status "All system dependencies have been installed automatically!"