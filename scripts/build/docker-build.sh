#!/bin/bash

# MicroDiff-MatDesign Docker Build Script
# Optimized multi-stage builds with caching and security scanning

set -euo pipefail

# Configuration
IMAGE_NAME="microdiff-matdesign"
REGISTRY="ghcr.io/danieleschmidt"
BUILD_CACHE_REGISTRY="${REGISTRY}/cache"
CUDA_VERSION="11.8"
PYTHON_VERSION="3.11"
BUILD_PLATFORM="linux/amd64,linux/arm64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
MicroDiff-MatDesign Docker Build Script

Usage: $0 [OPTIONS] [TARGET]

Targets:
  development  - Build development image with all tools
  production   - Build production image (default)
  testing      - Build testing image
  docs         - Build documentation image
  benchmark    - Build benchmark image
  all         - Build all targets

Options:
  -h, --help              Show this help
  -t, --tag TAG          Additional tag for the image
  -r, --registry URL     Container registry URL
  --no-cache             Disable build cache
  --push                 Push images to registry
  --platform PLATFORMS   Target platforms (default: linux/amd64,linux/arm64)
  --scan                 Run security scan after build
  --parallel             Build stages in parallel
  --cuda-version VER     CUDA version (default: 11.8)
  --python-version VER   Python version (default: 3.11)
  --verbose              Verbose output

Examples:
  $0 production --push                    # Build and push production image
  $0 development --tag latest --scan      # Build dev image with security scan
  $0 all --platform linux/amd64          # Build all targets for AMD64 only
EOF
}

# Parse command line arguments
TARGET="production"
ADDITIONAL_TAGS=()
NO_CACHE=false
PUSH=false
SCAN=false
PARALLEL=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--tag)
            ADDITIONAL_TAGS+=("$2")
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --platform)
            BUILD_PLATFORM="$2"
            shift 2
            ;;
        --scan)
            SCAN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        development|production|testing|docs|benchmark|all)
            TARGET="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate dependencies
check_dependencies() {
    local deps=("docker" "git")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is required but not installed"
            exit 1
        fi
    done
    
    # Check Docker Buildx
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is required for multi-platform builds"
        exit 1
    fi
}

# Get version information
get_version_info() {
    local git_hash=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local git_tag=$(git describe --tags --exact-match 2>/dev/null || echo "")
    local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    local build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "git_hash=$git_hash,git_tag=$git_tag,git_branch=$git_branch,build_date=$build_date"
}

# Build single target
build_target() {
    local target="$1"
    local image_tag="${REGISTRY}/${IMAGE_NAME}:${target}"
    
    log_info "Building $target image: $image_tag"
    
    local build_args=()
    build_args+=("--target" "$target")
    build_args+=("--platform" "$BUILD_PLATFORM")
    build_args+=("--build-arg" "CUDA_VERSION=$CUDA_VERSION")
    build_args+=("--build-arg" "PYTHON_VERSION=$PYTHON_VERSION")
    build_args+=("--build-arg" "BUILD_VERSION=$(get_version_info)")
    build_args+=("--tag" "$image_tag")
    
    # Add additional tags
    for tag in "${ADDITIONAL_TAGS[@]}"; do
        build_args+=("--tag" "${REGISTRY}/${IMAGE_NAME}:${tag}")
    done
    
    # Cache configuration
    if [[ "$NO_CACHE" == "false" ]]; then
        build_args+=("--cache-from" "type=registry,ref=${BUILD_CACHE_REGISTRY}:${target}")
        build_args+=("--cache-to" "type=registry,ref=${BUILD_CACHE_REGISTRY}:${target},mode=max")
    fi
    
    # Push configuration
    if [[ "$PUSH" == "true" ]]; then
        build_args+=("--push")
    else
        build_args+=("--load")
    fi
    
    # Verbose output
    if [[ "$VERBOSE" == "true" ]]; then
        build_args+=("--progress" "plain")
    fi
    
    # Execute build
    if ! docker buildx build "${build_args[@]}" .; then
        log_error "Failed to build $target image"
        return 1
    fi
    
    log_success "Successfully built $target image"
    
    # Security scanning
    if [[ "$SCAN" == "true" ]] && [[ "$PUSH" == "false" ]]; then
        scan_image "$image_tag"
    fi
}

# Security scanning with Trivy
scan_image() {
    local image="$1"
    
    log_info "Scanning image for vulnerabilities: $image"
    
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL --exit-code 1 "$image"
        log_success "Security scan passed"
    else
        log_warning "Trivy not installed, skipping security scan"
    fi
}

# Build all targets
build_all() {
    local targets=("development" "production" "testing" "docs" "benchmark")
    
    if [[ "$PARALLEL" == "true" ]]; then
        log_info "Building all targets in parallel"
        
        local pids=()
        for target in "${targets[@]}"; do
            build_target "$target" &
            pids+=("$!")
        done
        
        # Wait for all builds to complete
        local failed=false
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=true
            fi
        done
        
        if [[ "$failed" == "true" ]]; then
            log_error "One or more parallel builds failed"
            return 1
        fi
    else
        log_info "Building all targets sequentially"
        
        for target in "${targets[@]}"; do
            if ! build_target "$target"; then
                return 1
            fi
        done
    fi
    
    log_success "All targets built successfully"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup logic here
}

# Trap cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    log_info "MicroDiff-MatDesign Docker Build"
    log_info "Target: $TARGET"
    log_info "Registry: $REGISTRY"
    log_info "Platform: $BUILD_PLATFORM"
    log_info "CUDA Version: $CUDA_VERSION"
    log_info "Python Version: $PYTHON_VERSION"
    
    check_dependencies
    
    # Create buildx builder if not exists
    if ! docker buildx ls | grep -q "microdiff-builder"; then
        log_info "Creating multi-platform builder"
        docker buildx create --name microdiff-builder --driver docker-container --use
        docker buildx inspect --bootstrap
    else
        docker buildx use microdiff-builder
    fi
    
    case "$TARGET" in
        all)
            build_all
            ;;
        *)
            build_target "$TARGET"
            ;;
    esac
    
    log_success "Build process completed successfully!"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
