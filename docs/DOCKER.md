# Docker Support

PyAI-Slayer includes Docker support for containerized development and test execution.

## 🐳 Docker Images

### Development Image (`Dockerfile`)

Full-featured image for development:
- Python 3.11
- All system dependencies
- Playwright browsers installed
- Development dependencies included

### CI/CD Image (`Dockerfile.test`)

Lightweight image for CI/CD:
- Minimal dependencies
- Optimized for test execution
- Faster build times

## 📦 Quick Start

### Build Docker Image

```bash
make docker-build
# Or manually:
docker build -t pyai-slayer:latest .
```

### Run Container

```bash
make docker-run
# Or manually:
docker run --rm -it \
  -v $(PWD):/app \
  -v $(PWD)/test-results:/app/test-results \
  -e ENVIRONMENT=sandbox \
  pyai-slayer:latest /bin/bash
```

## 🚀 Docker Compose

Docker Compose provides multiple services for different use cases:

### Development Container

Keep container running for development:

```bash
docker-compose up -d pyai-slayer
docker-compose exec pyai-slayer /bin/bash
```

### Run Tests

```bash
# Run all tests
make docker-test
# Or:
docker-compose --profile test run --rm test-runner

# Run unit tests only
make docker-unit
# Or:
docker-compose --profile unit run --rm unit-tests
```

### Run Linting

```bash
make docker-lint
# Or:
docker-compose --profile lint run --rm lint
```

### Interactive Shell

```bash
make docker-shell
# Or:
docker-compose run --rm pyai-slayer /bin/bash
```

## 📝 Available Services

### `pyai-slayer`
Main development container:
- Keeps running for interactive use
- Mounts source code
- Mounts test results and reports

### `test-runner`
Runs all tests:
- Executes pytest with standard reporting
- Exits after completion
- Profile: `test`

### `unit-tests`
Runs unit tests only:
- Executes unit tests with coverage
- Generates HTML coverage report
- Profile: `unit`

### `lint`
Runs code quality checks:
- Ruff linting
- Ruff formatting check
- MyPy type checking
- Profile: `lint`

## 🔧 Configuration

### Environment Variables

Set via `docker-compose.yml` or command line:

```bash
# Set environment
export ENVIRONMENT=production
export USE_CUDA=true

# Run with environment
docker-compose run --rm -e ENVIRONMENT=production pyai-slayer pytest
```

### Volumes

Default volumes mounted:
- `.:/app` - Source code
- `./test-results:/app/test-results` - Test results
- `./reports:/app/reports` - Reports

### Networks

All services use `pyai-network` bridge network.

## 🧪 Running Tests in Docker

### All Tests

```bash
docker-compose --profile test run --rm test-runner
```

### Specific Test File

```bash
docker-compose run --rm pyai-slayer pytest tests/unit/test_ai_validator.py -v
```

### With Coverage

```bash
docker-compose --profile unit run --rm unit-tests
```


## 🛠️ Development Workflow

### 1. Start Development Container

```bash
docker-compose up -d pyai-slayer
```

### 2. Access Container

```bash
docker-compose exec pyai-slayer /bin/bash
```

### 3. Run Commands

```bash
# Inside container
pytest tests/unit/ -v
make lint
make type-check
```

### 4. Stop Container

```bash
docker-compose down
```

## 🧹 Cleanup

### Clean Docker Resources

```bash
make docker-clean
# Or:
docker-compose down -v
docker system prune -f
```

### Remove Images

```bash
docker rmi pyai-slayer:latest
```

## 📊 CI/CD Integration

### GitHub Actions

Use `Dockerfile.test` for CI:

```yaml
- name: Run tests in Docker
  run: |
    docker build -f Dockerfile.test -t pyai-slayer-test .
    docker run --rm pyai-slayer-test
```

### Local CI Simulation

```bash
docker build -f Dockerfile.test -t pyai-slayer-test .
docker run --rm pyai-slayer-test
```

## 🐛 Troubleshooting

### Permission Issues

If you encounter permission issues:

```bash
# Fix ownership
sudo chown -R $USER:$USER test-results reports
```

### Browser Issues

If Playwright browsers fail:

```bash
# Reinstall browsers in container
docker-compose exec pyai-slayer playwright install --with-deps chromium
```

### Volume Mount Issues

Ensure volumes are properly mounted:

```bash
# Check volumes
docker-compose config
```

## 🔗 Related Documentation

- [Getting Started](getting_started.rst)
- [Testing Guide](contributing.rst)
