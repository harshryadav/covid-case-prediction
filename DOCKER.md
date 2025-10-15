# Docker Guide

Complete guide for running the project in Docker containers.

---

## 🐳 Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Build and start
docker-compose build
docker-compose up -d

# 2. Run pipeline
docker-compose exec covid-forecasting python run_pipeline.py

# 3. View results (on your host machine!)
open results/baseline_forecasts.png

# 4. Stop when done
docker-compose down
```

### Using Docker Directly

```bash
# Build image
docker build -t covid-forecasting .

# Run interactively
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  covid-forecasting bash

# Inside container:
python run_pipeline.py
```

---

## 📋 Prerequisites

- **Docker Desktop** installed ([download](https://www.docker.com/products/docker-desktop))
- **4GB RAM** allocated to Docker
- **~2GB disk space** for image + data

**Check Docker is installed:**
```bash
docker --version
docker-compose --version
```

---

## 🏗️ Docker Setup

### Dockerfile Structure

```dockerfile
# Multi-stage build for optimization
Stage 1: Python 3.9 base + system dependencies
Stage 2: Install Python packages (requirements.txt)
Stage 3: Copy application code
```

**Benefits**:
- ✅ Smaller final image (~1.5GB)
- ✅ Faster rebuilds (cached layers)
- ✅ Production-ready

### Volume Mounts

The `docker-compose.yml` configuration mounts:

| Local Path | Container Path | Mode | Purpose |
|------------|----------------|------|---------|
| `./data` | `/app/data` | Read-only | Input datasets |
| `./results` | `/app/results` | Read/Write | Plots & metrics |
| `./models` | `/app/models` | Read/Write | Model checkpoints |

**Why volumes?**
- ✅ Data persists between container restarts
- ✅ Results saved to your host machine
- ✅ No need to copy files in/out

---

## 🚀 Common Commands

### Build & Run

```bash
# Build image
docker-compose build

# Start container (detached)
docker-compose up -d

# Start container (with logs)
docker-compose up

# Rebuild (if you changed code/requirements)
docker-compose build --no-cache
```

### Execute Commands

```bash
# Run full pipeline
docker-compose exec covid-forecasting python run_pipeline.py

# Individual scripts
docker-compose exec covid-forecasting python src/models/train_baseline.py

# Interactive shell
docker-compose exec covid-forecasting bash
```

### Container Management

```bash
# List running containers
docker ps

# View logs
docker-compose logs -f covid-forecasting

# Stop container
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Cleanup

```bash
# Remove stopped containers
docker-compose down

# Remove image
docker rmi covid-forecasting

# Clean up everything (careful!)
docker system prune -a
```

---

## 📦 Inside the Container

```
/app/
├── data/                    # Your datasets (mounted, read-only)
├── src/                     # Source code
├── results/                 # Output (mounted, writable)
├── models/                  # Saved models (mounted, writable)
├── requirements.txt
└── run_pipeline.py
```

---

## 🎯 Usage Scenarios

### Scenario 1: Quick Test

```bash
docker-compose up -d
docker-compose exec covid-forecasting python src/visualization/create_eda_plots.py
open results/eda_visualization.png
docker-compose down
```

### Scenario 2: Full Pipeline

```bash
docker-compose up -d
docker-compose exec covid-forecasting python run_pipeline.py
docker-compose down
```

### Scenario 3: Interactive Development

```bash
# Start container
docker-compose up -d

# Enter shell
docker-compose exec covid-forecasting bash

# Inside container, run multiple commands:
python src/data_processing/preprocess.py
python src/models/train_baseline.py
python src/models/train_deepar.py

# Exit
exit

# Stop
docker-compose down
```

### Scenario 4: Jupyter Notebook (Optional)

```bash
# Start with Jupyter profile
docker-compose --profile jupyter up -d jupyter

# View token in logs
docker-compose logs jupyter

# Access at: http://localhost:8889
# Copy token from logs
```

---

## 🐛 Troubleshooting

### Issue: Build fails with "No space left on device"

**Solution:**
```bash
docker system prune -a
docker volume prune
```

### Issue: Permission denied on mounted volumes (Linux)

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER results/ models/

# Or run as your user
docker run --user $(id -u):$(id -g) ...
```

### Issue: "requirements.txt: not found"

**Solution:**
```bash
# Make sure you're in project root
cd /Users/HarshYadav/Documents/Misc/covid-case-prediction
docker-compose build
```

### Issue: GluonTS/MXNet crashes

**Solution:**
Already configured in `docker-compose.yml`:
```yaml
environment:
  - MXNET_ENGINE_TYPE=NaiveEngine
```

### Issue: Container exits immediately

**Solution:**
```bash
# Check logs
docker-compose logs covid-forecasting

# Run with interactive terminal
docker run -it covid-forecasting bash
```

### Issue: Slow performance

**Tips:**
- Allocate more CPU/RAM to Docker Desktop (Preferences → Resources)
- Use GPU if available (requires nvidia-docker setup)
- Close other Docker containers

---

## ⚡ Performance Tips

### Build Time
- **First build**: ~5-10 minutes (downloads dependencies)
- **Subsequent builds**: ~30 seconds (uses cache)

### Runtime
- **Baseline models**: <1 second
- **DeepAR (10 epochs, CPU)**: 2-5 minutes
- **DeepAR (10 epochs, GPU)**: 30-60 seconds

### Optimize Build
```dockerfile
# In Dockerfile, requirements are installed in separate layer
# Change only requirements.txt → Only that layer rebuilds
# Change code → Only application layer rebuilds
```

---

## 🎓 Best Practices

### 1. Development Workflow

```bash
# Start once
docker-compose up -d

# Iterate (container stays running)
docker-compose exec covid-forecasting python src/models/train_baseline.py
docker-compose exec covid-forecasting python src/models/train_deepar.py

# Stop when done
docker-compose down
```

### 2. Data Management

```bash
# Data directory is read-only by default (safer)
# If you need to download data inside container:
docker-compose exec covid-forecasting bash
# Inside: Download to /app/data (but it's read-only)
# Better: Download on host, mount to container
```

### 3. Code Changes

```bash
# Option A: Edit on host, rebuild
vim src/models/train_deepar.py
docker-compose build
docker-compose up -d

# Option B: Mount src/ as volume (for development)
# Add to docker-compose.yml:
#   - ./src:/app/src
```

---

## 🚀 Production Deployment

### Push to Registry

```bash
# Tag image
docker tag covid-forecasting:latest your-registry.com/covid-forecasting:v1.0

# Push to registry
docker push your-registry.com/covid-forecasting:v1.0
```

### Use on Another Machine

```bash
# Pull image
docker pull your-registry.com/covid-forecasting:v1.0

# Run
docker run -v ./data:/app/data -v ./results:/app/results \
  your-registry.com/covid-forecasting:v1.0 python run_pipeline.py
```

---

## 📊 Docker vs Local

| Aspect | Docker | Local |
|--------|--------|-------|
| **Setup Time** | 5-10 min (first time) | 2-3 min |
| **Reproducibility** | ✅ Perfect | ⚠️ Depends on environment |
| **Isolation** | ✅ Complete | ❌ Uses system Python |
| **Portability** | ✅ Runs anywhere | ❌ May have dependency issues |
| **Performance** | ⚠️ Slight overhead (~5%) | ✅ Native |
| **Debugging** | ⚠️ Requires exec/attach | ✅ Direct |

**Recommendation**: Use Docker for production, either works for development.

---

## 📚 Further Reading

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🎯 Quick Reference Card

```bash
# BUILD
docker-compose build                    # Build image
docker-compose build --no-cache         # Force rebuild

# RUN
docker-compose up -d                    # Start detached
docker-compose up                       # Start with logs

# EXECUTE
docker-compose exec covid-forecasting bash              # Shell
docker-compose exec covid-forecasting python run_pipeline.py  # Run script

# MANAGE
docker-compose ps                       # List containers
docker-compose logs -f                  # View logs
docker-compose down                     # Stop & remove
docker-compose restart                  # Restart

# CLEAN
docker system prune -a                  # Remove unused images/containers
docker volume prune                     # Remove unused volumes
```

---

**Ready to containerize? Run:**

```bash
docker-compose build && docker-compose up -d
```

