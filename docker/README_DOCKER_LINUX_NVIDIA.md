# LeafMachine2 Docker GPU Setup (NVIDIA) — Linux-Focused Guide

This guide captures the Linux-side adjustments required to run the **LeafMachine2 GPU Docker image** using **Docker Engine + NVIDIA Container Toolkit**. It is designed to prevent the two common failure modes you hit:

1) `could not select device driver "" with capabilities: [[gpu]]`  
2) `/entrypoint.sh: ... exec: python: not found`

It also includes a **verification ladder** so you can isolate where GPU enablement breaks.

---

## Scope and assumptions

- Host OS: **Ubuntu 22.04** (or compatible Debian-based distro)
- GPU: **NVIDIA** (RTX/Quadro/Data Center)
- Docker: **Docker Engine** (not Podman)
- You have already cloned the LeafMachine2 repo and have `docker/compose.yml`, `docker/Dockerfile.gpu`, etc.

---

## Quick verification ladder (use in order)

Run these on the **host**, not inside the container:

### 1) Verify the GPU is healthy on the host
```bash
nvidia-smi
```
You should see your GPUs listed with a driver version.

### 2) Verify Docker is installed and functional
```bash
docker --version
docker compose version
docker run --rm hello-world
```

### 3) Verify Docker can see GPUs (NVIDIA runtime configured)
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```
If this works, Docker + NVIDIA runtime are correct. If it fails, fix the host runtime before debugging LeafMachine2.

---

## Step A — Install Docker Engine (Ubuntu 22.04)

If Docker is already installed and working, you can skip this section.

```bash
sudo apt-get remove -y docker docker.io containerd runc docker-compose || true
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker "$USER"
newgrp docker
```

Re-test:
```bash
docker run --rm hello-world
```

---

## Step B — Install NVIDIA Container Toolkit (the critical Linux adjustment)

This is the missing piece that causes:

`could not select device driver "" with capabilities: [[gpu]]`

Use NVIDIA’s official install flow. (The `nvidia-ctk runtime configure` step matters.) citeturn0search0

### 1) Add the NVIDIA repo and install
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
sudo chmod a+r /etc/apt/keyrings/nvidia-container-toolkit.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 2) Configure Docker runtime + restart Docker
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

NVIDIA documents this exact configuration sequence. citeturn0search0

### 3) Validate runtime with a CUDA container
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If this still fails:
- Confirm you are using **Docker Engine**, not a rootless alternative.
- Confirm the Docker daemon restarted successfully:
  ```bash
  systemctl status docker --no-pager
  ```
- Confirm your user is in the `docker` group:
  ```bash
  groups | grep docker || echo "NOT in docker group"
  ```

---

## Step C — LeafMachine2 repo layout and where to run commands

From your repo root:
```bash
cd /datac/LM2/LeafMachine2/docker
ls -lah
```

You should see:
- `compose.yml`
- `Dockerfile.gpu`
- `entrypoint.sh`
- `README_DOCKER.md`

---

## Step D — Build the GPU image

From `LeafMachine2/docker`:
```bash
docker compose build --no-cache lm2-gpu
```

Note: your log shows a large build context transfer (multiple GB). If builds feel slow, consider adding a `.dockerignore` at repo root to exclude bulky folders (e.g., datasets, outputs) from the build context.

---

## Step E — Run GPU mode (recommended commands)

### Preferred (most explicit) test command
```bash
docker compose run --rm --user "$(id -u):$(id -g)" lm2-gpu python3 test.py
```

### Optional: run without overriding user
If you do not care about host-side file ownership:
```bash
docker compose run --rm lm2-gpu python3 test.py
```

---

## Compose GPU configuration (what to put in compose.yml)

Docker Compose has documented GPU support; the canonical path is to request GPUs in the service definition. citeturn0search1

You currently use this pattern:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

This can work in modern Compose implementations, but two practical points matter:

1) Some environments ignore `deploy:` (historically it was Swarm-oriented).  
2) The most portable approach is to also include a direct GPU request when supported.

### Practical recommendation for portability

Keep your `deploy.resources.reservations.devices` stanza and consider adding `gpus: all` when your Compose version supports it. Docker’s Compose GPU guidance covers the supported patterns. citeturn0search1

Example:

```yaml
services:
  lm2-gpu:
    # ...
    gpus: all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

If your Compose version errors on `gpus: all`, remove that line and rely on `deploy:` (or upgrade Docker Compose).

---

## Fixing `/entrypoint.sh: ... python: not found`

You hit:
`/entrypoint.sh: line 15: exec: python: not found`

This happened because:
- Your Dockerfile installs **python3.11** and sets the `python3` alternative,
- But it does not guarantee that the `python` alias exists (many modern distros do not ship `/usr/bin/python` by default),
- Your entrypoint prints:
  ```bash
  python --version
  ```
  and you initially invoked `python test.py`.

### Fix option 1 (recommended): Standardize on python3 everywhere
- Run:
  ```bash
  docker compose run --rm --user "$(id -u):$(id -g)" lm2-gpu python3 test.py
  ```
- In `compose.yml`, keep:
  ```yaml
  command: ["python3", "test.py"]
  ```

### Fix option 2 (recommended for ergonomics): Create a `python` alias inside the image

Add one line to `docker/Dockerfile.gpu` after you set the python3 alternative:

```dockerfile
RUN ln -sf /usr/bin/python3 /usr/bin/python
```

or install the canonical Debian/Ubuntu shim:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends python-is-python3 \
 && rm -rf /var/lib/apt/lists/*
```

Either will ensure `python` works consistently.

### Fix option 3: Update entrypoint to use python3 for the version print

Change:
```bash
echo "  Python: $(python --version 2>/dev/null || true)"
```

to:
```bash
echo "  Python: $(python3 --version 2>/dev/null || true)"
```

This avoids dependence on the `python` alias, even if a user runs `python3 ...`.

---

## Common failure modes and what they mean

### 1) `could not select device driver "" with capabilities: [[gpu]]`

Meaning: Docker cannot access NVIDIA GPUs.  
Fix: Install NVIDIA Container Toolkit + configure runtime + restart Docker. citeturn0search0

### 2) `docker run --rm --gpus all ... nvidia-smi` fails, but `nvidia-smi` works on host

Meaning: Host drivers are OK, but Docker is not configured for NVIDIA runtime.  
Fix: Step B (toolkit + `nvidia-ctk runtime configure`). citeturn0search0

### 3) Container starts, but `python` missing

Meaning: Image has python3 installed, but not the `python` alias.  
Fix: Use `python3`, or add alias (above).

---

## Recommended “golden” command sequence (Ubuntu 22.04)

From repo root:

```bash
cd docker

# (one-time) verify Docker sees GPUs:
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

# build:
docker compose build --no-cache lm2-gpu

# run:
docker compose run --rm --user "$(id -u):$(id -g)" lm2-gpu python3 test.py
```

---

## References

- NVIDIA Container Toolkit install + Docker runtime configuration (includes `nvidia-ctk runtime configure`). citeturn0search0  
- Docker Compose GPU support guidance. citeturn0search1