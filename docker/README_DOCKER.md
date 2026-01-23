# LeafMachine2 Docker Workflow


docker compose run --rm --user "$(id -u):$(id -g)" lm2-cpu python test_cpu_only.py



This document describes the **official, supported Docker-based installation** for LeafMachine2.
Using Docker is the **recommended installation method** for new users, as it avoids Python,
CUDA, and dependency conflicts.

---

## What Docker Does (Plain English)

Docker packages LeafMachine2 together with:
- the correct Python version
- all required libraries
- the correct PyTorch setup

This means:
- no manual Python installs
- no virtual environments
- no dependency conflicts

You run LeafMachine2 exactly the same way on **Windows, macOS, and Linux**.

---

## Supported Docker Modes

| Mode | Works On | GPU Required |
|----|----|----|
| **CPU** | Windows / macOS / Linux | ❌ |
| **GPU (NVIDIA)** | Linux (best), Windows (WSL2) | ✅ |
| **GPU on macOS** | ❌ Not supported | ❌ |

> macOS users must use **CPU mode**. Apple GPUs are not supported.

---

## Folder Layout (Important)

After cloning the repo, the Docker setup lives in:

```
LeafMachine2/
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── entrypoint.sh
│   ├── compose.yml
│   └── README_DOCKER.md
├── demo/
│   ├── demo_images/
│   └── demo_output/
├── bin/
└── leafmachine2/
```

Docker mounts the repository root into the container at `/app`.
All outputs and downloaded models remain in the repository.

---

## STEP 1 — Install Docker

### Windows / macOS
1. Install **Docker Desktop**  
   https://www.docker.com/products/docker-desktop
2. Start Docker Desktop
3. Verify:
   ```bash
   docker --version
   docker compose version
   ```

---

### Linux

```bash
sudo apt-get remove -y docker docker.io containerd runc docker-compose || true
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker --version
docker compose version
docker run --rm hello-world
```

---

## STEP 2 — Get LeafMachine2

```bash
git clone https://github.com/Gene-Weaver/LeafMachine2.git
cd LeafMachine2
```

---

## STEP 3 — CPU Mode

```bash
cd docker
docker compose build lm2-cpu
docker compose run --rm lm2-cpu
```

Outputs:
- Results: `demo/demo_output/`
- Models: `bin/`

---

## STEP 4 — GPU Mode (NVIDIA)

```bash
docker compose build lm2-gpu
docker compose run --rm lm2-gpu
```

---

## Linux File Ownership

If output files appear locked:
```bash
sudo chown -R $USER:$USER demo bin
```

---

This Docker workflow is the **official, reproducible installation path** for LeafMachine2.