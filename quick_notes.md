- “RBAC = identity-based, needs a role.
Connection string = key-based, works anywhere.”
- CUDA (Compute Unified Device Architecture) is NVIDIA’s programming interface that lets your Python libraries (like PyTorch or Whisper) talk directly to the GPU.
- confirm rg: `az group show -n rg-amia-ml -o table`
- list all vms: `az vm list-sizes --location eastus -o table`
- ssh arthur@172.191.111.144

| Alias | Azure Name | Purpose | Specs | State |
|--------|-------------|----------|--------|--------|
| `vm-amia-preproc` | `vm-amia-t4-cpu` | CPU preprocessing | D4s_v5 | Deallocated |
| `vm-amia-gpu` | `vm-amia-t4` | GPU inference/fine-tuning | NC4as_T4_v3 | On demand |


## Remote-SSH
- GPU VM: `ssh amia-gpu` (alias in `~/.ssh/config`)
- CPU VM: `ssh amia-preproc`
- VS Code: *Remote-SSH → Connect to Host → amia-gpu*; open `/home/arthur/projects/amia2025-stt-benchmarking`
- Streamlit/Gradio: open http://localhost:8501 (tunneled via `LocalForward 8501 127.0.0.1:8501`)