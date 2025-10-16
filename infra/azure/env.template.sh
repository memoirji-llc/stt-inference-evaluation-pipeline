# === Azure environment for AMIA2025 STT Benchmarking ===
# Fill in and copy to infra/azure/env.local.sh
export SUB=""                          # filled by `az account show` usually
export RG="rg-amia-ml"
export LOC="eastus"
export VM="vm-amia-t4"
export ADMIN="arthur"

# GPU SKU and base image
export SIZE="Standard_NC4as_T4_v3"
export IMAGE="Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest"

# OS disk
export OSDISKSIZE=128
export OSDISKTYPE="Premium_LRS"

# Networking
export VNET="${VM}-vnet"
export SUBNET="${VM}-subnet"
export NSG="${VM}-nsg"
export PUBLICIPSKU="Standard"