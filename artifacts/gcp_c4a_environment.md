# GCP C4A environment audit

The guest metadata confirms `c4a-standard-16`, Spot/preemptible provisioning, and zone `europe-west4-c`. The OS exposes **16 online Neoverse-V2 CPUs as 16 cores, one thread per core, one socket, and one NUMA node**. `nproc`, `_NPROCESSORS_ONLN`, the effective cpuset, and `lscpu` all agree. No SMT siblings are exposed, so all 16 visible vCPUs behave as independent physical cores from the guest perspective.

The machine has 62 GiB guest-visible RAM and no swap. The only block device is a 100 GB Hyperdisk Balanced NVMe-presented boot disk, with a 96 GB ext4 root filesystem. There is no separately attached persistent data disk. Training data and recovery checkpoints therefore live under the root mount. GCP preemption stops the Spot VM, but the disk auto-delete policy could not be queried with the instance's local credentials; Hugging Face recovery uploads are required as the off-instance safety layer.

Ubuntu 24.04.4 LTS runs kernel 7.0.0-1011-gcp on AArch64. CPU features include ASIMD/NEON, SVE/SVE2, BF16 and integer matrix multiply. Guest cpufreq governor and frequency files are not exposed.

The isolated `.venv` uses Python 3.12.3 and native CPU-only PyTorch 2.14.0+cpu. PyTorch reports oneDNN 3.12.0, OpenMP 4.5, SVE128 CPU capability, and MKLDNN availability. NumPy 2.5.2 reports OpenBLAS 0.3.34 with dynamic Neoverse-V2 selection. GCC/G++ 13.3, Clang 18.1.3, CMake 3.28.3, Ninja 1.11.1, and Git 2.43.0 are installed. The cgroup v2 scope has no CPU or RAM quota beyond cpuset 0-15.

The complete machine-readable audit is in `gcp_c4a_environment.json`. Repository validation after installation: 22 tests pass, `compileall` passes, and `git diff --check` passes.
