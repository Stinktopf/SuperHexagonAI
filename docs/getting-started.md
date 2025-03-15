# Getting Started

## Installation

### Prerequisites

Before getting started, ensure you have the following installed on your system:

- **Visual Studio Community Edition 2022**
- **Python 3.12 or lower**
- **Git**
- **Steam**
- **Pre-Neo Beta version of [Super Hexagon](https://store.steampowered.com/app/221640/Super_Hexagon/)**

---

### Step 1: Install C++ Build Tools

To install **Visual Studio Community Edition 2022**, follow these steps:

1. Download and run the [Visual Studio Installer](https://visualstudio.microsoft.com/de/downloads/).
2. During the installation process, ensure the following components are selected:
   - **MSVC C++ x64/x86 Build Tools**
   - **C++ CMake Tools for Windows**
   - **Windows 11 SDK**

---

### Step 2: Install Python Dependencies

Ensure you have Python installed (preferably **Python 3.12** for CUDA support). The following Python libraries are required:

```bash
ghapi==1.0.6
gymnasium==1.0.0
keyboard==0.13.5
matplotlib==3.10.1
nox==2025.2.9
numpy==2.2.3
opencv_python==4.10.0.84
pyrlhook==0.0.1
pytest==8.3.5
rich==13.9.4
setuptools==70.0.0
stable_baselines3==2.5.0
torch==2.7.0.dev20250228+cu128
```

To install them, run:

```bash
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

### Step 3: Setup Super Hexagon

To use this setup, you need **Super Hexagon in the Pre-Neo Steam Beta version**. Ensure the following settings are applied:

- **Run as Administrator:**  
  - Right-click `superhexagon.exe`, select `Properties`, go to the `Compatibility` tab, and check **"Run this program as an administrator"**.
  
- **Windowed Mode & VSync:**  
  - The game must be in **windowed mode**, and **VSync should be disabled**.

<div align="center">
  <img src="../images/steam.png" width="50%">
</div>

Additionally, ensure that both **the Python process and the game process are run with admin privileges**.

---

### Step 4: Clone the Repository

To download the necessary files, run:

```bash
git clone https://github.com/Stinktopf/SuperHexagonAI.git
cd SuperHexagonAI
```

---

### Step 5: Compile the DLL & Helper Executable

To compile the necessary binaries, execute:

```bash
cd RLHookLib
python compile_additional_binaries.py
```

---

### Step 6: Install the Library

Install the library globally using pip:

```bash
pip install .
```

### Step 7: Run the Trainer

Execute a trainer in an administrator command line:

```bash
python trainer_PPO_GYM_SB3.py
```