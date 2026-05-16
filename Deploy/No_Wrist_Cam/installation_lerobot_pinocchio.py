# 1. Umgebung erstellen (wie du es bereits hast)
conda create -y -n lerobot_pin python=3.12
conda activate lerobot_pin

# 2. ffmpeg und pinocchio über conda-forge installieren
conda install -y -c conda-forge ffmpeg pinocchio

# 3. lerobot über pip
pip install lerobot