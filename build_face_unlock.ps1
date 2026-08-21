$ErrorActionPreference = "Stop"
python -m pip install -r requirements-face-unlock.txt
python -m PyInstaller --noconfirm --clean --onedir --name FaceUnlock --add-data "enrollment_embedding.npy;." --add-data "face_detection_yunet_2023mar.onnx;." --add-data "face_recognition_sface_2021dec.onnx;." face_unlock.py
Write-Host "Built dist\FaceUnlock\FaceUnlock.exe"
