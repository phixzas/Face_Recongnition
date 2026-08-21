# Face Recognition Prototype

A local Windows webcam face-verification prototype built with OpenCV, YuNet, TensorFlow MobileNetV2, and PyInstaller.

## Run locally

Install dependencies:

```powershell
python -m pip install -r requirements-face-unlock.txt
```

Run the app:

```powershell
python face_unlock.py
```

The app expects local files that are intentionally not committed because they contain biometric data:

- `enrollment_embedding.npy`
- `face_detection_yunet_2023mar.onnx`

The PyInstaller build also expects the enrollment and detector files beside the source during packaging.

## Important limitation

This prototype verifies a face after Windows has started. It cannot unlock the Windows login screen. Windows login facial recognition requires a Windows Hello-compatible infrared camera and a signed Windows Credential Provider. Use Windows Hello, a PIN, fingerprint, or security key for system sign-in.
