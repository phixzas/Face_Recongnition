# FaceUnlock: Local Face Verification for Windows

**Author:** Mayowa Philips Olusanjo
[GitHub](https://github.com/phixzas) · [LinkedIn](https://www.linkedin.com/in/mayowa-philips-olusanjo-982663152)

FaceUnlock is a local Windows desktop prototype that verifies a person through a webcam using face detection and an enrolled facial embedding. It provides a live preview, detects the largest face in view, captures a verification frame, and reports whether the result passes the configured similarity threshold.

The project is intended for learning and local experimentation with computer vision. It is not a replacement for Windows authentication.

## Features

- Live webcam preview with face-detection status and bounding box
- Automatic capture after a five-second countdown, with Space-to-capture support
- YuNet face detection through OpenCV
- SFace face embeddings through OpenCV
- Cosine-similarity verification against a locally stored enrollment embedding
- Standalone Windows executable packaging with PyInstaller

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- A working webcam
- A well-lit, front-facing enrollment embedding generated with the SFace model

Install the Python dependencies from PowerShell:

```powershell
python -m pip install -r requirements-face-unlock.txt
```

## Run From Python

From the project directory, run:

```powershell
python face_unlock.py
```

Look at the webcam during the countdown. Press Space to capture immediately, or press Q or Esc to cancel. The application prints the similarity score and either `UNLOCKED` or `LOCKED` before exiting.

## Enrollment Data

The application requires `enrollment_embedding.npy` beside the Python script, or bundled into the executable. This file contains the reference embedding used for verification and is intentionally excluded from the repository because it is biometric data. The embedding must be generated with the same SFace model used by the application; older MobileNetV2 embeddings are not compatible.

Create or update the enrollment embedding with the enrollment workflow in `Untitled.ipynb`, then run the application again. Capture images in consistent lighting with the face clearly visible and facing the camera.

## Build The Windows Application

The repository includes a PowerShell build script. Run it from the project directory:

```powershell
.\build_face_unlock.ps1
```

The script installs the dependencies and creates a one-directory build at:

```text
dist\FaceUnlock\FaceUnlock.exe
```

Double-click `FaceUnlock.exe` to start the webcam verification application. The detector model and enrollment embedding are bundled into the build.

## Local Data And Privacy

The following files are local runtime or biometric data and should not be committed to a public repository:

- `enrollment_embedding.npy` stores the enrolled face embedding.
- `enrollment_images\` may contain captured enrollment photos.
- `face_detection_yunet_2023mar.onnx` is the local face-detector model.
- `build\` and `dist\` contain generated PyInstaller output.

Keep enrollment data private and delete it when it is no longer needed. The application is designed to process the webcam locally; it does not provide a cloud service or remote authentication endpoint.

## Important Security Limitation

This application can verify a face after Windows has already started. A normal Python program or `.exe` cannot unlock the Windows login screen. Windows login facial recognition requires Windows Hello, a compatible infrared camera, and the Windows security infrastructure behind it.

For actual Windows sign-in, use **Settings -> Accounts -> Sign-in options** to configure Windows Hello Face, a PIN, fingerprint, or a security key. If Windows reports that no camera is compatible with Windows Hello Face, the current webcam does not have the required infrared hardware.

## Project Structure

```text
face_unlock.py                  Webcam verification application
build_face_unlock.ps1           PyInstaller build script
requirements-face-unlock.txt    Python dependencies
FaceUnlock.spec                 PyInstaller configuration
face_detection_yunet_2023mar.onnx
                                YuNet face-detector model
face_recognition_sface_2021dec.onnx
                                SFace face-recognition model
enrollment_embedding.npy        Local enrollment data
```

## License

