# Dataset preparation pipeline

This package contains dataset management, annotation conversion and evaluation logic for:

- CrowdHuman for person detection
- MOT17 for multi-object tracking evaluation
- SCUT-HEAD for classroom head detection/counting

Important boundaries:

- No automatic dataset downloads are performed here.
- No training is started from this package.
- Activity recognition and fall-like event training remain separate future tasks and require temporal datasets that are not part of the above three sources.
- Each dataset is kept isolated because they have different labels, annotation formats and evaluation targets.

Directory layout:

- raw/: downloaded or mounted dataset archives and raw image folders
- processed/: converted YOLO labels and normalized dataset manifests
- reports/: generated CSV/JSON/summary reports and EDA outputs
