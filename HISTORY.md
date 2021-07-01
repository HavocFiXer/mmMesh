## 06-30-2021 (V1.0)

- Release the repository as following:
```
├── 0.preliminary
│   └── extract_SMPL_model.py
├── 1.mmWave_data_capture
│   ├── capture.py
│   └── steaming.py
├── 2.deep_model
│   ├── data.py
│   ├── infer_model.py
│   ├── network.py
│   ├── smpl_utils_extend.py
│   └── train_model.py
├── HISTORY.md
├── LICENSE
├── README.md
└── .gitignore
```
- Release code in folder `0.preliminary` to generate the SMPL model in the training of mmMesh model.
- Release code in folder `1.mmWave_data_capture` for real-time mmWave radar steaming.
- Release code in folder `2.deep_model` for the training and inference of the deep model in mmMesh.
