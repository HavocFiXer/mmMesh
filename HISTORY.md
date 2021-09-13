## 08-08-2021 (V1.2)
- Release `DataCaptureDemo_1843new.lua` in folder `1.mmWave_data_capture`.
- Add a simple tutorial to explain how to enable real-time data steaming.
- The structure of the added folder is as following:
```
├── 1.mmWave_data_capture
│   ├── DataCaptureDemo_1843new.lua
```


## 08-08-2021 (V1.1)
- Release code in folder `2.point_cloud_generation` for the point cloud generation from binary file of mmWave radar (no packet head)
- The structure of the added folder is as following:
```
├── 2.point_cloud_generation
│   ├── configuration.py
│   └── pc_generation.py
```
- Change the folder name from `2.deep_model` to `3.deep_model`

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
