# mmMesh

---

Last Update Date: Jun 30 2021

![mmMesh Model Image](https://havocfixer.github.io/mmMesh/resources/model.png)

This repository includes:
- The UDP-based real-time TI mmWave radar data capture code of mmMesh system.
- The code for the point cloud generation from binary file of mmWave radar (Note that the binary file should not have packet head).
- The Pytorch implementation of mmMesh deep model.

The structure of the repository is listed as following:
```
├── 0.preliminary
│   └── extract_SMPL_model.py
├── 1.mmWave_data_capture
│   ├── capture.py
│   └── steaming.py
├── 2.point_cloud_generation
│   ├── configuration.py
│   └── pc_generation.py
├── 3.deep_model
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

---

## 0. Preliminary

Due to the copyright issue, the SMPL model file cannot be provided. You need to access [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/) to access the SMPL model [download page](https://smpl.is.tue.mpg.de/downloads). Then, dowload the model package: version 1.0.0 for Python 2.7 (10 shape PCs) and extract the file. You need to use **extract_SMPL_model.py** (use Python 3 to run the code) in **0.preliminary** to generate SMPL model used in the training of mmMesh. For example, if you extract the file in the same folder with **extract_SMPL_model.py**, you can use:
```bash
python extract_SMPL_model.py ./SMPL_python_v.1.0.0/smpl/models/
```
to generate two files: **smpl_f.pkl** and **smpl_m.pkl**.
**Note**: you need to install [Chumpy](https://github.com/mattloper/chumpy) to run the code.

## 1. Real-time mmWave Data Capturing

In **steaming.py**, the methods in the class will automatically collect the packets from the radar and assemble them into frames. It also allows you to access these frames using `getFrame` method. As an example, if you want to capture the data from the mmWave radar for 5 mins and store them, just run:
``` bash
python capture.py 5
```

## 2. Point Cloud Generation from Binary mmWave Data

The methods in **pc_generation** allows you to generate the point cloud from the binary mmWave radar.For example, if you want to generate the point cloud data from `test.bin` for 10 frames, try:
``` bash
python pc_generation.py test.bin 10
```

**Note**: to successfully read the data from the binary file, you should change the radar configuration to generation the binary file without the packet head.

## 3. mmMesh Deep Model

Model training:
``` bash
python train_model.py
```

Model inference:
``` bash
python infer_model.py
```

**Note**: to run the code, you need to move the **smpl_f.pkl** and **smpl_m.pkl** generated in step 0.Preliminary to the current folder.

## Citation

If you find our work useful in your research, please cite:
``` BibTeX
@inproceedings{xue2021mmmesh,
  title={mmMesh: towards 3D real-time dynamic human mesh construction using millimeter-wave},
  author={Xue, Hongfei and Ju, Yan and Miao, Chenglin and Wang, Yijiang and Wang, Shiyang and Zhang, Aidong and Su, Lu},
  booktitle={Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services},
  pages={269--282},
  year={2021}
}
```

## Acknowledgements

- The Pytorch version of SMPL code is partly borrowed from [https://github.com/CalciferZh/SMPL](https://github.com/CalciferZh/SMPL).
- The grouping code of point cloud in the Anchor Point Module is partly borrowed from [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
