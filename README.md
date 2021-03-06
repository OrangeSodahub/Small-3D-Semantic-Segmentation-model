# Small-3D-Semantic-Segmentation-model
Implement a small 3D Semantic Segmentation model using Sparse Convolutions and overfit (train and eval on the same data) on the given dataset.


Environment: **ubuntu20.04**,**python3.8**

## INSTALL
Clone the repository and the 'data.npy' is located in **~/dataset/raw/** by default

```bash
git clone https://OrangeSodahub.com/Open3D_assignment.git
```

Then install additional required packages

```bash
cd ~/
python install -r requirements
```

## Usage
### Generate Datasets
First generate train dataset and test dataset from given dataset **data.npy**. Default data file path is given.

```bash
cd ~/
python tools/generate_dataset.py
```

To specify the path to config file, use config **--config**. And it also support the visualization of the raw dataset via open3d, if you want just add **--visible**

```bash
python tools/generate_dataset.py --config /path/to/config/file --visible True
```

### Train
Train using the given data file.

```bash
python train.py
```


### Test
Test on the given data file using trained model.

```bash
python test.py
```
