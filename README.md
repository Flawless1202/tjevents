# TJEvents: A general deep learning based event camera data processing toolbox

This repo contains various general tools for event camera data processing as well as many general modules or models
for deep learning on event camera, especially for graph-based methods or video reconstruction from events. 

Now we have implemented:
- The Event2Graph Dataset from the paper [Graph-Based Object Classification for Neuromorphic 
Vision Sensing](https://openaccess.thecvf.com/content_ICCV_2019/html/Bi_Graph-Based_Object_Classification_for_Neuromorphic_Vision_Sensing_ICCV_2019_paper.html),
 based on the [code](https://github.com/PIX2NVS/NVS2Graph.git).
- The FireNet from the paper [Fast Image Reconstruction with an Event Camera](https://openaccess.thecvf.com/content_WACV_2020/html/Scheerlinck_Fast_Image_Reconstruction_with_an_Event_Camera_WACV_2020_paper.html),
 based on the [code](https://github.com/cedric-scheerlinck/rpg_e2vid.git).
- The event data voxel grid representations from the paper [High Speed and High Dynamic Range Video with an Event Camera](https://arxiv.org/abs/1906.07165), 
based on the [code](https://github.com/uzh-rpg/rpg_e2vid).

## Requirements

- torch >= 1.4.0
- torchvision >= 0.5.0
- torch_geometric >= 1.5.0
- opencv-python
- scipy


## Install

a. Install PyTorch 1.5 and torchvision 0.6 following the [official instructions](https://pytorch.org/).

b. Install Pytorch Geometric 1.5 following the [official instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

c. Clone the repository.

```bash
git clone https://github.com/Flawless1202/tjevents.git
```

d. Install the other requirements.

```shell
pip install -r requirements.txt
```

d. Install.

```bash
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```

## Run a train example

a. Prepare the dataset: Download the ASL-Dataset here and unzip them all to `data/NVS2Graph/raw`.

b. Run the train example with spcific config

```bash
python examples/train_e2g.py --load_config ./config/dgcnn.yaml
```

