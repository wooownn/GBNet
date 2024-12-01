# GBNet: Gated Boundary-aware Network for Camouflaged Object Detection



## 1. Preface

- This repository provides code for "GBNet: Gated Boundary-aware Network for Camouflaged Object Detection" 

## 2. Trainning and Testing

1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n GBNet python=3.9`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
2. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing).
3. Downloading necessary weights:
   + downloading pvt_v2_b4 weights and move it into `./models/pvt_v2_b4.pth`[download link (github)](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth).
4. Next, for training and testing run`train.sh`.

## 3. Our Results.
  1. Our checkpoints can be found here [Google Drive](https://drive.google.com/file/d/1xPo7WKxkOyOZrZfnmqZu41ceuKCWxbw4/view?usp=drive_link).
  2. Our results can be found here [Google Drive](https://drive.google.com/file/d/1vKT8UvWk2MTyiC8bJDWy0Wtq6daexhhq/view?usp=drive_link).
