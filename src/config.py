
from typing_extensions import TypedDict
import yaml 
class Paramaters(TypedDict):
    image_size : int
    num_Channle : int
    num_Filters : int
    Dropout_Encoder: bool
    Dropout_Decoder: bool
    category: str
    attention: str
    num_hidden_layers : int
    initializer_range: float    
    num_classes : int
    monitor_cam:str        
    weight_decay : float
    lr: float
    batch_size: int
    num_workers: int 
    gpus: int
    Epoch: int
    device: int
    train_path: str
    val_path: str



def Intializer() -> Paramaters:
    config: Paramaters = {
          "image_size":256 ,
          "num_Filters":64,
          "Dropout_Encoder": False,
          "Dropout_Decoder":True,
          "category": "atruim",
          "num_Channle":1,
          "attention":"Unet",
          "num_hidden_layers":1,
          "num_classes": 1,
          "initializer_range":0.02,
          "monitor_cam": "GradCAM++",
          "weight_decay": 1e-2,
          "lr":1e-3,
          "batch_size": 5,
          "num_workers": 2,
          "gpus":1,
          "Epoch":100,
          "device":1,
          "train_path":"dataset/Processed/train",
          "val_path":"dataset/Processed/val"


}
    return config

