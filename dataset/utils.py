from pathlib import Path
import os
import numpy as np 

# creat function 
lenght_img = 19
def image_to_label(sample: int ,path_img: str, path_lab: str):
    if sample <= lenght_img:
        image_index = list(os.listdir(path_img))
        label_index = list(filter(("._la_029.nii.gz").__ne__,os.listdir(path_lab)))
        image_index.remove("._la_029.nii.gz")
        label_index.remove('._la_014.nii.gz')
        list_img = image_index[sample]
        list_lab = label_index[sample]
        for (img_idx , lab_idx) in zip(list_img,list_lab):
            if img_idx == lab_idx:  
                path_img = os.path.join(root_img,list_img)
                path_lab = os.path.join(root_lab,list_lab)
            else: 
                print("the path doesn't exist")

        return Path(path_img) , Path(path_lab)  
    else:
        raise Exception("out of range index list images")

def Normalization(full_volume):
    mean = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mean ) / std
    return normalized

def Standrization(normalized_img):
    Standrization = (normalized_img - normalized_img.min()) / (normalized_img.max() - normalized_img.min())
    return Standrization

def image_to_label_file(sample: int ,path_img: str, path_lab: str):
    if sample <= 120:
        image_index = list(os.listdir(path_img))[sample]
        label_index = list(os.listdir(path_lab))[sample]
        for (img_idx , lab_idx) in zip(image_index,label_index):
            if img_idx == lab_idx:  
                path_img = os.path.join(path_img,image_index)
                path_lab = os.path.join(path_lab,label_index)
                break
            else: 
                print("the path doesn't exist")

        return Path(path_img) , Path(path_lab)  
    else:
        raise Exception("out of range index list images")

def image_to_mask(idx_path:int ,idx_sample:int, path_train: str):
    list_root = os.listdir(train_path)
    data_mask_path = []
    file_idx = list_root[idx_path]      
    for (root,dire,files) in os.walk(train_path/str(file_idx)):
        for folder in range(len(dire)):
            path_data = os.path.join(root,dire[folder])
            data_mask_path.append(path_data)
    data_path, mask_path = data_mask_path[0] , data_mask_path[1]
    image_path , label_path= image_to_label_file(idx_sample,data_path,mask_path)
    return image_path , label_path 




class AttentionFilter(nn.Module):
    def __init__(self, filter_in , filter_out , filter_dim):
        super(AttentionFilter, self).__init__()
        self.Wieghts_signal = GlobalFilter(filter_in , filter_dim)
        self.Gate_signal = GlobalFilter(filter_out . filter_in)
        self.NormLayar = nn.NormLayar(filter_in)
        self.act = nn.Softmax(dim=(-1,-1))
    def forward(self . x , g ):
        x = self.Gate_signal(x)
        x = self.NormLayar(x)
        g = self.Wieghts_signal(g)
        g = self.NormLayar(g)
        out = torch.bmm(x,g,axis=-1)
        scaled_out = self.Softmax(out / torch.sqrt(2*Pi*out.var(-1 , unbiased=False , keepdims=True)))
        return scaled_out
