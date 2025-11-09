from torchvision.utils import save_image

def save_image_grid(tensor, path, nrow=4):
    imgs = (tensor + 1) / 2.0  # map [-1,1] -> [0,1]
    save_image(imgs, path, nrow=nrow)
