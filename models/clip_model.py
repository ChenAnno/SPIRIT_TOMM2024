import torch
from torch import nn
import clip


class ImageCLIP(nn.Module):
    def __init__(self, clip_model):
        super(ImageCLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, images, mode="eval"):
        self.clip_model.eval()  # 先切换为eval
        if mode == "train":
            # clip.model.convert_weights(self.clip_model)
            return self.clip_model.encode_image(images)
        with torch.no_grad():
            return self.clip_model.encode_image(images)


class TextCLIP(nn.Module):
    def __init__(self, clip_model):
        super(TextCLIP, self).__init__()
        self.clip_model = clip_model

    def forward(self, text):
        # return self.clip_model.encode_text(text)
        self.clip_model.eval()  # 一直为eval
        clip.model.convert_weights(self.clip_model)

        with torch.no_grad():
            return self.clip_model.encode_text(text)

