import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, '/workspace/module/image2text/blip')
from models.blip import blip_decoder

class BLIP:
    def __init__(self):
        self.image_size = 384
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def generate_caption_from_path(self, image_path):
        text = ""
        raw_image = Image.open(image_path).convert('RGB')

        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def generate_caption_from_image(self, raw_image):
        text = ""
        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)