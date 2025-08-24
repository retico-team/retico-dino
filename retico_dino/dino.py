"""
DINO Module
==================

This module provides extracts features from DetectedObjectsIU using DINO.
"""

from collections import deque
import numpy as np
import threading
import time
import torch
from transformers import AutoModel
# from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as T
from PIL import Image
import retico_core
from retico_vision.vision import ExtractedObjectsIU, ObjectFeaturesIU

class Dinov2ObjectFeatures(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "DINOv2 Object Features"

    @staticmethod
    def description():
        return "Module for extracting visual features from images."

    @staticmethod
    def input_ius():
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu():
        return ObjectFeaturesIU
    

    def __init__(self, model_name = "dinov2_vits14", top_objects=1, show=False, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.top_objects = top_objects
        self.model = None
        self.feature_extractor = None
        self.show = show
        self.queue = deque(maxlen=1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    # def get_clip_subimage(self, I, img_box):
    #     # expected format:
    #     # Numpy array, length 4, [xmin, ymin, xmax, ymax]

    #     xmin = int(img_box[0])
    #     xmax = int(img_box[2])
    #     ymin = int(img_box[1])
    #     ymax = int(img_box[3])
    #     sub = I.crop([xmin,ymin,xmax,ymax])

    #     if self.show:
    #         import cv2
    #         img_to_show = np.asarray(sub)
    #         cv2.imshow('image',cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) 
    #         cv2.waitKey(1)
    #     # pim = PImage.fromarray(sub)
    #     sub.load()
    #     return sub
    
    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_features = {}

            for i, sub_img in enumerate(detected_objects):
                if i>=self.top_objects: break
                # print(sub_img)
                sub = detected_objects[sub_img]
                if self.show:
                    # print(sub.getbands())
                    # sub = sub.convert("BGR")
                    sub.show()
                # print(type(sub_img), type(detected_objects[sub_img]))
                # sub_img = self.get_clip_subimage(image, obj)
            

                # img = self.preprocess(sub_img).unsqueeze(0).to(self.device)
                # yhat = self.model.encode_image(img).cpu().numpy()
                # object_features[i] = yhat.tolist()
                # inputs = self.feature_extractor(images=sub_img_list, return_tensors="pt")
                # outputs = self.model(**inputs)
                # last_hidden_states = outputs.last_hidden_state
                img_tensor = self.feature_extractor(sub).unsqueeze(0)#.to(self.device)
                feat = self.model(img_tensor).squeeze(0).detach().numpy().tolist()   

                # print(len(feat))
                object_features[i] = feat

            output_iu = self.create_iu(input_iu)
            output_iu.set_object_features(image, object_features)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        # self.model = ViTModel.from_pretrained(self.model_name)
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.feature_extractor = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()
    
    def shutdown(self):
        self._extractor_thread_active = False


class Dinov3ObjectFeatures(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "DINOv3 Object Features"

    @staticmethod
    def description():
        return "Module for extracting visual features from images with DINOv3."

    @staticmethod
    def input_ius():
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu():
        return ObjectFeaturesIU

    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", top_objects=1, show=False, **kwargs):
        super().__init__(**kwargs)

        self.model_name = model_name

        # !!NORMALIZE VALUES ARE SPECIFIC TO MODEL!!
        if self.model_name.endswith('lvd1689m'):
            normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        elif self.model_name.endswith('sat493m'):
            normalize = T.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
        else:
            raise RuntimeWarning(f"Unrecognized model name '{self.model_name}'. Unable to get normalization values.")

        # must resize incoming images BEFORE converting to tensors because of images with possible empty dimensions
        resize = (224, 224)
        self.img_transform = T.Compose([T.Resize(resize, antialias=True), T.ToTensor(), normalize])
        self.model = AutoModel.from_pretrained(model_name)

        self.top_objects = top_objects
        self.show = show
        self.queue = deque(maxlen=1)
        self._extractor_thread_active = False

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)

    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            img = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_features = {}

            for i, sub_img in enumerate(detected_objects):
                if i>=self.top_objects: break

                # pull sub-image
                sub = detected_objects[sub_img]

                # TODO why can sub-images have size dimensions == 0? (e.g. (0, 0), (60, 0), (0, 24), etc.)
                # Displays images without a size dim equal to 0 if show=True
                if self.show and sub.size[0] * sub.size[1] != 0:
                    sub.show()

                # feed sub-image to DINOv3 and collect features
                img_tensor = self.img_transform(sub).unsqueeze(0)
                output = self.model(img_tensor)
                feats = output.pooler_output.squeeze(0).detach().numpy().tolist()

                # save features for all sub-images
                object_features[i] = feats

            output_iu = self.create_iu(input_iu)
            output_iu.set_object_features(img, object_features)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread, daemon=True).start()
