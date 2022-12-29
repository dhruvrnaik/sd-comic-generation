from torch.utils.data import Dataset
import os
import json
from torchvision import transforms
from PIL import Image


class COCOCropDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.tokenizer = tokenizer

        annotation_file = "annotations/captions_train2014.json"
        annot_json = json.load(open(data_root + "/" + annotation_file))

        self.captions = []
        self.img_paths = []
        for i in range(len(annot_json['annotations'])):
            annotation = annot_json['annotations'][i]
            self.captions.append(annotation['caption'])
            self.img_paths.append(data_root + f"/train2014/COCO_train2014_{str(annotation['image_id']).zfill(12)}.jpg")


        self._length = len(self.img_paths)

        print(self._length)


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(int(1*size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(int(1*size)),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        instance_image = Image.open(self.img_paths[index])
        instance_prompt = self.captions[index]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["bias_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example