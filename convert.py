import base64
import json
import os
from io import BytesIO

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel


def load_blip_clip():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # BLIPv2 model: we associate a model with its preprocessors to make it easier for inference.
    blip_model, blip_vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
    # CLIP model: get processer for text
    text_version = "openai/clip-vit-large-patch14"
    clip_text_model = CLIPModel.from_pretrained(text_version).cuda().eval()
    clip_text_processor = CLIPProcessor.from_pretrained(text_version)
    return blip_model, blip_vis_processors, clip_text_model, clip_text_processor


def load_clip():
    text_version = "openai/clip-vit-large-patch14"
    clip_text_model = CLIPModel.from_pretrained(text_version).cuda().eval()
    clip_text_processor = CLIPProcessor.from_pretrained(text_version)
    return clip_text_model, clip_text_processor


clip_text_model, clip_text_processor = load_clip()


def preprocess_text(processor, input):
    inputs = processor(text=input, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['pixel_values'] = torch.ones(1, 3, 224, 224).cuda()  # placeholder
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    return inputs


def get_clip_feature_text(model, processor, input):
    inputs = preprocess_text(processor, input)
    outputs = model(**inputs)
    feature = outputs.text_model_output.pooler_output
    return feature


def find_bounding_box(mask):
    # Assuming mask is a 2D tensor
    # Get the indices of non-zero elements
    rows = torch.nonzero(mask, as_tuple=True)[0]
    cols = torch.nonzero(mask, as_tuple=True)[1]

    # If there are no non-zero elements, return an indication that there's no bounding box
    if len(rows) == 0 or len(cols) == 0:
        return None

    # Calculate bounding box coordinates
    min_row, max_row = torch.min(rows), torch.max(rows)
    min_col, max_col = torch.min(cols), torch.max(cols)

    # Return the bounding box as (top-left) and (bottom-right) coordinates
    box = [min_col.item(), min_row.item(), max_col.item(), max_row.item()]
    box_xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    return box_xywh


def mask_2_rle(binary_mask):
    # binary_mask_encoded = mask_pycoco.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # area = mask_pycoco.area(binary_mask_encoded)
    # if area < 16:
    #     return None, None
    rle = mask_util.encode(np.array(binary_mask[..., None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')


def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_anno(mask, caption):
    bbox = find_bounding_box(mask)
    assert bbox is not None

    rle = mask_2_rle(mask.cpu().numpy())

    with torch.no_grad():
        emb = get_clip_feature_text(clip_text_model, clip_text_processor, caption)
        emb = encode_tensor_as_string(emb)

    anno = {
        "id": 0,
        "bbox": bbox,  # xywh
        "mask": rle,
        "category_id": 0,
        "data_id": 0,
        "category_name": "",
        "text_embedding_before": emb,
        "caption": caption,
        "area": int(bbox[2] * bbox[3])
    }
    return anno


def get_output(file_name, data_id, masks, captions):
    image = Image.open(file_name)
    image_base64 = encode_pillow_to_base64(image)

    annos = []
    for i, (mask, caption) in enumerate(zip(masks, captions)):
        anno = get_anno(mask, caption)
        anno['id'] = i
        anno['data_id'] = data_id
        annos.append(anno)

    return {
        "file_name": file_name,
        "data_id": data_id,
        "is_det": False,
        "image": image_base64,
        "dataset_name": "my_dataset",
        "annos": annos
    }


lines = open('train.txt').readlines()
new_lines = []

for i, line in enumerate(lines):
    path, _ = line.strip().split(' ')
    json_path = path.replace('images', 'jsons')
    json_path = os.path.splitext(json_path)[0] + '.json'

    masks = None
    captions = None

    output = get_output(path, i, masks, captions)
    with open(json_path, 'w+') as f:
        json.dump(output, f)

    new_lines.append(json_path)

with open('train_json.txt', 'w+') as f:
    for line in new_lines:
        f.write(line + '\n')
