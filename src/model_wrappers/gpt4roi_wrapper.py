import argparse
import copy
import os
import uuid
from ftplib import error_proto
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.sys.path.insert(0, "./")
from PIL import Image

DEBUG = False

if not DEBUG:
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

    from gpt4roi.train.train import preprocess, preprocess_multimodal
    from llava.model.utils import KeywordsStoppingCriteria
    from llava.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
multimodal_cfg = {
    "is_multimodal": True,
    "sep_image_conv_front": False,
    "image_token_len": 256,
    "image_aspect_ratio": "square",
    "use_im_start_end": True,
}

os.makedirs("image", exist_ok=True)

import re


def count_num_bboxes(text):
    pattern = r"<region\d+>"
    matches = re.findall(pattern, text)
    return len(matches), matches


class GPT4ROIModelWrapper:
    def __init__(self, model_name="/home/shilong/Desktop/xgpt/heavy_roi_checkpoints/debug"):
        print("Have fun, cheems!")
        if DEBUG:
            print("Debug mode ....")
            self.first_round = True
            self.chat_history = []

        else:
            if not os.path.exists(model_name):
                raise ValueError(f"model path {model_name} does not exist")
            self.build_model(model_name)

    def build_model(self, model_name):
        ########################  base model define ########################
        print("Start loading model...")
        disable_torch_init()
        model_name = os.path.expanduser(model_name)
        # NOTE: dependency hell. Check the issue of open llama, do not use use_fast=True. Otherwise it stuck.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        from gpt4roi.models.spi_llava import SPILlavaMPTForCausalLM

        # TODO add detector for normal conversation
        self.model = SPILlavaMPTForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model.config.mm_vision_tower, torch_dtype=torch.float16
        )

        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        spi_tokens = ["<bbox>", "<point>"]
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)

        vision_tower = self.model.get_model().vision_tower[0]

        if vision_tower.device.type == "meta":
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).cuda()
            self.model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device="cuda", dtype=torch.float16)

        vision_tower.to(device="cuda", dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end

        vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )

        # init inputs: img, inputs ids, texts
        self.last_source = dict()

    def init_image(self, image):
        width, height = image.size
        image = Image.fromarray(np.array(image)).convert("RGB")

        image = self.image_processor.preprocess(image, do_center_crop=False, return_tensors="pt")["pixel_values"][0]

        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)

        return dict(
            image=image,
            cur_token_len=cur_token_len,
            width=width,
            height=height,
        )

    def init_boxes(self, question_str, bboxes, width, height, cur_token_len):
        if len(bboxes) == 0:
            no_spi_this_round = True
        else:
            no_spi_this_round = False
        if not no_spi_this_round:
            ori_bboxes = np.array(bboxes, dtype=np.float64)
            norm_bboxes = ori_bboxes / np.array([width, height, width, height])

        # TODO: resize bounding boxes

        history_cache = []
        begin_str = """The <image> provides an overview of the picture.\n"""

        init_question = begin_str + question_str

        _, bbox_names = count_num_bboxes(init_question)

        if not no_spi_this_round:
            init_question = re.sub(r"<region(\d+)>", r"region\g<1> <bbox>", init_question)
            init_question = re.sub(r"\<(\d+)\>", r"region\g<1> <bbox>", init_question)
            init_question = init_question.replace("<>", "<bbox>")
            # init_question = init_question.replace("<region>", "<bbox>"*len(norm_bboxes))

        sources = dict()
        sources["conversations"] = []
        sources["conversations"].append({"from": "human", "value": init_question})
        history_cache.append({"sources": copy.deepcopy(sources)})
        history_cache[-1]["region_name_set"] = set(bbox_names)

        print(sources["conversations"])
        sources = preprocess_multimodal([sources["conversations"]], multimodal_cfg, cur_token_len)
        ori_source = copy.deepcopy(sources)

        data_dict = preprocess(sources, self.tokenizer)

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            sources=ori_source,
        )
        if not no_spi_this_round:
            data_dict["bboxes"] = torch.Tensor(norm_bboxes)
        else:
            data_dict["bboxes"] = history_cache[-1]["bboxes"]
        history_cache[-1]["bboxes"] = copy.deepcopy(data_dict["bboxes"])
        return data_dict

    def init_inputs(self, input_dict, question_str, history_cache):
        bboxes = input_dict["boxes"]
        image = input_dict["image"]

        if len(bboxes) == 0:
            no_spi_this_round = True
        else:
            no_spi_this_round = False
        width, height = image.size
        if not no_spi_this_round:
            ori_bboxes = np.array(bboxes, dtype=np.float64)
            norm_bboxes = ori_bboxes / np.array([width, height, width, height])

        image = Image.fromarray(np.array(image)).convert("RGB")

        image = self.image_processor.preprocess(image, do_center_crop=False, return_tensors="pt")["pixel_values"][0]

        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

        # TODO: resize bounding boxes

        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size
        if len(history_cache) == 0:
            begin_str = """The <image> provides an overview of the picture.\n"""

            init_question = begin_str + question_str

            _, bbox_names = count_num_bboxes(init_question)

            if not no_spi_this_round:
                init_question = re.sub(r"<region(\d+)>", r"region\g<1> <bbox>", init_question)
                init_question = re.sub(r"\<(\d+)\>", r"region\g<1> <bbox>", init_question)
                init_question = init_question.replace("<>", "<bbox>")
                # init_question = init_question.replace("<region>", "<bbox>"*len(norm_bboxes))

            sources = dict()
            sources["conversations"] = []
            sources["conversations"].append({"from": "human", "value": init_question})
            history_cache.append({"sources": copy.deepcopy(sources)})
            history_cache[-1]["region_name_set"] = set(bbox_names)

        else:
            sources = history_cache[-1]["sources"]
            question_str = re.sub(r"<region(\d+)>", r"region\g<1> <bbox>", question_str)
            question_str = re.sub(r"\<(\d+)\>", r"region\g<1> <bbox>", question_str)
            question_str = question_str.replace("<>", "<bbox>")

            sources["conversations"].append({"from": "human", "value": question_str})
        print(sources["conversations"])
        sources = preprocess_multimodal([sources["conversations"]], multimodal_cfg, cur_token_len)
        ori_source = copy.deepcopy(sources)

        data_dict = preprocess(sources, self.tokenizer)

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            sources=ori_source,
        )
        data_dict["image"] = image
        if not no_spi_this_round:
            data_dict["bboxes"] = torch.Tensor(norm_bboxes)
        else:
            data_dict["bboxes"] = history_cache[-1]["bboxes"]
        history_cache[-1]["bboxes"] = copy.deepcopy(data_dict["bboxes"])
        return data_dict, history_cache

    def check_input(self, text, image, history_cache):
        if image is None:
            # no image in input
            return (
                "GPT4RoI is a Vision and Language model. Please should upload an image first. Please click Clear All and try again.",
                text,
            )

        if len(history_cache) == 0:
            first_round = True
            if len(image["boxes"]) == 0:
                return (
                    "Please provide your region of interest by drawing boxes on your uploaded image. Please click Clear All and try again.",
                    text,
                )
        else:
            first_round = False

        ##### check num_bboxes in image and text  #####
        if first_round:
            num_draw_bbox_this_round = len(image["boxes"])
        else:
            if len(image["boxes"]) == 0:
                num_draw_bbox_this_round = 0
            else:
                num_draw_bbox_this_round = len(image["boxes"]) - len(history_cache[-1]["bboxes"])

        ### fix <regionx> for old reference
        if not first_round:
            region_name_set = copy.deepcopy(history_cache[-1]["region_name_set"])
            num_bboxes_in_text, region_names = count_num_bboxes(text)
            for region_name in region_names:
                if region_name in region_name_set:
                    text = text.replace(region_name, region_name[1:-1])
                else:
                    region_name_set.add(region_name)
            history_cache[-1]["region_name_set"] = region_name_set

        num_bboxes_in_text, region_names = count_num_bboxes(text)
        if num_bboxes_in_text != num_draw_bbox_this_round:
            if num_bboxes_in_text == 0:
                return (
                    f"""🐛🐛🐛: Your question: {text} doesn't have correct reference(in <regionx> format) to your drawing boxes.
                            Please refer to User Manual 1 for more details. Click `Clear All` and try again.
                        """,
                    text,
                )
            else:
                return (
                    f"""🐛🐛🐛 In Your question: `{text}`
                        the number of <regionx> is {num_bboxes_in_text}, which does not match the number of bounding box in the image,  {num_draw_bbox_this_round}.
                        Please refer to User Manual 1 for more details. Click `Clear All` and try again.
                        """,
                    text,
                )
        return None, text

    # TODO fix the refernece <>

    def run(self, text, image):
        print("GPT4RoI starting")
        # error_string, text = self.check_input(text, image, history_cache)
        # if error_string is not None:
        #     chat_history.append(("Error: {}".format(error_string.replace("<", "&lt;").replace(">", "&gt;")), ""))
        #     return None, chat_history, state, history_cache

        # state = self.visualize(image, state)
        # show_img = state[-1]["img"]
        # new_path = state[-1]["path"]
        # if len(image["boxes"]):
        #     chat_history.append(((new_path,), None))
        text = text.strip()
        if text is None or len(text) == 0:
            print("Warning: empty text, using hello, world!")
            text = "hello, world!"
        print(text)
        this_round_input = copy.deepcopy(text)

        # if DEBUG:
        #     outputs = "Output copy from: {}".format(text)
        #     self.first_round = False
        #     return show_img, [((new_path,), outputs)], state

        # history_cache = []
        # init_inputs, history_cache = self.init_inputs(image, text, history_cache)
        # bboxes = init_inputs["bboxes"]

        init_image_dict = self.init_image(image["image"])
        init_inputs = self.init_boxes(
            text, image["boxes"], init_image_dict["width"], init_image_dict["height"], init_image_dict["cur_token_len"]
        )
        init_inputs["image"] = init_image_dict["image"]
        bboxes = init_inputs["bboxes"]

        if bboxes is not None:
            bboxes = [bboxes.cuda().half()]
        else:
            # raise NotImplementedError("Pure text inference is not implemeted")
            bboxes = None

        image = init_inputs["image"]
        input_ids = init_inputs["input_ids"].cuda()[None]
        stop_str = "###"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # TODO: why modify model at forward process? necessary or not?
        self.model.model.tokenizer = self.tokenizer

        with torch.inference_mode():
            self.model.orig_forward = self.model.forward
            self.model.forward = partial(self.model.orig_forward, img_metas=[None], bboxes=bboxes)

            with torch.amp.autocast(device_type="cuda"):
                output_ids = self.model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    # NOTE: struggle to make it work... -_-|, check docs/ENV.md
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria],
                )
            self.model.forward = self.model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        if not len(outputs):
            # FIXME
            outputs = "There is internal error. Please click 'Clear All' and try again."

        # init_outputs = outputs
        print(f"this_round_input: {this_round_input}")
        print(f"outputs: {outputs}")
        outputs = outputs.replace("Assistant: ", "").replace("Assistant:", "")
        # history_cache[-1]["sources"]["conversations"].append({"from": "gpt", "value": outputs})
        # chat_history.append(("Question: {}".format(init_text.replace("<", "&lt;").replace(">", "&gt;")), init_outputs))
        return outputs

    def infer_sample(self, sample, region_cnt_base):
        text = "Can you give a description of the region mentioned by <region1>?"
        text = text.strip()
        raw_image = sample.get("image")
        bbox_ls = sample.get("bbox_ls")
        gt_caption_ls = sample.get("gt_caption_ls")
        image_id = sample.get("image_id")
        region_id_ls = sample.get("region_id_ls")

        init_image_dict = self.init_image(raw_image)

        ret_list = []

        for region_cnt in range(len(bbox_ls)):
            raw_boxes = bbox_ls[region_cnt : region_cnt + 1]
            init_inputs = self.init_boxes(
                text, raw_boxes, init_image_dict["width"], init_image_dict["height"], init_image_dict["cur_token_len"]
            )
            init_inputs["image"] = init_image_dict["image"]
            bboxes = init_inputs["bboxes"]

            if bboxes is not None:
                bboxes = [bboxes.cuda().half()]
            else:
                # raise NotImplementedError("Pure text inference is not implemeted")
                bboxes = None

            image = init_inputs["image"]
            input_ids = init_inputs["input_ids"].cuda()[None]
            stop_str = "###"
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            # TODO: why modify model at forward process? necessary or not?
            self.model.model.tokenizer = self.tokenizer

            with torch.inference_mode():
                self.model.orig_forward = self.model.forward
                self.model.forward = partial(self.model.orig_forward, img_metas=[None], bboxes=bboxes)

                with torch.amp.autocast(device_type="cuda"):
                    output_ids = self.model.generate(
                        input_ids,
                        images=image.unsqueeze(0).half().cuda(),
                        # NOTE: struggle to make it work... -_-|, check docs/ENV.md
                        do_sample=True,
                        temperature=0.2,
                        max_new_tokens=1024,
                        stopping_criteria=[stopping_criteria],
                    )
                self.model.forward = self.model.orig_forward

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")

            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs.replace("Assistant: ", "").replace("Assistant:", "")

            ret_list.append(
                dict(
                    _id=region_cnt_base + region_cnt,
                    split="inference",
                    references=gt_caption_ls[region_cnt],
                    candidate=[outputs],
                    metadata=dict(
                        image_id=image_id,
                        bbox=raw_boxes[0],
                        region_id=region_id_ls[region_cnt],
                    ),
                )
            )
        return ret_list

    def visualize(self, image, state):
        """
        1. Non-first round and without new boxes, the image['boxes'] is empty []
        2. Non-first round but with new drawing box, the image['boxes'] contains all previous boxes
        """
        if len(image["boxes"]) == 0:
            # didn't draw boxe  # TODO: add full boxes for the first round if not boxes provided
            return state

        img = np.array(image["image"])
        img = Image.fromarray(img)
        boxes = image["boxes"]

        new_image = np.array(image["image"], dtype=np.uint8).copy()
        color = (255, 0, 0)
        thickness = 2  # Line thickness of 2 px
        text = "<region_1>"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        for bbox_id, box in enumerate(boxes):
            start_point = int(box[0]), int(box[1])
            end_point = int(box[2]), int(box[3])
            new_image = cv2.rectangle(new_image, start_point, end_point, color, thickness)
            new_image = cv2.putText(
                new_image,
                f"<{bbox_id + 1}>",
                (int(box[0]), int(box[1]) + text_size[1]),
                font,
                font_scale,
                (255, 0, 0),
                thickness,
            )

        new_image = Image.fromarray(new_image).convert("RGB")
        new_path = "image/{}.png".format(uuid.uuid4().hex)
        # new_image.save(new_path)
        state.append(dict(img=img, path=new_path))
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./GPT4RoI-7B")
    args = parser.parse_args()
    if not os.path.exists(args.model):
        raise ValueError(f"model path {args.model} does not exist")

    model_wrapper = GPT4ROIModelWrapper(args.model)

    import requests

    img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw)
    # image = dict(image=raw_image, boxes=[[120, 270, 1720, 830]])
    image = dict(image=raw_image, boxes=[[0, 0, 100, 100]])
    question = "Can you give a description of the region mentioned by <region1>?"

    outputs = model_wrapper.run(question, image)
