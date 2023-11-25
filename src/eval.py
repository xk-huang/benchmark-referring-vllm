import logging
import os

import hydra
from hydra.utils import instantiate
from datasets import (
    Dataset,
    load_dataset,
    IterableDataset,
    concatenate_datasets,
    interleave_datasets,
)
from omegaconf import DictConfig, OmegaConf
from src.data.transforms import SamCaptionerDataTransform, SCADataTransform
from src.data.collator import SamCaptionerDataCollator, SCADataCollator
from src.arguments import (
    Arguments,
    global_setup,
    SAMCaptionerModelArguments,
    SCAModelBaseArguments,
    SCAModelArguments,
    SCADirectDecodingModelArguments,
    SCAMultitaskModelArguments,
    SCAMultitaskSplitMixerModelArguments,
    ScaMultitaskV2ModelArguments,
    VGDenseCapDataArgument,
    RefCOCODataArgument,
    SA1BCapDataArgument,
    COCOInstanceDataArgument,
    SCADirectDecodingV2ModelArguments,
    SCAMultitaskROIPoolModelArguments,
    ScaTimmMultitaskV2ModelArguments,
)

from transformers import set_seed
import json
import torch
import tqdm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="conf")
def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    train_dataset, eval_dataset = prepare_datasets(args)

    # Initialize our model
    model_type = os.getenv("MODEL_TYPE", None)
    ckpt_path = os.getenv("CKPT_PATH", None)
    if model_type is None:
        raise ValueError(f"MODEL_TYPE must be provided, got {model_type}")
    if ckpt_path is None:
        raise ValueError(f"CKPT_PATH must be provided, got {ckpt_path}")

    if model_type == "gpt4roi":
        from src.model_wrappers import GPT4ROIModelWrapper

        model = GPT4ROIModelWrapper(ckpt_path)

    is_distributed = False
    for eval_dataset_k, eval_dataset_v in eval_dataset.items():
        if os.getenv("RANK", None) is not None and os.getenv("WORLD_SIZE", None) is not None:
            is_distributed = True
            from datasets.distributed import split_dataset_by_node

            eval_dataset_v = split_dataset_by_node(
                eval_dataset_v, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
            )
        eval_dataloader = torch.utils.data.DataLoader(
            DummyDataset(eval_dataset_v),
            batch_size=1,
            shuffle=False,
            num_workers=int(os.getenv("NUM_WORKERS", 0)),
            collate_fn=lambda x: x[0],
        )
        pred_ls = []
        max_samples = int(os.getenv("MAX_SAMPLES", -1))
        for sample_cnt, sample in enumerate(tqdm.tqdm(eval_dataloader)):
            if max_samples > 0 and sample_cnt == max_samples:
                break
            outputs = model.infer_sample(sample, len(pred_ls))
            pred_ls.extend(outputs)

        if is_distributed:
            json_name = f"{int(os.environ['RANK']):05d}.part_json"
            json_dir = os.path.join(args.training.output_dir, eval_dataset_k)
            if not os.path.exists(json_dir):
                os.makedirs(json_dir)
        else:
            json_name = f"{eval_dataset_k}.json"
            json_dir = args.training.output_dir
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(pred_ls, f, indent=4)

        # If all the json is ready
        if is_distributed and os.path.exists(json_dir) and len(os.listdir(json_dir)) == int(os.environ["WORLD_SIZE"]):
            # Merge json
            all_json_ls = []
            for each_json_name in sorted(os.listdir(json_dir)):
                each_json_path = os.path.join(json_dir, each_json_name)
                with open(each_json_path, "r") as f:
                    each_json = json.load(f)
                    all_json_ls.extend(each_json)
            for _cnt in range(len(all_json_ls)):
                all_json_ls[_cnt]["_id"] = _cnt

            all_json_path = os.path.join(args.training.output_dir, f"{eval_dataset_k}.json")
            with open(all_json_path, "w") as f:
                json.dump(all_json_ls, f, indent=4)


class DummyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        image_id = sample["image_id"]

        bbox_ls = []
        gt_caption_ls = []
        region_id_ls = []
        for region in sample["regions"]:
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            xyxy = [x, y, x + w, y + h]
            bbox_ls.append(xyxy)
            gt_caption_ls.append(region["phrases"])
            region_id_ls.append(region["region_id"])

        return dict(
            image=image,
            image_id=image_id,
            bbox_ls=bbox_ls,
            gt_caption_ls=gt_caption_ls,
            region_id_ls=region_id_ls,
        )


def prepare_datasets(args):
    train_data = []
    for train_data_config_name in args.train_data:
        cfg = hydra.compose(
            config_name=f"data/{train_data_config_name}",
            overrides=args.train_data_overrides,
        )
        train_data.append(cfg.data)
    args.train_data = train_data

    # NOTE(xiaoke): We should only inference one eval dataset
    if len(args.eval_data) > 1:
        logger.warning(f"We should only inference one dataset, got {args.eval_data}")
    eval_data = []
    for eval_data_config_name in args.eval_data:
        cfg = hydra.compose(
            config_name=f"data/{eval_data_config_name}",
            overrides=args.eval_data_overrides,
        )
        eval_data.append(cfg.data)

    train_dataset = []
    for i, each_train_data in enumerate(train_data):
        # NOTE: add data `split` to each dataset
        each_train_data.split = "train"

        _train_dataset = instantiate(each_train_data)
        train_dataset.append(_train_dataset)
        logger.info(f"Train Dataset [{i}]: {each_train_data}\n{_train_dataset}")

    eval_dataset = {}
    for i, each_eval_data in enumerate(eval_data):
        # NOTE: add data `split` to each dataset
        # NOTE: visual genome has validation set, but we use test set for evaluation
        if "visual_genome.py" in each_eval_data.path and getattr(each_eval_data, "use_densecap_splits", None) is True:
            logger.info("Using densecap splits in Visual Genome, using test split to eval")
            each_eval_data.split = "test"

        # NOTE: refcoco has validation set, but we use test set for evaluation
        elif "refcoco.py" in each_eval_data.path:
            if each_eval_data.name.startswith("refcoco-") or each_eval_data.name.startswith("refcoco+-"):
                if each_eval_data.split is None or each_eval_data.split == "train":
                    raise ValueError(f"refcoco{{,+}} must have split for eval. got {each_eval_data.split}")
                logger.info(f"Using refcoco{{,+}}: {each_eval_data.split} split to eval")
            elif each_eval_data.name.startswith("refcocog"):
                logger.info("Using refcocog val split to eval")
                each_eval_data.split = "validation"
            elif each_eval_data.name.startswith("refclef"):
                logger.info("Using refclef val split to eval")
                each_eval_data.split = "validation"

        # NOTE: coco has validation set, but it does not have test set.
        elif "coco_instance.py" in each_eval_data.path or "coco_instance-local.py" in each_eval_data.path:
            logger.info("Using coco val split to eval")
            each_eval_data.split = "validation"

        elif "objects365-local.py" in each_eval_data.path:
            logger.info("Using objects365 (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif "v3det-local.py" in each_eval_data.path:
            logger.info("Using v3det (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif "sbu-pseudo_region-local.py" in each_eval_data.path or "sbu-pseudo_region.py" in each_eval_data.path:
            logger.info("Using sbu to eval, but it does not have test split, so we use train split")
            each_eval_data.split = "train"

        elif "coco_caption-pseudo_region.py" in each_eval_data.path:
            logger.info("Using coco_caption (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif (
            "visual_genome-densecap-local.py" in each_eval_data.path
            or "visual_genome-grit-local.py" in each_eval_data.path
        ):
            logger.info(f"Using visual_genome (They are my custom splits for GRiT and Densecap) test split to eval")
            each_eval_data.split = "test"
        else:
            raise ValueError(
                f"Unknown dataset {each_eval_data.path}, we cannot determine the split for it. Please edit `src/train.py:prepare_datasets` to add the split for it."
            )

        _eval_dataset = instantiate(each_eval_data)
        eval_dataset_name = _get_data_name(each_eval_data)
        eval_dataset[eval_dataset_name] = _eval_dataset
        logger.info(f"Eval Dataset [{i}]: {each_eval_data}\n{_eval_dataset}")
    args.eval_data = eval_data  # NOTE: overwrite previous eval_data

    if args.train_data_interleave_probabilities is not None and len(train_dataset) != len(
        args.train_data_interleave_probabilities
    ):
        raise ValueError(
            f"train_data_interleave_probabilities must have the same length as train_data, got {len(train_dataset)} and {len(args.train_data_interleave_probabilities)}"
        )
    # NOTE(xiaoke): Expected a list of Dataset objects or a list of IterableDataset objects.
    if len(train_dataset) > 0:
        if args.train_data_interleave_probabilities is None:
            logger.warning(
                "train_data_interleave_probabilities is not provided, "
                "the resulting dataset will have max_length_datasets*nb_dataset samples. "
                "As we use `all_exhausted` stopping strategy which is a oversampling strategy."
            )
        else:
            if sum(args.train_data_interleave_probabilities) != 1.0:
                logger.info(f"Normalize train_data_interleave_probabilities to sum to 1.0")
                args.train_data_interleave_probabilities = [
                    each_prob / sum(args.train_data_interleave_probabilities)
                    for each_prob in args.train_data_interleave_probabilities
                ]
                logger.info(f"train_data_interleave_probabilities: {args.train_data_interleave_probabilities}")
        # NOTE(xiaoke): Accourding to `datasets/src/datasets/arrow_dataset.py:_interleave_map_style_datasets:6079` and
        # `Breadcrumbsdatasets/src/datasets/iterable_dataset.py:_interleave_iterable_datasets:2293`
        train_dataset = interleave_datasets(
            train_dataset,
            probabilities=args.train_data_interleave_probabilities,
            seed=args.training.seed,
            stopping_strategy="all_exhausted",
        )
    else:
        train_dataset = None

    logger.info(f"Train Dataset: {train_dataset}")
    logger.info(f"Eval Dataset: {eval_dataset}")
    return train_dataset, eval_dataset


def _get_data_name(dataset_config_dict):
    # NOTE: path is the path for data script
    path = dataset_config_dict.path
    path_name = os.path.splitext(os.path.basename(path))[0]
    name = dataset_config_dict.name
    split = dataset_config_dict.split
    return f"{path_name}-{name}-{split}"


if __name__ == "__main__":
    main()
