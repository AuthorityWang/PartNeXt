import os
import json
import argparse
import numpy as np
from datasets import Dataset, DatasetDict
from sample_from_glb import glb2pcsam_idx_full

class PartNeXtDataGenerator:
    def __init__(
            self,
            annotation_folder,
            glb_folder,
            num_points=32768,
            seed=42
        ):
        self.annotation_folder = annotation_folder
        self.glb_folder = glb_folder
        self.num_points = num_points
        self.seed = seed

    def annotation2pcd(self, glb_id, type_id):
        annotation_path = os.path.join(self.annotation_folder, type_id, glb_id, f"{glb_id}.json")
        glb_path = os.path.join(self.glb_folder, type_id, f"{glb_id}.glb")
        if not os.path.exists(annotation_path) or not os.path.exists(glb_path):
            return None
        try:
            result = glb2pcsam_idx_full(
                glb_path,
                annotation_path,
                num_samples=self.num_points,
                seed=self.seed,
            )
            mask = result['masks'].astype(np.bool_)
            # if the shape is not correct, return None
            if result['obj_points'].shape != (32768, 3):
                print(f"Error: {glb_id} {type_id} obj_points shape is not correct: {result['obj_points'].shape}")
                return None
            if result['obj_colors'].shape != (32768, 3):
                print(f"Error: {glb_id} {type_id} obj_colors shape is not correct: {result['obj_colors'].shape}")
                return None
            if mask.shape[1] != 32768:
                print(f"Error: {glb_id} {type_id} mask shape is not correct: {mask.shape}")
                return None
            if mask.ndim != 2:
                print(f"Error: {glb_id} {type_id} mask ndim is not correct: {mask.ndim}")
                return None
        except Exception as e:
            print(e)
            return None
        return {
            "xyz": result['obj_points'].astype(np.float32),
            "rgb": result['obj_colors'].astype(np.uint8),
            "mask": mask
        }

    def __call__(self, object_type_list):
        for data_info in object_type_list:
            result = self.annotation2pcd(data_info["glb_id"], data_info["type_id"])
            if result is not None:
                yield result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_folder", type=str, default="/data/new_disk5/wangph1/3daffordance/data")
    parser.add_argument("--annotation_folder", type=str, default="/data/new_disk5/wangph1/3daffordance/annotation/production")
    parser.add_argument("--cache_dir", type=str, default="/data/new_disk5/wangph1/FunPart3D/cache")
    parser.add_argument("--index_dir", type=str, default="/data/new_disk5/wangph1/FunPart3D/DataProcessing-PCSAM/index")
    parser.add_argument("--train_index", type=str, default="data_index_20250509_023305_train.json")
    parser.add_argument("--test_index", type=str, default="data_index_20250509_023305_test.json")
    parser.add_argument("--output", type=str, default="/data/new_disk5/wangph1/FunPart3D/datasets/part3d")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output) if not os.path.exists(args.output) else None

    generator = PartNeXtDataGenerator(
        annotation_folder=args.annotation_folder,
        glb_folder=args.glb_folder,
        num_points=32768,
        seed=42
    )

    # data_list = json.load(open(args.index_file, "r"))
    train_index_path = os.path.join(args.index_dir, args.train_index)
    test_index_path = os.path.join(args.index_dir, args.test_index)
    train_data_list = json.load(open(train_index_path, "r"))
    test_data_list = json.load(open(test_index_path, "r"))

    train_dataset = Dataset.from_generator(
        generator, 
        cache_dir=os.path.join(args.cache_dir, "train"), 
        gen_kwargs=dict(object_type_list=train_data_list), 
        # features=features,
        num_proc=args.num_proc
    )
    test_dataset = Dataset.from_generator(
        generator, 
        cache_dir=os.path.join(args.cache_dir, "test"), 
        gen_kwargs=dict(object_type_list=test_data_list), 
        # features=features,
        num_proc=args.num_proc
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    dataset_dict.save_to_disk(args.output)