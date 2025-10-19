# [Neurips 2025 DB] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding
Official dataset release for _PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding_.

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://authoritywang.github.io/partnext/)
[![Project Page](https://img.shields.io/badge/Project_Page-Website-green?logo=homepage&logoColor=white)](https://authoritywang.github.io/partnext/)
[![🤗 Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/AuWang/PartNeXt)

**[Penghao Wang](https://authoritywang.github.io/), Yiyan He, Xin Lv, Yukai Zhou, [Lan Xu](https://www.xu-lan.com/), [Jingyi Yu](http://www.yu-jingyi.com/cv/), [Jiayuan Gu†](https://jiayuan-gu.github.io/)**

**ShanghaiTech University**

**Neurips 2025 Dataset and Benchmark Track**

| [Project Page](https://authoritywang.github.io/partnext/) | [Paper]() | [Dataset](https://huggingface.co/datasets/AuWang/PartNeXt) | [Dataset Toolkit]() | [Benchmark code (Soon)]() | [Annotation code (Soon)]() |<br>

</div>

![Teaser image](assets/teaser.png)

## News

- **2025-10-20**: PartNeXt is on arXiv.

## TODO
- [x] Provide dataset toolkit. 
- [x] Provide example point cloud sampling code. 
- [ ] Release dataset toolkit V2 supporting parse hierarchy semantic. 
- [ ] Release benchmark code and data split. 
- [ ] Release annotation platform code. 
- [ ] Provide the index from PartNeXt to original dataset. 
- [ ] Provide template hierarchy and guildline on using it. 
- [ ] Release filtered data. 

## Download PartNeXt dataset
Our partnext dataset contains two parts:
- 3D meshes in `.glb` format, download from [https://huggingface.co/datasets/AuWang/PartNeXt_mesh](https://huggingface.co/datasets/AuWang/PartNeXt_mesh). 
- 3D part and hierarchy annotations, download from [https://huggingface.co/datasets/AuWang/PartNeXt](https://huggingface.co/datasets/AuWang/PartNeXt). 

You can download the dataset with huggingface-cli:
```bash
hf download --repo-type dataset AuWang/PartNeXt_mesh --local-dir /your/own/path
hf download --repo-type dataset AuWang/PartNeXt --local-dir /your/own/path
```

## PartNeXt Dataset Toolkit
PartNeXt dataset toolkit supports loading meshes and annotations, and get part geometries. This is a early version, we will support parse hierarchy semantic in the next version toolkit (hope to be done around Nov 2025). 

We have upload our dataset toolkit to pypi, so that you can directly install the toolkit by
```
pip install partnext
```

If you want to install the toolkit from source or refer to the code, you can clone the [toolkit repo](https://github.com/AuthorityWang/PartNeXt_lib.git)
```
git clone https://github.com/AuthorityWang/PartNeXt_lib.git
cd PartNeXt_lib
pip install -e .
```

### Toolkit Usage
Pleaes refer to `example/toolkit_example.py`. 

## PartNeXt BenchMark

We propose 2 benchmarks for PartNeXt dataset:
- Class-Agnostic 3D Part Segmentation. 
- Part-Centric 3D Question Answering. 

The data split and evaluation code will be provided soon. Please contact us if you needs to get the code now. 

## PartNeXt Dataset Format

Currently our dataset toolkit is very simple, if you want to parse part semantic hierarchy and get non-leaf node parts, you can refer to the format and try to load the data with your own code. 

### 3D Mesh
We store our 3D meshes in `.glb` format, folder structure follow Objaverse
```
PartNeXt_mesh/
├── glbs
│   ├── 000-000
│   │   ├── 000074a334c541878360457c672b6c2e.glb
│   │   ├── 0000ecca9a234cae994be239f6fec552.glb
│   │   └── ...
│   ├── 000-001
│   ├── ...
```

### Part and hierarchy annotation
We store our in arrow format, all info stores as string. After loading the dataset with huggingface's `datasets` library, the data has follow columns:
- `model_id`: Model uuid, same as glb name. (Object from Objaverse uses the original Objaverse id)
- `type_id`: Subfolder name, follow the objaverse structure.
- `user_id`: Annotator id. 
- `anno_time`: Time consumed for annotation, in seconds.
- `mesh_face_num`: As each object can be consisted of multiple meshes, we store the number of faces for each mesh. 
- `masks`: Leaf node's part masks, corresponding to the ​​finest granularity​​ parts. 
- `hierarchyList`: Hierarchy list, each node can have children and semantics, is a tree structure like PartNet. 

For `mesh_face_num` `masks` `hierarchyList`, we give a further explanation on data structure and show examples. 

<details>
<summary><span style="font-weight: bold;">mesh_face_num</span></summary>

  the key is the index of the mesh in the glb, start from 0

  the value is the number of faces in the mesh

  the order of the index is same as using `dump(concatenate=False)` from `triemsh`

  ```
  {
      "0": 2416,
      "1": 672,
      "2": 2
  }
  ```

</details>

<details>
<summary><span style="font-weight: bold;">masks</span></summary>

  The key is the index of the mask, corresponding to leaf nodes in hierarchyList, start from 0

  The value is a dict, which is the mask

  The key of the mask dict is the index of the mesh in the glb,
  the value is the index of the face in the mesh
  ```
  {
      "0": {
          "0": [0, 1, 2, 3, 4, ...], 
          "1": [221, 222, 223, ...]
      }, 
      "1": {
          "0": [5, 6, 7, 8, 9, ...], 
          "1": [220, 221, 222, ...]
      }, 
      ...
  }
  ```

</details>

<details>
<summary><span style="font-weight: bold;">masks</span></summary>

  The `hierarchyList` is a tree of node, each node is a dict, which has the following keys:
  - `name`: The name of the node, which is the name of the part.
  - `nodeId`: The id of the node, which is the index of the node in the tree.
  - `refNodeId`: The id corresponding to the node in hierarchy template. We will release template soon. 
  - `children`: The children of the node, which is a list of node. (Only non-leaf node has children)
  - `maskId`: The id of the mask of the node, which coresponding to the mask index in the `masks`. (Only leaf node has maskId)
  ```
  [
    {
      "name": "Table",
      "nodeId": 0,
      "refNodeId": 0,
      "children": [
        {
          "name": "Standard Table",
          "nodeId": 1,
          "refNodeId": 1,
          "children": [
            {
              "name": "Tabletop",
              "nodeId": 2,
              "refNodeId": 2,
              "children": [
                {
                  "name": "Surface Panel",
                  "nodeId": 3,
                  "refNodeId": 3,
                  "maskId": 0
                }
              ]
            },
            ...
          ]
        },
        ... 
      ]   
    }
  ]
  ```

</details>

## Usage Examples

We give some examples under `example` folder using our PartNeXt dataset. You can refer to these code to better understand the dataset. 

### Sample part point clouds

[Point-SAM](https://github.com/zyc00/Point-SAM) needs part level point cloud to train promptable 3D part segmentation model. We provide a sample code to sample part point clouds from PartNeXt dataset, and save in Point-SAM's format. 

This example requires raw annotation in json format, which can be downloaded from [https://huggingface.co/datasets/AuWang/PartNeXt_raw](https://huggingface.co/datasets/AuWang/PartNeXt_raw). 

### More example can be provided

You can open a issue to ask for more examples on speicific tasks. 

## Error in annotation

Though we perform a check on the annotation, we admit that PartNeXt still have some error in the annotation. We will filter some error annotations in the future. If you found some error in the annotation, please open a issue or contact us. 

## Acknowledgement
Our PartNeXt dataset is based on [Objaverse](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-Future](https://tianchi.aliyun.com/dataset/98063), thanks for these awesome datasets. If there is any license issue, please contact us and we will remove the data. 

Thanks for Benyuan AI data for data annotation. 

If you find our dataset useful in your research, please consider citing our paper.
```
BibTex Coming Soon
```