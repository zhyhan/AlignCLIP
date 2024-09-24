# AlignCLIP: Navigating the Misalignments for Robust Vision-Language Generalization

AlignCLIP aims to address the attention and predictive misalignment problems in vision-language models like CLIP, improving robustness in real-world out-of-distribution (OOD) settings. We propose the following features in AlignCLIP:

- **Attention Alignment Loss (AAL):** Realigns the attention mechanism to prioritize salient foreground entities over background elements.
- **Semantic Label Smoothing (SLS):** Ensures that model predictions better reflect the relationships between classes, reducing misclassification.
- **Efficient Training with $\mathbf{D}$-V Attention:** A diagonal matrix structure significantly improves computational efficiency.
- **State-of-the-art performance on multiple OOD tasks**, including domain shift and open-world generalization.

## Get Started

1. **Install Python 3.8** and other required packages by running:

```bash
pip install -r requirements.txt
```

2. **Prepare Datasets:** AlignCLIP uses datasets built on the [DomainBed](https://github.com/facebookresearch/DomainBed) and [CoOp](https://github.com/KaiyangZhou/CoOp) codebases. Follow the instructions below to prepare the datasets and arrange the folders.

    Clone the repositories and download the datasets following their instructions. Organize the folder structure as follows:

```plain
AlignCLIP/
|-- CoOp/
    |-- data/
        |-- caltech-101/
        |-- eurosat/
        |-- ...  # other CoOp datasets
    |-- ...
|-- DomainBed/
    |-- domainbed/
        |-- data/
            |-- domain_net/
            |-- office_home/
            |-- ...  # other DomainBed datasets
        |-- datasets.py  # the file to update next
        |-- ...
    |-- ...
|-- ...
```

3. **(Important!)** Modify `datasets.py` in `DomainBed/domainbed/datasets.py` (line 192-208) with the following code for correct data preprocessing:

```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```

4. **Train and Evaluate**: You can start training and evaluation by running the following commands:

```bash
python train_alignclip.py --config configs/alignclip_config.yaml
```


## Contact

For questions or usage of the code, please contact [hanzhongyicn@gmail.com](hanzhongyicn@gmail.com).

## Acknowledgement

We appreciate the following repositories for their valuable codebase and datasets:

- https://github.com/facebookresearch/DomainBed
- https://github.com/KaiyangZhou/CoOp
- https://github.com/thuml/Transfer-Learning-Library

