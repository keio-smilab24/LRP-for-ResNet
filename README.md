# [ECCV24] Layer-Wise Relevance Propagation with Conservation Property for ResNet

- Accepted at ECCV 2024
- [Project page](https://5ei74r0.github.io/lrp-for-resnet.page/)
- [ArXiv](https://arxiv.org/abs/2407.09115)

<i>
The transparent formulation of explanation methods is essential for elucidating the predictions of neural networks, which are typically black-box models. Layer-wise Relevance Propagation (LRP) is a well-established method that transparently traces the flow of a model's prediction backward through its architecture by backpropagating relevance scores. However, the conventional LRP does not fully consider the existence of skip connections, and thus its application to the widely used ResNet architecture has not been thoroughly explored. In this study, we extend LRP to ResNet models by introducing Relevance Splitting at points where the output from a skip connection converges with that from a residual block. Our formulation guarantees the conservation property throughout the process, thereby preserving the integrity of the generated explanations. To evaluate the effectiveness of our approach, we conduct experiments on ImageNet and the Caltech-UCSD Birds-200-2011 dataset. Our method achieves superior performance to that of baseline methods on standard evaluation metrics such as the Insertion-Deletion score while maintaining its conservation property. We will release our code for further research at this https URL
</i>


## Getting started
Clone this repository and get in it. Then run `poetry install --no-root`.

We used the following env.
- Python 3.9.15
- Poetry 1.7.1
- cuda 11.7

See [pyproject.toml](pyproject.toml) to check python dependencies.

### Datasets
Follow the instructions [here](datasets/README.md).

### Get models
If you want to test the method on the CUB, follow the instructions [here](checkpoints/README.md).
You do not have to prepare models for ImageNet.


## Quantitative Experiments
E.g.: Run ours on ImageNet.
```bash
poetry run python visualize.py -c configs/ImageNet_resnet50.json --method "lrp" --heat-quantization --skip-connection-prop-type "flows_skip" --notes "imagenet--type:flows_skip--viz:norm+positive" --all_class --seed 42 --normalize --sign "positive"
```


## Visualize attribution maps for specific images
E.g.: Visualize attribution maps for water ouzel in ImageNet by our method.
```bash
poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" --skip-connection-prop-type "flows_skip" --heat-quantization --image-path ./qual/original/water_ouzel.png --label 20 --save-path ./qual/ours/water_ouzel.png --normalize --sign "positive"
```
See `exp.sh` for more examples


## Bibtex

```
@article{otsuki2024layer,
    title={{Layer-Wise Relevance Propagation with Conservation Property for ResNet}},
    author={Seitaro Otsuki, Tsumugi Iida, F\'elix Doublet, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi, Komei Sugiura},
    journal={arXiv preprint arXiv:2407.09115},
    year={2024},
}
```

## License
This work is licensed under the BSD-3-Clause-Clear license.
To view a copy of this license, see [LICENSE](LICENSE).
