# PaintsTorch2

## Roadmap

**Dataset**:

- [x] Modular Dataset Pipelines
- [ ] Dataset Pipelines
    - Color:
        - [x] Quantization
        - [x] kMeans Colors
        - [ ] Semantic Mean Color Segmentation
    - Hints:
        - [x] Random Pixel Activations
        - [ ] Random Circles
        - [ ] Random Strokes
    - Linear:
        - [X] xDoG
        - [ ] Canny Edge Filter
    - Mask:
        - [x] Patch Masking
        - [x] kMeans Masking
        - [ ] Semantic Segmentation

**Model**:

- [x] Generator
    - [x] ResNetXtBottleneck
    - [x] ModularConv2D
    - [x] ToRGB
    - [x] UpsampleBlock
- [x] Discriminator
    - [x] ResNetXtBottleneck
- [ ] Training Pipeline
    - [x] Gradient Penalty
    - [x] Classic cWGAN-GP Training Pipeline
    - [x] Spectral Normalization
    - [x] Use DataParallel
    - [x] Use `torch.cuda.amp` for Autamatic Mixed Precision
    - [x] Step Learning Rate Decay

**Metrics**

- [x] FID Score
- [ ] MOS Score

## Authors

- Yliess HATI - [Github](https://github.com/yliess86)
- Vincent THEVENIN - [Github](https://github.com/vincent-thevenin)
- Grégor JOUET - [Github](https://github.com/WIN32GG)

## Bibliography

- [GAN Improvements](https://arxiv.org/pdf/1710.10196.pdf) - Progressive Growing of GANS for Improved Quality, Stability, and Variation - Karras and al. - *ICLR 2018*
- [SN-GAN](https://arxiv.org/pdf/1802.05957.pdf) - Spectral Normalization for Generative Adervsarial Networks - Miyato and al - *ICLR 2018*
- [AlacGAN](https://arxiv.org/pdf/1808.03240.pdf) - User-Guided Deep Anime Line Art Colorization with Conditional Adversarial Networks - Ci and al. - *ACM MM 2018* - [Code](https://github.com/orashi/AlacGAN)
- [PaintsTorch](https://dl.acm.org/doi/abs/10.1145/3359998.3369401) - PaintsTorch: a User-Guided Anime Line Art Colorization Tool with Double Generator Conditional Adversarial Network - Hati, Jouet and al. - *CVMP 2019* - [Code](https://github.com/yliess86/PaintsTorch)
- [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) - Analyzing and Improving the Image Quality of StyleGAN - Karras and al. - *CVPR 2020* - [Code](https://github.com/lucidrains/stylegan2-pytorch)