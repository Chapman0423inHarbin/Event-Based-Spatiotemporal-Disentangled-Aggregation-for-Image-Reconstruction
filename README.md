# ESDAR: Event-Based Spatiotemporal Disentangled Aggregation
## Abstract
The Event-Based Spatial-Temporal Disentangled Aggregation (ESDAR) method reconstructs clear grayscale images for vibration-intensive industrial scenarios like coal mine substations. This framework integrates diffusion-inspired progressive denoising, deformable convolution alignment (DCM) and lightweight DF-TCN modules to disentangle static background and dynamic motion features. Dual complementary branches adopt coordinate attention for feature fusion. The model is optimized with joint MSE, SSIM, LPIPS and adversarial losses, and achieves superior performance compared to FireNet on noisy industrial datasets.

## Environment Dependencies

Install command:
```bash
pip install torch torchvision torchmetrics lpips opencv-python numpy pillow
```
Cite：
```bash
@article{Wang2026EventBasedSD,
  title={Event-Based Spatiotemporal Disentangled Aggregation for Image Reconstruction},
  author={Yanwei Wang and Chubin Peng and Zhenhui Min and Qingju Tang},
  journal={Int. J. Intell. Syst.},
  year={2026},
  volume={2026},
  url={https://api.semanticscholar.org/CorpusID:288029299}
}
```
