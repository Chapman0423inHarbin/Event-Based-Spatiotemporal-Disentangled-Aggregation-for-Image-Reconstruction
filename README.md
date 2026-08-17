# ESDAR: Event-Based Spatiotemporal Disentangled Aggregation
## Abstract
The Event-Based Spatial-Temporal Disentangled Aggregation (ESDAR) method reconstructs clear grayscale images for vibration-intensive industrial scenarios like coal mine substations. This framework integrates diffusion-inspired progressive denoising, deformable convolution alignment (DCM) and lightweight DF-TCN modules to disentangle static background and dynamic motion features. Dual complementary branches adopt coordinate attention for feature fusion. The model is optimized with joint MSE, SSIM, LPIPS and adversarial losses, and achieves superior performance compared to FireNet on noisy industrial datasets.

## Running Instructions
1. Prepare image data (including clean and noisy versions) and their corresponding event‑stream data.
2. Use `event_tensor.py` to tensorize the event‑stream data.
3. Train `Static_branch.py` with image data, and export all required data via `Static_branch_test.py`. Note: configure the image size matching your dataset.
4. Train `Event1_branch.py` with event‑tensor data, and export all required data via `Event1_branch_test.py`. Note: configure the image size matching your dataset.
5. Train `Event2_branch.py` using the outputs obtained from Step 4, and export all required data via `Event2_branch_test.py`. Note: configure the image size matching your dataset.
6. Train `fusion_network.py` with all outputs generated from Step 3 to Step 5. Obtain the final reconstruction results using `fusion_network_test.py`.
7. The networks are trained with progressive freezing strategy. Clean images are adopted uniformly as the ground‑truth labels.

## Environment Dependencies

Install command:
```bash
pip install torch torchvision torchmetrics lpips opencv-python numpy pillow
```

## Citation
If you find our work useful for your research, please cite our paper.

DOI:
```bash
10.1155/int/4139129
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
