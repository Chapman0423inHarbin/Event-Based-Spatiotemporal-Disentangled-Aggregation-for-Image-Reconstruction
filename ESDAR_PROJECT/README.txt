Abstract：
The Event-Based Spatial-Temporal Decoupling Aggregation (ESDAR) method can reconstruct clear grayscale images in industrial scenarios with intense vibrations, such as coal mine substations. This approach combines diffusion-inspired gradual denoising, deformable convolution alignment (DCM), and lightweight DF-TCN, effectively separating static backgrounds from dynamic motion features. The dual branches complement information through feature fusion based on coordinate attention. The model is trained using a joint MSE/SSIM/LPIPS loss and adversarial loss, and outperforms FireNet on noisy industrial datasets.

Environment Dependencies：
python >= 3.10
pytorch >= 2.9.0
torchvision
torchmetrics
lpips
opencv‑python
numpy
pillow

Install command:
pip install torch torchvision torchmetrics lpips opencv-python numpy pillow

We build a mixed dataset containing synthetic event‑image pairs and real‑world vibration‑scene data captured from coal‑mine substation environments.
Synthetic subset: 1000 samples with simulated event streams and degraded grayscale inputs.
Real‑world subset: 1670 real frames collected under mechanical vibration and low‑illumination conditions.

If this work helps your research, please cite:

Bib Tex：
@article{Wang2026EventBasedSD,
  title={Event-Based Spatiotemporal Disentangled Aggregation for Image Reconstruction},
  author={Yanwei Wang and Chubin Peng and Zhenhui Min and Qingju Tang},
  journal={Int. J. Intell. Syst.},
  year={2026},
  volume={2026},
  url={https://api.semanticscholar.org/CorpusID:288029299}
}

MLA:
Wang, Yanwei et al. “Event-Based Spatiotemporal Disentangled Aggregation for Image Reconstruction.” Int. J. Intell. Syst. 2026 (2026): n. pag.

APA:
Wang, Y., Peng, C., Min, Z., & Tang, Q. (2026). Event-Based Spatiotemporal Disentangled Aggregation for Image Reconstruction. Int. J. Intell. Syst., 2026.

Chicago：
Wang, Yanwei, Chubin Peng, Zhenhui Min and Qingju Tang. “Event-Based Spatiotemporal Disentangled Aggregation for Image Reconstruction.” Int. J. Intell. Syst. 2026 (2026): n. pag.