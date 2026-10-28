Recent scientific literature highlights lightweight deep learning models like MobileNet variants and YOLO adaptations as top performers for rice disease and pest detection on resource-constrained devices. These prioritize low parameters, FLOPs, offline inference via ONNX/TensorFlow Lite, and mobile compatibility while achieving high accuracy on datasets like Rice Leaf Diseases.[^1][^2][^3]

## Model Selection Criteria

Models were selected from 2016-2026 publications emphasizing CNNs and object detectors for rice-specific tasks. Key metrics include parameters (<10M preferred), FLOPs (<1G for mobiles), accuracy/mAP >90%, and explicit mobile/offline mentions. Excluded heavy models like standard ResNet50 despite good accuracy (92.5%) due to high compute.[^4][^5][^6]

## Benchmark Comparison

| Model                        | Base Architecture       | Accuracy/mAP        | Params (M) | FLOPs (G) | Mobile FPS    | Offline Support |
|:---------------------------- |:----------------------- |:------------------- |:---------- |:--------- |:------------- |:--------------- |
| LiSA-MobileNetV2 [^7]        | MobileNetV2 + Attention | 95.7%               | ~2.0       | ~0.4      | 30+           | Yes (TFLite)    |
| MobileNet-CA-YOLO [^8]       | YOLOv7 + MobileNetV3    | 93.7% mAP           | ~5-7       | ~1.0      | High          | Yes             |
| TLI-YOLO (YOLOv8n) [^9]      | YOLOv8 + Transfer       | ~95%                | 3.2        | Low       | 30            | Yes (Android)   |
| GE-ShuffleNet [^6]           | ShuffleNetV2 + Ghost    | 96.6%               | 5.9        | Low       | High          | Yes             |
| SSD-YOLO [^10]               | YOLO + SSD lightweight  | 98.99% (blast)      | 6.0        | Low       | High          | Yes             |
| EfficientNet-B0 [^11]        | EfficientNet            | 96%+                | 5.3        | 0.39      | 25+           | Yes (TFLite)    |
| MobileNetV3 Large [^4]       | MobileNetV3             | 96.3%               | ~12        | ~0.3      | High          | Yes             |
| Lightweight CNN (2018) [^3]  | Custom two-stage        | High (mobile-tuned) | Small      | Low       | Mobile-suited | Yes             |
| Improved ShuffleNet V2 [^6]  | ShuffleNet              | 94.4%               | Low        | Low       | High          | Yes             |
| Rice-DP-YOLO (YOLOv8n) [^12] | YOLOv8n improved        | 77.8% (15 classes)  | ~3         | Low       | High          | Yes             |

Data aggregated from benchmarks; FPS on entry-level Android (e.g., Snapdragon 6xx).[^10][^8][^9][^6][^7][^12][^4]

## Advantages and Limitations

**Advantages:** These models excel in edge deployment with depthwise convolutions (MobileNet/ShuffleNet) reducing compute by 50-75% vs. VGG/ResNet, enabling 20-30 FPS offline on mobiles without internet. YOLO variants add real-time pest bounding boxes for multi-class detection. Attention (CA/SE) boosts accuracy 2-6% on small datasets.[^3][^8][^7]

**Limitations:** Smaller models trade ~1-5% accuracy for speed; struggle with tiny pests or diverse lighting (mAP drops 5-10%). Limited multi-pest coverage in some (focus on 3-6 diseases); need fine-tuning for local rice varieties.[^9][^13][^10]

## Technical Justification

Selections favor inverted residuals (MobileNetV3/EfficientNet) and ghost modules (GE-ShuffleNet) for parameter efficiency, proven in rice datasets (e.g., 3000+ images). YOLO adaptations suit pests needing localization, with SIoU loss improving small-object mAP. All convert to TFLite for offline mobile use, aligning with entry-level hardware (2-4GB RAM). Top picks: LiSA-MobileNetV2 for classification, MobileNet-CA-YOLO for detection.[^2][^11][^8][^7]
<span style="display:none">[^14][^15][^16][^17][^18][^19][^20]</span>

<div align="center">⁂</div>

[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10088765/

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11253532/

[^3]: https://ar5iv.labs.arxiv.org/html/1812.01043

[^4]: https://d-nb.info/1371393532/34

[^5]: https://arxiv.org/html/2501.06740v1

[^6]: https://www.techscience.com/cmc/v75n2/52135

[^7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12370686/

[^8]: https://ui.adsabs.harvard.edu/abs/2023Agric..13.1285J/abstract

[^9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12031063/

[^10]: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1643096/full

[^11]: https://www.arxiv.org/pdf/2408.01752.pdf

[^12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12528418/

[^13]: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1416940/full

[^14]: https://www.atlantis-press.com/article/126004097.pdf

[^15]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0313387

[^16]: https://arccjournals.com/journal/indian-journal-of-agricultural-research/A-6061

[^17]: https://www.rcs.cic.ipn.mx/2023_152_10/Pests%20Detection%20in%20Agricultural%20Crops%20Using%20Computer%20Vision.pdf

[^18]: https://www.ijsat.org/papers/2025/2/4410.pdf

[^19]: https://repositorio.iscte-iul.pt/bitstream/10071/26147/1/article_90301.pdf

[^20]: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1619365/full
