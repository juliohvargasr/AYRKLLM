### **Architectural Decision Record (ADR)**

**Title**: Adopt a Hybrid Tech Stack for Cross-Platform AI Deployment in Resource-Constrained Environments  
**Status**: Approved  
**Context**:  
The project requires deploying AI models (e.g., computer vision for agriculture/health) on **mobile devices** and **augmented reality (AR) glasses** with strict constraints:  

- **Offline operation** (no cloud dependency).  
- **Computational efficiency** (support for entry-level hardware: ≤2 GB RAM, limited GPU/CPU).  
- **Cross-platform compatibility** (iOS, Android, AR glasses like HoloLens/Magic Leap).  
- **Real-time performance** for tasks like object detection in agriculture or AR overlays.  

---

### **Decision**

Adopt a **hybrid technology stack** combining:  

1. **Python/PyTorch** for model development and training.  
2. **ONNX/TensorFlow Lite** for model optimization and deployment.  
3. **Flutter** for cross-platform mobile apps.  
4. **Unity + TensorFlow Lite/ONNX** for AR glasses integration.  
5. **Platform-specific SDKs** (e.g., HoloLens/Magic Leap) for AR hardware.  

---

### **Drivers**

1. **Efficiency & Portability**:  
   - Lightweight models (e.g., YOLOv8nano, EfficientNetB0) require frameworks like **ONNX Runtime** and **TensorFlow Lite** for quantization and edge optimization.  
2. **Cross-Platform Support**:  
   - **Flutter** enables code reuse for iOS/Android, reducing development effort.  
   - **Unity** is industry-standard for AR/3D visualization.  
3. **Offline & Real-Time Needs**:  
   - ONNX/TFLite enable on-device inference without cloud connectivity.  
4. **Research-Friendly Development**:  
   - **PyTorch** offers flexibility for custom model architectures and training.  

---

### **Alternatives Considered**

1. **Native-Only Approach**:  
   - Use **Core ML** (iOS) and **MLKit** (Android) for mobile, **C#** for AR glasses.  
   - *Rejected*: Limits cross-platform code reuse and complicates model optimization.  
2. **Single Framework for All**:  
   - Rely solely on **TensorFlow Lite** or **ONNX** for all deployment.  
   - *Rejected*: Lacks native AR/3D integration (Unity/ARKit) and poor mobile UX compared to Flutter.  
3. **No AR Support**:  
   - Focus only on mobile.  
   - *Rejected*: Excludes critical use cases (e.g., AR overlays for pest detection in crops).  
4. **Heavy Models with Cloud Offload**:  
   - Use larger models with cloud inference.  
   - *Rejected*: Violates offline requirement and introduces latency.  

---

### **Rationale**

| **Choice**                 | **Pros**                                                                              | **Cons**                                                              |
| -------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **PyTorch + ONNX/TFLite**  | - PyTorch’s flexibility for research; ONNX/TFLite enable cross-platform optimization. | - Learning curve for PyTorch; model conversion adds complexity.       |
| **Flutter**                | - Single-codebase for iOS/Android; rich widget ecosystem.                             | - Performance overhead vs. native Swift/Kotlin (mitigated by TFLite). |
| **Unity + AR SDKs**        | - Robust 3D/AR tooling; integrates with TFLite/ONNX for inference.                    | - Steeper learning curve for Unity; larger app size.                  |
| **Platform-Specific SDKs** | Direct hardware access for AR glasses (e.g., HoloLens tracking).                      | - Fragmented codebase; increases maintenance effort.                  |

**Final Justification**:  
The hybrid stack balances **efficiency**, **cross-platform reach**, and **performance**. PyTorch accelerates model development, while ONNX/TFLite ensure lightweight deployment. Flutter/Unity unify mobile/AR experiences, and platform-specific SDKs address hardware constraints. This approach meets all technical criteria while minimizing lock-in.

---

### **Implications**

1. **Complexity**: Managing multiple frameworks (PyTorch, Flutter, Unity) increases initial development effort.  
2. **Performance Tuning**: Requires iterative profiling (e.g., TensorFlow Profiler, AR emulator testing).  
3. **Future Compatibility**: AR glasses platforms (e.g., HoloLens 2 vs. Magic Leap 2) may require SDK updates.  

---

### **References**

1. **Model Optimization**:  
   - [TensorFlow Lite Model Optimization Guide](https://www.tensorflow.org/lite/performance)  
   - [ONNX Runtime Performance Best Practices](https://github.com/microsoft/onnxruntime/tree/main/docs/performance_optimization)  
2. **Cross-Platform Development**:  
   - [Flutter Mobile SDK](https://flutter.dev/)  
   - [Unity TensorFlow Lite Plugin](https://github.com/Pierpaolo28/Unity-TensorFlow-Lite)  
3. **AR Glasses Integration**:  
   - [HoloLens SDK](https://developer.microsoft.com/en-us/mixedreality/hololens)  
   - [Magic Leap SDK](https://developer.magicleap.com/)  
4. **Model Selection Justification**:  
   - [YOLOv8nano for Edge Devices](https://github.com/ultralytics/yolov8) (Ultra-lightweight, <2MB).  
   - [EfficientNetB0 for Agriculture](https://arxiv.org/abs/2005.11950) (High accuracy with low compute).  
5. **Project Constraints**:  
   - [Allen Institute Model Weights](https://huggingface.co/allenai) (Open-source foundation models).  

---

**Reviewers**: AI Team, Mobile/AR Engineers  
**Next Review Date**: [Insert Date]  


