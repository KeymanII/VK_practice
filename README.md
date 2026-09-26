## Mitigating catastrophic forgetting via L2LoRA


### Overview
  This project provides complete pipeline for VLM(LLaVA) finetuning with implementation of custom L2-Lora[1] Trainer. Evaluation shows L2-LoRA outperforms
standard QLoRA finetuning.
**The L2-LoRA method is not my invention; all credit for the algorithm goes to the authors. This repository is an independent implementation and empirical study.**

### Inference

The L2-LoRA adapter is trained on top of
[`deepvk/llava-gemma-2b-lora`](https://huggingface.co/deepvk/llava-gemma-2b-lora),
which is itself a LoRA-adapted LLaVA model based on Gemma-2b.

```python
import kagglehub, torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from peft import PeftModel

adapter_path = f"{kagglehub.model_download("andreykaymanov/llava-l2lora")}/l2-adapter/llava-gemma-simple-l2"

base_id = "deepvk/llava-gemma-2b-lora"
model = LlavaForConditionalGeneration.from_pretrained(
    base_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(base_id)
tokenizer = AutoTokenizer.from_pretrained(base_id)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
```

### References

[1] Zhang X. et al. L2-LoRA: Improving Low-Rank Adaptation with Layer-Specific Regularization. AAAI, 2026.
    https://ojs.aaai.org/index.php/AAAI/article/view/40784
