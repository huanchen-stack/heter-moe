from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch
import re

from quantization import create_fake_quantized_module, Precision


class MultiPrecisionMoEModel:
    def __init__(
            self, 
            model_name="allenai/OLMoE-1B-7B-0924", 
            cache_dir="/models/",
            quantization_dtypes=[Precision.MXFP8],
            quantized_cache_dir="/models/quantized_experts/",
            force_quantization=True,
            autostore_quantized_experts=False
        ):

        cache_dir = Path(cache_dir) / model_name.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantized_cache_dir = Path(quantized_cache_dir) / model_name.replace("/", "_")
        quantized_cache_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )

        print("Downloading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        print("Done!")
        
        self.quantization_dtypes = quantization_dtypes

        print("Quantizing experts... to precisions:", 
              [p.value for p in quantization_dtypes])

        self.quantized_experts = None

        if not force_quantization:
            self._load_quantized_experts(quantized_cache_dir)
        if not self.quantized_experts:
            self._quantize_experts()
        if autostore_quantized_experts:
            self._save_quantized_experts(quantized_cache_dir)

    def _quantize_experts(self):
        self.quantized_experts = {}

        for precision in self.quantization_dtypes:
            self.quantized_experts[precision.value] = {}

            for name, module in self.model.named_modules():
                if not ("experts" in name and "_proj" in name):
                    continue
                
                match = re.search(r'layers\.(\d+)\..*experts\.(\d+)\.(\w+_proj)', name)
                assert match is not None, f"Could not parse: {name}"

                layer_id = int(match.group(1))
                expert_id = int(match.group(2))
                if (layer_id, expert_id) not in self.quantized_experts[precision.value]:
                    self.quantized_experts[precision.value][(layer_id, expert_id)] = {}
                proj_type = str(match.group(3))
                quant_linear = create_fake_quantized_module(
                    module, precision=precision
                )

                self.quantized_experts[precision.value][(layer_id, expert_id)][proj_type] = quant_linear
        print("Quantization of experts complete.")

    def _load_quantized_experts(self, quantized_cache_dir: Path):
        quantized_experts_path = quantized_cache_dir / "quantized_experts.pt"
        if quantized_experts_path.exists():
            self.quantized_experts = torch.load(quantized_experts_path)
            print("Loaded quantized experts from cache.")
        else:
            print("No cached quantized experts found.")
    
    def _save_quantized_experts(self, quantized_cache_dir: Path):
        torch.save(
            self.quantized_experts, 
            quantized_cache_dir / "quantized_experts.pt"
        )
        print("Saved quantized experts to cache.")



if __name__ == "__main__":
    model = MultiPrecisionMoEModel(
        model_name="allenai/OLMoE-1B-7B-0924",
        quantization_dtypes=[Precision.MXFP8, Precision.MXFP4]
    )
    print("Model and quantized experts ready.")
    sample_layer, sample_expert = 0, 0
    print(f"Sample quantized expert for layer {sample_layer}, expert {sample_expert}:")
    print(model.quantized_experts[Precision.MXFP8.value][(sample_layer, sample_expert)]["down_proj"].weight)
    print(model.quantized_experts[Precision.MXFP4.value][(sample_layer, sample_expert)]["down_proj"].weight)
    print("Done.")