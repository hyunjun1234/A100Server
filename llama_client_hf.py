# experiments/unified_llama8b/llama_client_hf.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaHFClient:
    def __init__(
        self,
        model_id: str       = "meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens: int = 4096,
        temperature: float  = 0.2,
        top_p: float        = 0.95,
        use_4bit: bool      = False,
    ):
        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p

        token = os.environ.get("HF_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.model.eval()

    @torch.inference_mode()
    def generate_messages(
        self,
        messages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        OpenAI-compatible chat messages를 그대로 받아서 생성한다.
        CudaForge처럼 JSON/code/feedback 등 다양한 형식을 요구하는 framework용.
        여기서는 code-only system prompt를 강제로 붙이지 않는다.
        """

        def _apply(msgs):
            return self.tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )

        try:
            inputs = _apply(messages)
        except Exception as e:
            # 일부 chat template(예: AutoTriton / Seed-Coder)은 "system" role을
            # 거부한다 -> system 내용을 첫 user 메시지에 합쳐서 재시도.
            if "system" not in str(e).lower():
                raise
            sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
            merged = [dict(m) for m in messages if m.get("role") != "system"]
            prefix = "\n\n".join(sys_parts)
            placed = False
            for m in merged:
                if m.get("role") == "user":
                    m["content"] = prefix + "\n\n" + m["content"]
                    placed = True
                    break
            if not placed:
                merged.insert(0, {"role": "user", "content": prefix})
            inputs = _apply(merged)

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
            if k != "token_type_ids"   # 일부 토크나이저는 generate가 안 받는 token_type_ids를 반환한다
        }

        gen_temperature = self.temperature if temperature is None else temperature
        gen_top_p = self.top_p if top_p is None else top_p
        gen_max_new_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=gen_max_new_tokens,
            do_sample=(gen_temperature > 0.0),
            temperature=gen_temperature,
            top_p=gen_top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return text.strip()