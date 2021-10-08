import torch
from torchnlp.encoders.text.text_encoder import stack_and_pad_tensors
from transformers import AutoConfig, AutoTokenizer


class Tokenizer:

    def __init__(self, pretrained_model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.pad_index = self.tokenizer.pad_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.bos_index = self.tokenizer.eos_token_id
        self.stoi = self.tokenizer.get_vocab()
        self.itos = {v: k for k, v in self.stoi.items()}
        self.configs = AutoConfig.from_pretrained(pretrained_model)
        self.max_positions = self.configs.max_position_embeddings
        
    def batch_encode(self, sources: list, hypothesis: list):
        SPIECE_UNDERLINE = "▁"
        encoded_batch = {"input_ids": [], "attention_mask": [], "mt_eos_ids": [],
                         "first_sentence_mask": [], "first_piece_mask": []}
        for mt, src in zip(hypothesis, sources):
            # format:  <s> A </s> | </s> B </s>
            #          --- MT ----|---- SRC ---
            src_inputs = self.tokenizer(src)
            src_inputs["input_ids"][0] = 2
            mt_inputs = self.tokenizer(mt)
            input_ids = mt_inputs["input_ids"] + src_inputs["input_ids"]
            attention_mask = mt_inputs["attention_mask"] + src_inputs["attention_mask"]
            first_sentence_mask = [1] * len(mt_inputs["attention_mask"]) + [0] * len(src_inputs["attention_mask"])

            if len(attention_mask) > self.max_positions-2:
                attention_mask = attention_mask[:self.max_positions-2]
                first_sentence_mask = first_sentence_mask[:self.max_positions-2]
                input_ids = input_ids[:self.max_positions-2]

            encoded_batch["input_ids"].append(input_ids)
            encoded_batch["attention_mask"].append(attention_mask)
            encoded_batch["mt_eos_ids"].append(len(mt_inputs["input_ids"]))
            encoded_batch["first_sentence_mask"].append(first_sentence_mask)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            encoded_batch["first_piece_mask"].append([int(s.startswith(SPIECE_UNDERLINE)) for s in tokens])

        model_input = {}
        for k, v in encoded_batch.items():
            if k == "mt_eos_ids":
                model_input["mt_eos_ids"] = torch.tensor(encoded_batch["mt_eos_ids"])
            else:
                padded_input = stack_and_pad_tensors([torch.tensor(l) for l in v])
                model_input[k] = padded_input.tensor

        return model_input
