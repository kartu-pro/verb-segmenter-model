"""
inference.py
------------
Python-native inference library supporting all 4 trained model variants.
Optimized for clean function calls, IDE autocompletion, and batch CSV processing.
"""

from typing import Literal, Optional, List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import seq_tagger
import seq_2_seq
from tagger_models import BiLSTMMetadataTagger


# ==========================================
# CACHED MODEL LOADER (Avoids reloading per row)
# ==========================================
_MODEL_CACHE = {}

def _load_cached_model(model_path: str, model_type: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "tagger":
        checkpoint = torch.load(model_path, map_location="cpu")
        model = BiLSTMMetadataTagger(checkpoint["vocabs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        artifact = {"model": model, "vocabs": checkpoint["vocabs"], "strategy": checkpoint["strategy"], "device": device}
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        artifact = {"model": model, "tokenizer": tokenizer, "device": device}
        
    _MODEL_CACHE[model_path] = artifact
    return artifact


# ==========================================
# CORE PREDICTION FUNCTION
# ==========================================
def predict(
    model_type: Literal["tagger", "seq2seq"],
    variant: Literal["boundaries", "bio", "single", "group6"],
    model_path: str,
    *,
    form: Optional[str] = None,
    lemma: Optional[str] = None,
    verb_class: Optional[Union[str, int]] = None,
    screeve: Optional[str] = None,
    subject: Optional[str] = None,
    forms: Optional[List[str]] = None
) -> Union[str, List[str]]:
    """
    Unified prediction function for Georgian verb morpheme segmentation.
    """
    
    if model_type == "tagger":
        if not form or not screeve or not subject:
            raise ValueError("Tagger inference requires: form, screeve, and subject.")
        
        artifact = _load_cached_model(model_path, "tagger")
        model, vocabs, strategy, device = artifact["model"], artifact["vocabs"], artifact["strategy"], artifact["device"]
        
        chars = ["^"] + list(form) if strategy == "boundaries" else list(form)
        c2id, s2id, subj2id, v2id = vocabs["char2id"], vocabs["screeve2id"], vocabs["subject2id"], vocabs["class2id"]
        id2label = {v: k for k, v in vocabs["label2id"].items()}

        token_ids = [c2id.get(c, c2id["<UNK>"]) for c in chars]
        
        with torch.no_grad():
            t_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
            t_screeve = torch.tensor([s2id.get(screeve, 0)], dtype=torch.long, device=device)
            t_subject = torch.tensor([subj2id.get(subject, 0)], dtype=torch.long, device=device)
            t_class = torch.tensor([v2id.get(str(verb_class), 0)], dtype=torch.long, device=device)

            logits = model(t_tokens, t_screeve, t_subject, t_class)
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        valid_preds = [id2label[p] for p in preds]
        return seq_tagger.decode(chars, valid_preds, strategy=strategy)

    elif model_type == "seq2seq":
        artifact = _load_cached_model(model_path, "seq2seq")
        model, tokenizer, device = artifact["model"], artifact["tokenizer"], artifact["device"]

        if variant == "single":
            if not form or not lemma or not subject:
                raise ValueError("Seq2Seq 'single' mode requires: lemma, form, subject, screeve, verb_class.")
            input_text = f"lemma: {lemma} | class: {verb_class} | screeve: {screeve} | subject: {subject} | surface: {form}"
        elif variant == "group6":
            if not forms or len(forms) != 6 or not lemma:
                raise ValueError("Seq2Seq 'group6' mode requires: lemma and a list of 6 forms.")
            input_text = f"lemma: {lemma} | class: {verb_class} | screeve: {screeve} | paradigm: {' | '.join(forms)}"
        else:
            raise ValueError(f"Invalid seq2seq variant: {variant}")

        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return seq_2_seq.decode(decoded, mode=variant)