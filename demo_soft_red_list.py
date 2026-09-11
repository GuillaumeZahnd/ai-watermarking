import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    )

from soft_red_list import (
    SecretGenerator,
    SoftRedListLogitsProcessor,
    SoftRedListDetector,
    )

from utils import print_results
from paraphrase_attack import paraphrase_attack


if __name__ == "__main__":

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MAX_NEW_TOKENS = 600
    DO_SAMPLE = True
    TEMPERATURE = 0.7
    SECRET_KEY = 217608
    MANTISSA = 35473991
    EXPONENT = 32
    GAMMA = 0.5  # Proportion of green tokens in the vocabulary
    DELTA = 2.5  # Boost the scores of the green tokens
    NB_TOKENS_MIN = 20
    IS_PROMPT_AVAILABLE = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        )

    secret_generator = SecretGenerator(
        hash_key=SECRET_KEY,
        mantissa=MANTISSA,
        exponent=EXPONENT,
        )

    watermark_processor = SoftRedListLogitsProcessor(
        secret_generator=secret_generator,
        vocab_size=len(tokenizer),
        gamma=GAMMA,
        delta=DELTA,
        )

    logits_processors = LogitsProcessorList([watermark_processor])

    watermark_detector = SoftRedListDetector(
        secret_generator=secret_generator,
        tokenizer=tokenizer,
        gamma=GAMMA,
        )

    # Input prompt
    prompt = "Explain why the sky appears blue during the day, and why the sky turns to red during sunset:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # Unmarked output
    output_ids_unmarked = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        )
    if IS_PROMPT_AVAILABLE:
        unmarked_text = tokenizer.decode(output_ids_unmarked[0], skip_special_tokens=True)
        analysis_unmarked = watermark_detector.detect(text=unmarked_text, prompt=prompt)
    else:
        unmarked_text = tokenizer.decode(output_ids_unmarked[0][input_length:], skip_special_tokens=True)
        analysis_unmarked = watermark_detector.detect(text=unmarked_text, prompt=None)
    print(f"\n[Unmarked output]\n{unmarked_text}\n")
    print_results(
        nb_green_tokens=analysis_unmarked.nb_green_tokens,
        nb_tokens=analysis_unmarked.nb_tokens,
        z_score=analysis_unmarked.z_score,
        nb_tokens_min=NB_TOKENS_MIN,
        )

    # Watermarked output
    output_ids_watermarked = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        logits_processor=logits_processors,  # Watermark processor
        )
    if IS_PROMPT_AVAILABLE:
        watermarked_text = tokenizer.decode(output_ids_watermarked[0], skip_special_tokens=True)
        analysis_watermarked = watermark_detector.detect(watermarked_text, prompt)
    else:
        watermarked_text = tokenizer.decode(output_ids_watermarked[0][input_length:], skip_special_tokens=True)
        analysis_watermarked = watermark_detector.detect(text=watermarked_text, prompt=None)
    print(f"\n[Watermarked output]\n{watermarked_text}\n")
    print_results(
        nb_green_tokens=analysis_watermarked.nb_green_tokens,
        nb_tokens=analysis_watermarked.nb_tokens,
        z_score=analysis_watermarked.z_score,
        nb_tokens_min=NB_TOKENS_MIN,
        )

    # Free memory
    del model
    torch.cuda.empty_cache()

    # Paraphrase attack
    paraphrase_model_id = "Qwen/Qwen2.5-7B-Instruct"
    paraphrase_tokenizer = AutoTokenizer.from_pretrained(paraphrase_model_id)
    paraphrase_model = AutoModelForCausalLM.from_pretrained(
        paraphrase_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if IS_PROMPT_AVAILABLE:
        query=prompt
    else:
        query=None

    paraphrased_text = paraphrase_attack(
        model=paraphrase_model,
        tokenizer=paraphrase_tokenizer,
        text=watermarked_text,
        query=query,
        temperature=0.9,
        max_new_tokens=MAX_NEW_TOKENS,
        )

    analysis_paraphrased = watermark_detector.detect(text=paraphrased_text, prompt=None)
    print(f"\n[Paraphrased output]\n{paraphrased_text}\n")
    print_results(
        nb_green_tokens=analysis_paraphrased.nb_green_tokens,
        nb_tokens=analysis_paraphrased.nb_tokens,
        z_score=analysis_paraphrased.z_score,
        nb_tokens_min=NB_TOKENS_MIN,
        )
