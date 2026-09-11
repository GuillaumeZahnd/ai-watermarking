def paraphrase_attack(
    model,
    tokenizer,
    text: str,
    query: str | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> str:
    query_context = (
        f"\n\nFor context, this text is a response to the following query. "
        f"Use it only to preserve meaning and relevant context accurately; "
        f"do not include the query itself in your output, and do not "
        f"paraphrase it.\nQuery:\n{query}"
        if query
        else ""
    )

    prompt = (
        "Rewrite the following text so that it expresses exactly the same "
        "meaning, facts, and logical structure, but uses different wording "
        "and sentence construction wherever possible. Do not add, omit, or "
        "alter any information. Do not change the tone. Output only the "
        "rewritten text, nothing else."
        f"{query_context}"
        f"\n\nText:\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
