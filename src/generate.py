import torch

def generate_claim_summary(model, tokenizer, prompt, max_length=150):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Create an attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Set pad_token_id to eos_token_id if it's not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
