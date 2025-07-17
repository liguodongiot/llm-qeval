




def generate_text(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda")
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_p=0.8,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)



