import gguf

model_path = 'models/SmolLM3-3B-GGUF/ggml-org_SmolLM3-3B-GGUF_SmolLM3-Q4_K_M.gguf'
out_path = 'models/SmolLM3-3B-GGUF/ggml-org_SmolLM3-3B-GGUF_SmolLM3-Q4_K_M_fixed.gguf'

reader = gguf.GGUFReader(model_path)
writer = gguf.GGUFWriter(out_path, 'llama')

for tensor in reader.tensors:
    writer.add_tensor_info(tensor.name, tensor.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

for key, field in reader.fields.items():
    if key != 'tokenizer.chat_template':
        val = field.parts[field.data[0]] if isinstance(field.data, list) and len(field.data) > 0 else field.data[0] if isinstance(field.data, list) else field.data
        writer.add_string(key, str(val))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
writer.close()
print("Successfully generated valid fixed GGUF without the chat_template flag.")
