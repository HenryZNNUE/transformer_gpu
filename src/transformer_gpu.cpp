#include "transformer_gpu.h"

#ifdef _WIN32
#include <Windows.h>
#endif

TransformerImpl::TransformerImpl(int d_m, int n_h, int n_el, int n_dl, int e_d, int dpo)
{
	auto options = torch::nn::TransformerOptions(d_m, n_h, n_el, n_dl)
		.dim_feedforward(e_d).dropout(dpo);

	transformer = register_module("transformer", torch::nn::Transformer(options));

	auto device = torch::Device(torch::kCUDA);
	this->to(device);
}

torch::Tensor TransformerImpl::forward(torch::Tensor& x)
{
	torch::Tensor out = transformer->forward(x, out);
	return out;
}

void TransformerGPU::print(torch::Tensor& tensor)
{
	std::cout << tensor << std::endl;
}

void TransformerGPU::load_model(std::shared_ptr<TransformerImpl>& transformer_gpu, std::string path)
{
	torch::load(transformer_gpu, path);
}

void TransformerGPU::save_model(std::shared_ptr<TransformerImpl>& transformer_gpu, std::string path)
{
	torch::save(transformer_gpu, path);
}

torch::Tensor TransformerGPU::generate(std::shared_ptr<TransformerImpl>& transformer_gpu,
	torch::Tensor& start_token, int max_length)
{
	transformer_gpu->eval();
	std::vector<torch::Tensor> generated_tokens;
	generated_tokens.emplace_back(start_token);

	auto current_input = start_token;

	for (int i = 0; i < max_length; ++i)
	{
		auto output = transformer_gpu->forward(current_input);
		auto next_token = output.argmax(2);

		generated_tokens.emplace_back(next_token);
		current_input = next_token;

		if (!next_token.item<int>())
		{
			break;
		}
	}

	auto result = torch::cat(generated_tokens, 1);
	return result;
}

void TransformerGPU::train(std::shared_ptr<TransformerImpl>& transformer_gpu, int epoch)
{
	auto criterion = torch::nn::CrossEntropyLoss();
	auto optimizer = torch::optim::AdamW(transformer_gpu->parameters());

	for (int ep = 0; ep < epoch; ++ep)
	{
		auto output = transformer_gpu->forward(input_tensor);
		auto loss = criterion(output, output_tensor);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		std::cout << "Epoch: " << ep << " Loss: " << loss.item<float>() << std::endl;
	}

	char* path;
	GetModuleFileNameA(NULL, path, MAX_PATH);
	(strrchr(path, '\\'))[0] = 0;
	std::stringstream ss;
	ss << path << "\\model\\T-GPU-JIT-ep" << epoch << ".pt";

	torch::save(transformer_gpu, ss.str());
}

void TransformerGPU::test(std::shared_ptr<TransformerImpl>& transformer_gpu)
{
	transformer_gpu->eval();
	auto output = transformer_gpu->forward(input_tensor);
	auto output_argmax = output.argmax(2);
	auto correct = output_argmax.eq(output_tensor).sum().item<int>();
	auto accuracy = (float)correct / output_tensor.numel();
	std::cout << "Accuracy: " << accuracy << std::endl;
}

// Tranformer on NVIDIA GPU with CUDA
// 
// Parameters:
// e_s: embedding size
// n_l: number of layers
// n_h: number of heads
// n_el: number of encoder layers (default: 6)
// n_dl: number of decoder layers (default: 6)
// 
// Library Used: PyTorch C++ API
TransformerGPU::TransformerGPU(int d_m, int n_h, int n_el, int n_dl, int e_d, int dpo)
{
	auto transformer_gpu = std::make_shared<TransformerImpl>(d_m, n_h, n_el, n_dl, e_d, dpo);
}

// Tranformer with Smolgen on NVIDIA GPU with CUDA
// Smolgen is inspired by Leela Chess Zero's Smolgen Architecture
// Link: https://lczero.org/blog/2024/02/transformer-progress/
// 
// Parameters:
// e_s: embedding size
// n_l: number of layers
// n_h: number of heads
// n_el: number of encoder layers (default: 6)
// n_dl: number of decoder layers (default: 6)
// 
// Library Used: PyTorch C++ API
Transformer_SmolgenImpl::Transformer_SmolgenImpl(int d_m, int n_h, int n_el, int n_dl, int e_d, int dpo)
	: d_model(d_m), num_heads(n_h), num_encoder_layers(n_el), num_decoder_layers(n_dl),
	  embedding_dim(e_d), dropout(dpo)
{
	// Embedding
	embedding = register_module("embedding",
		torch::nn::Embedding(torch::nn::EmbeddingOptions(d_model, embedding_dim)));
	// Linear Q
	q_linear = register_module("q_linear",
		torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
	// Linear K
	k_linear = register_module("k_linear",
		torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
	// Linear V
	v_linear = register_module("v_linear",
		torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));
	// Linear Out
	out_linear = register_module("out_linear",
		torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model)));

	// Smolgen
	// Linear 32 (No Bias)
	linear32 = register_module("linear32",
		torch::nn::Linear(torch::nn::LinearOptions(32, 256).bias(false)));
	// Linear 256
	linear256 = register_module("linear256",
		torch::nn::Linear(torch::nn::LinearOptions(256, 256)));
	// LayerNorm 256
	layernorm256 = register_module("layernorm256",
		torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 256 })));

	// Linear 256 (Head Size)
	for (int i = 0; i < num_heads; ++i)
	{
		linear256_head.emplace_back(register_module("linear256_head_" + std::to_string(i),
			torch::nn::Linear(torch::nn::LinearOptions(256, 256))));
		layernorm256_head.emplace_back(register_module("layernorm256_head_" + std::to_string(i),
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 256 }))));
	}

	// Shared Linear 64
	for (int i = 0; i < num_heads; i += 2)
	{
		shared_linear64.emplace_back(register_module("shared_linear64_" + std::to_string(i),
			torch::nn::Linear(torch::nn::LinearOptions(64, 64))));
		// Sharing weights
		shared_linear64.emplace_back(register_module("shared_linear64_" + std::to_string(i++),
			torch::nn::Linear(shared_linear64[i])));
	}

	// LayerNorm
	layernorm = register_module("layernorm",
		torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));

	// Multi-Head Attention Dropout
	attn_dropout = register_module("attn_dropout",
		torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));

	// Positional-Wise Fully Connected Feed-Forward Network
	ffn1 = register_module("ffn1",
		torch::nn::Linear(torch::nn::LinearOptions(d_model, 512)));

	ffn2 = register_module("ffn2",
		torch::nn::Linear(torch::nn::LinearOptions(512, d_model)));

	// FFN Dropout
	ffn_dropout = register_module("ffn_dropout",
		torch::nn::Dropout(torch::nn::DropoutOptions(dropout)));

	// Output LayerNorm
	out_layernorm = register_module("out_layernorm",
		torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model })));

	// Decoder Layer
	decoder_layer = register_module("decoder_layer",
		torch::nn::TransformerDecoderLayer(torch::nn::TransformerDecoderLayerOptions(d_model, num_heads)
			.dim_feedforward(embedding_dim).dropout(dropout)));

	// Decoder
	decoder = register_module("decoder",
		torch::nn::TransformerDecoder(decoder_layer, num_decoder_layers));

	auto device = torch::Device(torch::kCUDA);
	this->to(device);
}

torch::Tensor Transformer_SmolgenImpl::forward(torch::Tensor& x, bool m_attn_mask, bool use_decoder)
{
	// Multi-Head Attention with Smolgen
	// Embedding
	auto x_embed = embedding(x);
	// Linear Q
	auto q = q_linear(x_embed);
	// Linear K
	auto k = k_linear(x_embed);
	// Linear V
	auto v = v_linear(x_embed);

	// Multi-Head Attention
	auto q_reshaped = q.reshape({ q.size(0), q.size(1), num_heads, d_model / num_heads }).permute({ 0, 2, 1, 3});
	auto k_reshaped = k.reshape({ k.size(0), k.size(1), num_heads, d_model / num_heads }).permute({ 0, 2, 1, 3 });
	auto v_reshaped = v.reshape({ v.size(0), v.size(1), num_heads, d_model / num_heads }).permute({ 0, 2, 1, 3 });

	// Smolgen forward
	auto linear32_out = linear32(x_embed);
	auto linear256_out = linear256(linear32_out);
	auto layernorm256_out = layernorm256(linear256_out);
	torch::Tensor layernorm256_head_out, shared_linear64_out;

	for (int i = 0; i < num_heads; ++i)
	{
		auto linear256_head_out = linear256_head[i](layernorm256_out);
		layernorm256_head_out = layernorm256_head[i](linear256_head_out);
	}

	for (int i = 0; i < num_heads; i += 2)
	{
		shared_linear64_out = shared_linear64[i]->forward(layernorm256_head_out[i]);
		shared_linear64_out = shared_linear64[i++]->forward(layernorm256_head_out[i]);
	}

	auto sl64_reshaped = shared_linear64_out.reshape(
		{ shared_linear64_out.size(0), shared_linear64_out.size(1), num_heads, 64 })
		.permute({ 0, 2, 1, 3 });

	auto attn_v = torch::einsum("bqhd,bkhd->bhqk",
		{ q_reshaped, k_reshaped }) / sqrt(d_model / num_heads);
	
	if (m_attn_mask)
	{
		auto mask = torch::tril(torch::ones({ x.size(1), x.size(1) })).to(torch::kCUDA);
		attn_v.masked_fill_(mask == 0, -std::numeric_limits<float>::infinity());
	}

	attn_v = torch::softmax(attn_v + sl64_reshaped, -1) * v_reshaped;

	attn_v = attn_v.permute({ 0, 2, 1, 3 }).contiguous().view({ q.size(0), q.size(1), num_heads });

	auto attention = out_linear(attn_v);

	// Add & Norm
	auto m_attn_out = attn_dropout(attention);
	m_attn_out = layernorm(attention + x_embed);

	// Positional-Wise Fully Connected Feed-Forward Network
	auto ffn_out = ffn1(m_attn_out);
	ffn_out = torch::relu(ffn_out);
	ffn_out = ffn2(ffn_out);
	ffn_out = ffn_dropout(ffn_out);

	auto out = out_layernorm(m_attn_out + ffn_out);

	if (decoder)
	{
		out = decoder->forward(out, out);
	}

	return out;
}

void TransformerGPU_Smolgen::print(torch::Tensor& tensor)
{
	std::cout << tensor << std::endl;
}

void TransformerGPU_Smolgen::load_model(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
	std::string path)
{
	torch::load(transformer_smolgen, path);
}

void TransformerGPU_Smolgen::save_model(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
	std::string path)
{
	torch::save(transformer_smolgen, path);
}

torch::Tensor TransformerGPU_Smolgen::generate(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
	torch::Tensor& start_token, int max_length,
	bool m_attn_mask, bool use_decoder)
{
	transformer_smolgen->eval();

	std::vector<torch::Tensor> generated_tokens;
	generated_tokens.emplace_back(start_token);

	auto current_input = start_token;

	for (int i = 0; i < max_length; ++i)
	{
		auto output = transformer_smolgen->forward(current_input, m_attn_mask, use_decoder);
		auto next_token = output.argmax(2);

		generated_tokens.emplace_back(next_token);
		current_input = next_token;

		if (!next_token.item<int>())
		{
			break;
		}
	}

	auto result = torch::cat(generated_tokens, 1);
	return result;
}

void TransformerGPU_Smolgen::train(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
	int epoch, bool m_attn_mask, bool use_decoder)
{
	auto criterion = torch::nn::CrossEntropyLoss();
	auto optimizer = torch::optim::AdamW(transformer_smolgen->parameters());

	for (int ep = 0; ep < epoch; ++ep)
	{
		auto output = transformer_smolgen->forward(input_tensor, m_attn_mask, use_decoder);
		auto loss = criterion(output, output_tensor);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		std::cout << "Epoch: " << ep << " Loss: " << loss.item<float>() << std::endl;
	}

	char* path;
	GetModuleFileNameA(NULL, path, MAX_PATH);
	(strrchr(path, '\\'))[0] = 0;
	std::stringstream ss;
	ss << path << "\\model\\TS-GPU-JIT-" << (m_attn_mask ? "mam-" : "")
	   << (use_decoder ? "ud-" : "") << "ep" << epoch << ".pt";

	save_model(transformer_smolgen, ss.str());
}

void TransformerGPU_Smolgen::test(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
	bool m_attn_mask, bool use_decoder)
{
	transformer_smolgen->eval();
	output_tensor = transformer_smolgen->forward(input_tensor, m_attn_mask, use_decoder);
	auto output_argmax = output_tensor.argmax(2);
	auto correct = output_argmax.eq(output_tensor).sum().item<int>();
	auto accuracy = (float)correct / output_tensor.numel();
	std::cout << "Accuracy: " << accuracy << std::endl;
}

TransformerGPU_Smolgen::TransformerGPU_Smolgen(int d_m, int n_h, int n_el, int n_dl, int e_d, int dpo)
{
	auto transformer_smolgen = std::make_shared<Transformer_SmolgenImpl>(d_m, n_h, n_el, n_dl, e_d, dpo);
}