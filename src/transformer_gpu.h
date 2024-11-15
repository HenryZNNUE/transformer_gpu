#pragma once

#include <torch/torch.h>
#include <torch/script.h>

struct TransformerImpl : torch::nn::Module
{
	torch::nn::Transformer transformer = nullptr;

	TransformerImpl(int d_m = 512, int n_h = 8,
		int n_el = 6, int n_dl = 6,
		int e_d = 2048, int dpo = 0.1);

	torch::Tensor forward(torch::Tensor& x);
};

struct TransformerGPU : public torch::nn::Module
{
	torch::Tensor input_tensor, output_tensor;

	void print(torch::Tensor& tensor);
	void load_model(std::shared_ptr<TransformerImpl>& transformer_gpu, std::string path);
	void save_model(std::shared_ptr<TransformerImpl>& transformer_gpu, std::string path);
	torch::Tensor generate(std::shared_ptr<TransformerImpl>& transformer_gpu, torch::Tensor& start_token, int max_length = 2048);
	void train(std::shared_ptr<TransformerImpl>& transformer_gpu, int epoch = 30);
	void test(std::shared_ptr<TransformerImpl>& transformer_gpu);
	TransformerGPU(int d_m = 512, int n_h = 8,
		int n_el = 6, int n_dl = 6,
		int e_d = 2048, int dpo = 0.1);
};

// Inspired by Leela Chess Zero's Transformer Architecture
// Link: https://lczero.org/blog/2024/02/transformer-progress/
struct Transformer_SmolgenImpl : torch::nn::Module
{
	int d_model,
		num_heads,
		num_encoder_layers, num_decoder_layers,
		embedding_dim, dropout;

	Transformer_SmolgenImpl(int d_m = 512, int n_h = 8,
		int n_el = 6, int n_dl = 6,
		int e_d = 2048, int dpo = 0.1);

	torch::Tensor forward(torch::Tensor& x, bool m_attn_mask = false, bool use_decoder = false);

	torch::nn::Linear linear32 = nullptr;
	torch::nn::Linear  linear256 = nullptr;
	std::vector<torch::nn::Linear> linear256_head = {};
	std::vector<torch::nn::Linear> shared_linear64 = {};

	torch::nn::LayerNorm layernorm256 = nullptr;
	std::vector<torch::nn::LayerNorm> layernorm256_head = {};

	torch::nn::Embedding embedding = nullptr;
	torch::nn::Linear q_linear = nullptr;
	torch::nn::Linear k_linear = nullptr;
	torch::nn::Linear v_linear = nullptr;
	torch::nn::Linear out_linear = nullptr;

	torch::nn::Dropout attn_dropout = nullptr;
	torch::nn::LayerNorm layernorm = nullptr;

	torch::nn::Linear ffn1 = nullptr;
	torch::nn::Linear ffn2 = nullptr;
	torch::nn::Dropout ffn_dropout = nullptr;

	torch::nn::LayerNorm out_layernorm = nullptr;

	torch::nn::TransformerDecoderLayer decoder_layer = nullptr;
	torch::nn::TransformerDecoder decoder = nullptr;
};

struct TransformerGPU_Smolgen : public torch::nn::Module
{
	torch::Tensor input_tensor, output_tensor;

	void print(torch::Tensor& tensor);
	void load_model(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
		std::string path);
	void save_model(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
		std::string path);
	torch::Tensor generate(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
		torch::Tensor& start_token, int max_length = 2048,
		bool m_attn_mask = false, bool use_decoder = false);
	void train(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
		int epoch = 30, bool m_attn_mask = false, bool use_decoder = false);
	void test(std::shared_ptr<Transformer_SmolgenImpl>& transformer_smolgen,
		bool m_attn_mask = false, bool use_decoder = false);
	TransformerGPU_Smolgen(int d_m = 512, int n_h = 8,
		int n_el = 6, int n_dl = 6,
		int e_d = 2048, int dpo = 0.1);
};

TORCH_MODULE(Transformer);
TORCH_MODULE(Transformer_Smolgen);