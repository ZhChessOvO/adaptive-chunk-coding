// Research instrumentation for the exact installed HT-S CUDA implementation.
// No historical source or installed library is modified. Access control only
// is relaxed in this translation unit; class layout/ABI is unchanged. The
// Python loader binds the binary and header hashes. Never load against another
// HT-S ABI. Use only under the project's exclusive native-GPU lock.
#include "dmc_common.h"
#define private public
#include "dmc_hts_proxy.h"
#undef private
#include "common_cpp.h"
#include "def_cutlass.h"
#include "def_elementwise.h"

using Tensors = std::map<std::string, at::Tensor>;

Tensors context(DMCHTSProxy& p) {
    TORCH_CHECK(p.m_memory.defined(), "initialize a reference first");
    return {{"memory", p.m_memory.clone()}, {"ctx", p.m_ctx.clone()},
            {"feature_i", p.m_feature_i.clone()}, {"feature_p", p.m_feature_p.clone()},
            {"has_memory", at::tensor(static_cast<int>(p.m_memory_has_value))},
            {"features_valid", at::tensor(static_cast<int>(p.m_decoded_features_valid))}};
}

void restore(DMCHTSProxy& p, const Tensors& c) {
    for (auto pair : {std::make_pair(&p.m_memory, "memory"),
                      std::make_pair(&p.m_ctx, "ctx"),
                      std::make_pair(&p.m_feature_i, "feature_i"),
                      std::make_pair(&p.m_feature_p, "feature_p")}) {
        TORCH_CHECK(pair.first->sizes() == c.at(pair.second).sizes(), "context geometry changed");
        pair.first->copy_(c.at(pair.second));
    }
    p.m_memory_has_value = c.at("has_memory").item<bool>();
    p.m_decoded_features_valid = c.at("features_valid").item<bool>();
}

Tensors encoded(DMCHTSProxy& p) {
    // Call immediately after native compress, before another codec operation.
    return {{"symbols", p.m_y_q.clone()}, {"z", p.m_z_hat.clone()},
            {"latent", p.m_y_hat.clone()}, {"params", p.m_common_params.clone()}};
}

Tensors priors(DMCHTSProxy& p, const at::Tensor& z, int qp) {
    TORCH_CHECK(qp >= 0 && qp < 64 && p.m_skip_threshold == 0.f, "unsupported q/skip");
    TORCH_CHECK(z.sizes() == p.m_z_hat.sizes(), "hyperlatent geometry mismatch");
    p.m_z_hat.copy_(z);
    p.m_temporal_input = multiply_with_broadcast_cuda(p.m_memory, p.m_q_feature[qp], p.m_temporal_input);
    p.m_temporal_params = p.m_temporal_prior_encoder.forward(p.m_temporal_input);
    p.m_hyper_params_pad4 = p.m_hyper_decoder.forward(p.m_z_hat);
    if (p.m_hyper_params_pad4.sizes() != p.m_hyper_params.sizes()) {
        p.m_hyper_params.copy_(p.m_hyper_params_pad4.slice(2, 0, p.m_temporal_params.size(2))
                                                    .slice(3, 0, p.m_temporal_params.size(3)));
    }
    p.m_common_params = p.m_y_prior_fusion.forward(p.m_cat_prior_fusion);
    auto [qdec, scales, means] = chunk_tensors_3(p.m_common_params);
    return {{"qdec", qdec}, {"scales", scales}, {"means", means}};
}

Tensors inspect_prior(DMCHTSProxy& p, const at::Tensor& z, int qp, const Tensors& c) {
    auto saved = context(p);
    restore(p, c);
    auto out = priors(p, z, qp);
    for (auto& item : out) item.second = item.second.clone();
    restore(p, saved);
    return out;
}

Tensors synthesize(DMCHTSProxy& p, const at::Tensor& z, const at::Tensor& symbols,
                   int qp, const Tensors& c) {
    TORCH_CHECK(symbols.sizes() == p.m_y_q_r.sizes(), "symbol geometry mismatch");
    auto saved = context(p);
    restore(p, c);
    auto pr = priors(p, z, qp);
    p.m_y_q_r.copy_(symbols);
    Tensors out;
    out["means0"] = pr.at("means").clone();
    out["qdec"] = pr.at("qdec").clone();
    out["scales"] = pr.at("scales").clone();
    p.m_y_hat = restore_y_cuda(p.m_y_q_r, pr.at("means"), p.m_mask_0, p.m_y_hat);
    p.m_reduced_params = conv1x1_bias(get_gpu_sm(), p.m_common_params,
        p.m_y_spatial_prior_reduction.weight, p.m_y_spatial_prior_reduction.bias, p.m_reduced_params);
    p.m_means = p.m_y_spatial_prior.forward(p.m_y_spatial_prior_adaptor_1.forward(p.m_cat_spatial_prior_adaptor));
    out["means1"] = p.m_means.clone();
    p.m_y_hat = restore_y_and_add_inplace_cuda(p.m_y_q_r, p.m_means, p.m_mask_1, p.m_y_hat);
    p.m_means = p.m_y_spatial_prior.forward(p.m_y_spatial_prior_adaptor_2.forward(p.m_cat_spatial_prior_adaptor));
    out["means2"] = p.m_means.clone();
    p.m_y_hat = restore_y_and_add_inplace_cuda(p.m_y_q_r, p.m_means, p.m_mask_2, p.m_y_hat);
    p.m_means = p.m_y_spatial_prior.forward(p.m_y_spatial_prior_adaptor_3.forward(p.m_cat_spatial_prior_adaptor));
    out["means3"] = p.m_means.clone();
    p.m_y_hat = restore_y_and_add_multiply_inplace_cuda(p.m_y_q_r, p.m_means, p.m_mask_3, p.m_y_hat, pr.at("qdec"));
    out["latent"] = p.m_y_hat.clone();
    p.m_feature_p = p.m_decoder.forward(p.m_y_hat, p.m_cat_decoder, p.m_q_decoder[qp]);
    out["feature"] = p.m_feature_p.clone();
    std::tie(p.m_x_hat, p.m_feature_i) = p.m_recon_head.forward(p.m_feature_p);
    for (int i = 0; i < 8; ++i) out["frame" + std::to_string(i)] = p.m_x_hat[i].clone();
    // Display-only operation: no transition of base memory/reference.
    restore(p, saved);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("context", &context);
    m.def("restore", &restore);
    m.def("encoded", &encoded);
    m.def("prior", &inspect_prior);
    m.def("synthesize", &synthesize);
}
