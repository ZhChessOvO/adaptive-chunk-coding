// Separate extension: do not change the completed stage-A/B observation bridge.
#include "dmc_common.h"
#define private public
#include "dmc_hts_proxy.h"
#undef private

void supply_base(DMCHTSProxy& p, const at::Tensor& feature, bool reset) {
    TORCH_CHECK(feature.sizes() == p.m_feature_p.sizes(), "base feature geometry");
    p.m_feature_p.copy_(feature);
    if (reset) p.m_feature_i = p.m_recon_head.forward_reset(p.m_feature_p);
    p.m_memory_has_value = !reset;
    p.m_decoded_features_valid = false;
}

void prepare_encoder(DMCHTSProxy& p) {
    p.apply_feature_adaptor(p.m_memory_has_value);
    p.m_ctx = p.m_feature_extractor.forward(p.m_memory);
    p.m_memory_has_value = true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supply_base", &supply_base);
    m.def("prepare_encoder", &prepare_encoder);
}
