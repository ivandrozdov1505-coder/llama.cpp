#include "models.h"

#include <cstdlib>
#include <stdexcept>
#include "ggml-alloc.h"

// Own immutable, precomputed model coefficients for the lifetime of this graph.
// Keeping these outside the compute graph avoids repeating casts and scalar
// arithmetic for every decoded token, including every recurrent-layer visit.
class dory_frozen_scales final : public llm_graph_input_i {
public:
    explicit dory_frozen_scales(ggml_context * compute_ctx) : compute_ctx(compute_ctx) {}
    ggml_context * compute_ctx;
    std::map<const ggml_tensor *, ggml_tensor *> baked;
    std::vector<ggml_context *> contexts;
    std::vector<ggml_backend_buffer_t> buffers;

    ~dory_frozen_scales() override {
        for (auto * buffer : buffers) ggml_backend_buffer_free(buffer);
        for (auto * ctx : contexts) ggml_free(ctx);
    }
    void set_input(const llama_ubatch *) override {}
    bool can_reuse(const llm_graph_params &) override { return true; }

    ggml_tensor * get(ggml_tensor * source, float factor, int transform = 0, ggml_backend_buffer_t placement = nullptr) {
        const auto found = baked.find(source);
        if (found != baked.end()) return found->second;
        // The memory-fit pass builds a graph using unallocated placeholder weights.
        // Keep the ordinary graph for this dry run; bake only loaded weights.
        if (!source->data) {
            auto * tensor = source->type == GGML_TYPE_F32 ? source : ggml_cast(compute_ctx, source, GGML_TYPE_F32);
            tensor = ggml_scale(compute_ctx, tensor, factor);
            if (transform == 1) tensor = ggml_sqr(compute_ctx, tensor);
            if (transform == 2) tensor = ggml_abs(compute_ctx, tensor);
            baked[source] = tensor;
            return tensor;
        }
        GGML_ASSERT(ggml_is_contiguous(source) && source->buffer);
        const auto count = ggml_nelements(source);
        std::vector<uint8_t> raw(ggml_nbytes(source));
        std::vector<float> values(count);
        ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
        if (source->type == GGML_TYPE_F32) {
            memcpy(values.data(), raw.data(), raw.size());
        } else {
            auto * traits = ggml_get_type_traits(source->type);
            GGML_ASSERT(traits->to_float);
            traits->to_float(raw.data(), values.data(), count);
        }
        for (auto & value : values) {
            value *= factor;
            if (transform == 1) value *= value;
            if (transform == 2) value = fabsf(value);
        }
        auto * ctx = ggml_init({ggml_tensor_overhead() + 1024, nullptr, true});
        if (!ctx) throw std::runtime_error("Dory: cannot allocate coefficient metadata");
        contexts.push_back(ctx);
        auto * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, source->ne);
        ggml_format_name(tensor, "dory.frozen.%s", source->name);
        // BF16 vectors can be hosted on CPU because a backend lacks BF16 MUL.
        // Their F32 copies belong alongside the layer's matrix, so the scheduler
        // keeps elementwise operations on that layer's device.
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_buffer_get_type(placement ? placement : source->buffer));
        if (!buffer) throw std::runtime_error("Dory: cannot allocate coefficients");
        buffers.push_back(buffer);
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        baked[source] = tensor;
        return tensor;
    }
};

void llama_model_dory::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_INPUT_LAYER_COUNT, n_input);
    ml.get_key(LLM_KV_RECURRENT_LAYER_COUNT, n_recurrent);
    ml.get_key(LLM_KV_OUTPUT_LAYER_COUNT, n_output);
    ml.get_key(LLM_KV_NUM_LOOPS, n_loops);
    ml.get_key(LLM_KV_RECURRENT_KV_CACHE_MODE, recurrent_kv_cache_mode, false);
    if (recurrent_kv_cache_mode != "per_loop" && recurrent_kv_cache_mode != "last_loop") {
        throw std::runtime_error("Dory recurrent_kv_cache_mode must be per_loop or last_loop");
    }
    ml.get_key(LLM_KV_LAYER_PATTERN, pattern);
    uint32_t n_profile = 0;
    ml.get_arr_n(LLM_KV_ROPE_PROFILE, n_profile);
    if (n_profile > LLAMA_MAX_LAYERS) {
        throw std::runtime_error("Dory RoPE profile is too large");
    }
    ml.get_arr(LLM_KV_ROPE_PROFILE, profile);
    ml.get_key(LLM_KV_INIT_STD, init_std);
    ml.get_key(LLM_KV_ALPHA_INIT, alpha_init);
    ml.get_key(LLM_KV_GATE_INIT, gate_init);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, theta_1);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_2, theta_2);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    const uint64_t n_physical = uint64_t(n_input) + n_recurrent + n_output;
    const uint64_t n_virtual = uint64_t(n_input) + uint64_t(n_recurrent)*n_loops + n_output;
    if (!n_input || !n_recurrent || !n_output || !n_loops || n_loops > 4 ||
            n_virtual > LLAMA_MAX_LAYERS || n_physical != hparams.n_layer_all ||
            pattern.size() != n_physical || profile.size() != n_physical ||
            pattern.find_first_not_of("S*-") != std::string::npos ||
            !(init_std > 0 && std::isfinite(init_std)) ||
            !(alpha_init > 0 && std::isfinite(alpha_init)) ||
            !(gate_init > 0 && std::isfinite(gate_init)) ||
            !(theta_1 > 0 && std::isfinite(theta_1)) || !(theta_2 > 0 && std::isfinite(theta_2)) ||
            !hparams.n_swa || pattern.find('S') == std::string::npos || pattern.find('*') == std::string::npos) {
        throw std::runtime_error("unsupported or invalid Dory recurrence metadata");
    }
    const auto nh = hparams.n_head(0);
    const auto nk = hparams.n_head_kv(0);
    if (!nh || !nk || nh % nk || !hparams.n_embd_head_k() ||
            hparams.n_embd_head_k() != hparams.n_embd_head_v() || hparams.n_rot() != hparams.n_embd_head_k()) {
        throw std::runtime_error("invalid Dory attention dimensions");
    }
    const auto nf = hparams.n_ff(0);
    if (!nf) {
        throw std::runtime_error("invalid Dory feed-forward dimension");
    }
    for (auto p : profile) {
        if (p < 1 || p > 2) {
            throw std::runtime_error("invalid Dory RoPE profile");
        }
    }
    for (uint32_t i = 0; i < n_input; ++i) {
        physical.push_back(i);
    }
    for (uint32_t k = 0; k < n_loops; ++k) {
        for (uint32_t i = 0; i < n_recurrent; ++i) {
            physical.push_back(n_input + i);
        }
    }
    for (uint32_t i = 0; i < n_output; ++i) {
        physical.push_back(n_input + n_recurrent + i);
    }
    hparams.n_layer_all = n_virtual;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = theta_2;
    for (uint32_t il = 0; il < n_virtual; ++il) {
        const uint32_t p = physical[il];
        const bool recurrent = p >= n_input && p < n_input + n_recurrent;
        cache_layer.push_back(recurrent && recurrent_kv_cache_mode == "last_loop" ? p : il);
        hparams.n_head_arr[il] = nh;
        hparams.n_head_kv_arr[il] = pattern[p] == '-' ? 0 : nk;
        hparams.n_ff_arr[il] = nf;
        hparams.is_swa_impl[il] = pattern[p] == 'S';
        hparams.rope_pattern[il] = profile[p] != 0 && pattern[p] != '-';
    }
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dory::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    logit_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_SCALE, "weight"), {n_vocab}, 0);
    scale.resize(n_layer);
    std::vector<int> first(pattern.size(), -1);
    for (int il = 0; il < n_layer; ++il) {
        const int p = physical[il];
        if (first[p] >= 0) {
            layers[il] = layers[first[p]];
            scale[il] = scale[first[p]];
            continue;
        }
        first[p] = il;
        auto & l = layers[il];
        auto & s = scale[il];
        s.alpha = create_tensor(tn(LLM_TENSOR_RESIDUAL_ALPHA, "weight", p), {n_embd}, 0);
        if (pattern[p] == '-') {
            l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", p), {n_embd, n_ff}, 0);
            l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", p), {n_ff, n_embd}, 0);
            s.suv_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE_SCALE, "weight", p), {n_ff}, 0);
            s.suv_up = create_tensor(tn(LLM_TENSOR_FFN_UP_SCALE, "weight", p), {n_ff}, 0);
        } else {
            const int64_t nq = n_head * n_embd_head_k;
            const int64_t nkv = hparams.n_head_kv(il) * n_embd_head_k;
            l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", p), {n_embd, nq}, 0);
            l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", p), {n_embd, nkv}, 0);
            l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", p), {n_embd, nkv}, 0);
            l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", p), {nq, n_embd}, 0);
            l.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", p), {n_embd, nq}, 0);
            s.sqk = create_tensor(tn(LLM_TENSOR_ATTN_QK_SCALE, "weight", p), {nq}, 0);
            s.gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE_SCALE, "weight", p), {nq}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dory::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dory::graph::graph(const llama_model_dory & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    ggml_tensor * pos = build_inp_pos();
    // A single requested token is already in output order. Keep the gather
    // for pipeline graphs and batches that may select a subset of rows.
    ggml_tensor * out_ids = n_tokens == 1 && n_outputs == 1 && !cparams.pipeline_parallel
        ? nullptr : build_inp_out_ids();
    auto * attn = build_attn_inp_kv_iswa();
    ggml_tensor * input_h = nullptr;
    const int recurrent_end = model.n_input + model.n_recurrent * model.n_loops;
    const bool trace_rounds = std::getenv("DORY_WORKBENCH_TRACE") != nullptr;
    const auto norm = [&](ggml_tensor * x) { return build_gdn_l2_norm(ctx0, x, 1e-6f); };
    auto frozen_ptr = std::make_unique<dory_frozen_scales>(ctx0);
    auto * frozen = frozen_ptr.get();
    res->add_input(std::move(frozen_ptr));
    // Weight-derived scales do not depend on tokens or recurrent-loop state.
    // Reuse their graph nodes when a physical layer is visited again.
    struct cached_scales { ggml_tensor * alpha = nullptr; ggml_tensor * g = nullptr; ggml_tensor * u = nullptr; ggml_tensor * q = nullptr; ggml_tensor * gate = nullptr; };
    std::vector<cached_scales> scales(model.pattern.size());

    for (int il = 0; il < n_layer; ++il) {
        const int p = model.physical[il];
        const auto & l = model.layers[il];
        const auto & s = model.scale[il];
        auto & cs = scales[p];
        if (!cs.alpha) {
            auto * placement = (model.pattern[p] == '-' ? l.ffn_down : l.wo)->buffer;
            cs.alpha = frozen->get(s.alpha, model.alpha_init / model.init_std, 2, placement);
            if (model.pattern[p] == '-') {
                cs.g = frozen->get(s.suv_gate, sqrtf(n_embd), 0, placement);
                cs.u = frozen->get(s.suv_up, sqrtf(n_embd), 0, placement);
            } else {
                cs.q = frozen->get(s.sqk, sqrtf(hparams.n_embd_head_k(il)) / model.init_std, 1, placement);
                cs.gate = frozen->get(s.gate, sqrtf(model.gate_init) / model.init_std, 1, placement);
            }
        }
        if (il == (int) model.n_input) {
            input_h = cur;
        }
        if (il >= (int) model.n_input && il < recurrent_end && (il - model.n_input) % model.n_recurrent == 0) {
            cur = norm(ggml_add(ctx0, cur, input_h));
        }
        ggml_tensor * residual = cur;
        if (model.pattern[p] == '-') {
            auto * g = build_lora_mm(l.ffn_gate, cur);
            auto * u = build_lora_mm(l.ffn_up, cur);
            g = ggml_mul(ctx0, g, cs.g);
            u = ggml_mul(ctx0, u, cs.u);
            // One fused activation/gating kernel instead of writing the SiLU intermediate.
            cur = norm(ggml_swiglu_split(ctx0, g, u));
            cur = build_lora_mm(l.ffn_down, cur);
        } else {
            const auto hd = hparams.n_embd_head_k(il);
            const auto nh = hparams.n_head(il);
            const auto nk = hparams.n_head_kv(il);
            auto [q, k, v] = build_qkv(l, cur, hd, nh, nk, il);
            if (model.profile[p] != 0) {
                const float theta = model.profile[p] == 2 ? model.theta_2 : model.theta_1;
                q = ggml_rope_ext(ctx0, q, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                k = ggml_rope_ext(ctx0, k, pos, nullptr, hd, rope_type, n_ctx_orig, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            auto * sq = cs.q;
            q = ggml_mul(ctx0, norm(q), ggml_reshape_3d(ctx0, sq, hd, nh, 1));
            k = norm(k);
            auto * gate = build_lora_mm(l.wqkv_gate, cur);
            auto * sg = cs.gate;
            gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, sg));
            // Repeated writes replace only the current chunk; older tokens retain final-loop KV.
            cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(hd), model.cache_layer[il]);
            cur = norm(ggml_mul(ctx0, cur, gate));
            cur = build_lora_mm(l.wo, cur);
        }
        auto * a = norm(residual);
        auto * b = norm(cur);
        auto * alpha = cs.alpha;
        cur = norm(ggml_add(ctx0, a, ggml_mul(ctx0, ggml_sub(ctx0, b, a), alpha)));
        const bool round_end = il >= (int) model.n_input && il < recurrent_end &&
                               (il + 1 - model.n_input) % model.n_recurrent == 0;
        if (round_end) {
            cur = norm(cur);
        }
        cb(cur, "l_out", il);
        if (trace_rounds && round_end) {
            ggml_format_name(cur, "dory_round:%u/%u", (il - model.n_input) / model.n_recurrent + 1, model.n_loops);
        }
    }
    if (out_ids) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    res->t_embd = cur;
    cb(cur, "result_norm", -1);
    cur = build_lora_mm(model.output, cur);
    cur = ggml_mul(ctx0, cur, frozen->get(model.logit_scale, 1.0f / model.init_std, 0, model.output->buffer));
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    // Preserve F32 inputs through recurrent normalization; BF16 and quantized weights keep their normal kernels.
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F32) {
            ggml_prec_set_src(node, GGML_PREC_F32, 1);
        }
    }
}
