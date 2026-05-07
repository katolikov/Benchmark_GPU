// MNN benchmark binary using the modern Express::Module API (matches the
// MNN Python pkg behavior). Loads .mnn, runs inference repeatedly with the
// selected forward type, prints JSON timing, and optionally dumps the output
// tensor for cross-framework comparison.
//
// Backends: cpu, opencl, vulkan
// Variants for OpenCL: "buffer" (default) or "image" (memory mode)
// Variants for CPU/Vulkan: "fp32" / "fp16"
#include "common.h"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>

#include <cstdio>
#include <cstring>
#include <memory>

using namespace bench;
using namespace MNN::Express;

int main(int argc, char **argv) {
    Args a;
    parse_args(argc, argv, a);

    // ---- Forward type ---------------------------------------------------
    MNNForwardType ft = MNN_FORWARD_CPU;
    if      (a.backend == "cpu")    ft = MNN_FORWARD_CPU;
    else if (a.backend == "opencl") ft = MNN_FORWARD_OPENCL;
    else if (a.backend == "vulkan") ft = MNN_FORWARD_VULKAN;
    else { std::fprintf(stderr, "MNN: unknown backend '%s'\n", a.backend.c_str()); return 2; }

    MNN::ScheduleConfig sched;
    sched.type = ft;
    sched.numThread = a.threads;
    if (ft == MNN_FORWARD_OPENCL) {
        unsigned int mode = MNN_GPU_TUNING_FAST;
        if (a.variant == "image") mode |= MNN_GPU_MEMORY_IMAGE;
        else                      mode |= MNN_GPU_MEMORY_BUFFER;
        sched.mode = mode;
    } else if (ft == MNN_FORWARD_VULKAN) {
        sched.mode = MNN_GPU_TUNING_FAST;
    }

    MNN::BackendConfig bcfg;
    if (a.variant == "fp32") bcfg.precision = MNN::BackendConfig::Precision_Normal;
    else                     bcfg.precision = MNN::BackendConfig::Precision_Low; // fp16
    bcfg.power = MNN::BackendConfig::Power_High;
    sched.backendConfig = &bcfg;

    // ---- Load module via Module API (matches MNN.nn.load_module_from_file) ---
    double t0 = now_ms();
    std::shared_ptr<Executor::RuntimeManager> rtmgr(
        Executor::RuntimeManager::createRuntimeManager(sched));
    if (!rtmgr) { std::fprintf(stderr, "MNN: createRuntimeManager failed\n"); return 1; }
    rtmgr->setCache(a.model + ".cache");

    Module::Config mcfg;
    mcfg.shapeMutable = false;
    mcfg.rearrange    = true;       // pre-rearrange GPU weights at load time

    std::shared_ptr<Module> mod(
        Module::load({}, {}, a.model.c_str(), rtmgr, &mcfg));
    if (!mod) { std::fprintf(stderr, "MNN: Module::load failed\n"); return 1; }
    double t_create = now_ms() - t0;

    auto info = mod->getInfo();
    if (!info || info->inputs.empty() || info->outputNames.empty()) {
        std::fprintf(stderr, "MNN: module has no inputs/outputs\n"); return 1;
    }
    const auto &in_info = info->inputs[0];

    // ---- Build input VARP and feed bytes --------------------------------
    int in_count = 1;
    for (int d : in_info.dim) in_count *= (d > 0 ? d : 1);

    std::vector<float> in_data;
    if (!a.input.empty()) {
        auto in_bytes = read_file(a.input);
        if (in_bytes.size() != (size_t)in_count * sizeof(float)) {
            std::fprintf(stderr, "MNN: input %zu B != tensor %d fp32 (%zu B)\n",
                         in_bytes.size(), in_count, (size_t)in_count * sizeof(float));
            return 1;
        }
        in_data.assign((const float *)in_bytes.data(),
                       (const float *)in_bytes.data() + in_count);
    } else {
        in_data = generate_input_fp32(a.seed, (size_t)in_count);
    }

    VARP in_var = _Input(in_info.dim, in_info.order, halide_type_of<float>());
    auto *dst = in_var->writeMap<float>();
    std::memcpy(dst, in_data.data(), (size_t)in_count * sizeof(float));
    in_var->unMap();

    // ---- First inference (cold; GPU compiles kernels here) -------------
    double tf0 = now_ms();
    auto out_vars = mod->onForward({in_var});
    if (out_vars.empty()) { std::fprintf(stderr, "MNN: forward returned empty\n"); return 1; }
    auto *first_ptr = out_vars[0]->readMap<float>(); // forces eval
    (void)first_ptr;
    out_vars[0]->unMap();
    double t_first = now_ms() - tf0;

    std::vector<double> warm_v;
    warm_v.reserve(a.warmup);
    for (int i = 0; i < a.warmup; ++i) {
        double s = now_ms();
        auto ov = mod->onForward({in_var});
        const float *p = ov[0]->readMap<float>();
        (void)p;
        ov[0]->unMap();
        warm_v.push_back(now_ms() - s);
    }

    std::vector<double> run_v;
    run_v.reserve(a.runs);
    std::vector<float> last_out;
    int out_count = 0;
    for (int i = 0; i < a.runs; ++i) {
        double s = now_ms();
        auto ov = mod->onForward({in_var});
        const float *p = ov[0]->readMap<float>();
        if (i == a.runs - 1) {
            const auto &shp = ov[0]->getInfo()->dim;
            int n = 1;
            for (int d : shp) n *= (d > 0 ? d : 1);
            out_count = n;
            last_out.assign(p, p + n);
        }
        ov[0]->unMap();
        run_v.push_back(now_ms() - s);
    }

    if (!a.output_bin.empty()) {
        write_file(a.output_bin, last_out.data(), last_out.size() * sizeof(float));
    }

    CmpResult cmp;
    bool have_cmp = false;
    if (!a.ref_bin.empty()) {
        auto rb = read_file(a.ref_bin);
        if (rb.size() == (size_t)out_count * sizeof(float)) {
            cmp = compare_fp32(last_out.data(),
                               reinterpret_cast<const float *>(rb.data()),
                               (size_t)out_count);
            have_cmp = true;
        } else {
            std::fprintf(stderr, "MNN: ref %zu != out %zu\n",
                         rb.size(), (size_t)out_count * sizeof(float));
        }
    }

    Stats s_warm = compute_stats(warm_v);
    Stats s_run  = compute_stats(run_v);
    emit_json("mnn", a, t_create, t_first, s_warm, s_run,
              have_cmp ? &cmp : nullptr, (size_t)out_count);

    rtmgr->updateCache();
    return 0;
}
