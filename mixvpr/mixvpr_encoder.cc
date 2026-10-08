// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "mixvpr/mixvpr_encoder.h"

#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VSLAM_MIXVPR_CUDA_ENABLED
#include "mixvpr/mixvpr_preprocess.h"

#include <cuda_runtime.h>
#endif

#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
#include <NvInfer.h>
#include <NvInferRuntime.h>
#endif

namespace colmap {
namespace {

int ParseGpuIndex(const std::string& gpu_index) {
  if (gpu_index.empty() || gpu_index == "-1") {
    return 0;
  }
  return std::stoi(gpu_index);
}

void L2NormalizeDescriptor(Eigen::VectorXf* desc) {
  if (desc == nullptr || desc->size() == 0) {
    return;
  }
  const float n = desc->norm();
  if (n > 1e-12f) {
    *desc /= n;
  }
}

#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
class MixVprTrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      LOG(WARNING) << "TensorRT: " << msg;
    }
  }
};
#endif

}  // namespace

class MixVprEncoder::Impl {
 public:
  Impl(const std::string& model_path,
       bool use_gpu,
       const std::string& gpu_index)
      : used_gpu_(false), backend_("none") {
#ifndef VSLAM_MIXVPR_TENSORRT_ENABLED
    (void)model_path;
    (void)use_gpu;
    (void)gpu_index;
    throw std::runtime_error(
        "vslam_mixvpr was built without TensorRT; cannot run MixVPR");
#else
    if (!use_gpu) {
      throw std::runtime_error("MixVPR requires GPU (TensorRT)");
    }
    LoadTensorRt(model_path, ParseGpuIndex(gpu_index));
#endif
  }

  ~Impl() {
#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
    context_.reset();
    engine_.reset();
    runtime_.reset();
#endif
    ReleaseCuda();
  }

  bool UsedGpu() const { return used_gpu_; }
  int DescDim() const { return desc_dim_; }
  const std::string& Backend() const { return backend_; }

  bool Encode(const Bitmap& bitmap, Eigen::VectorXf* desc) {
    if (desc == nullptr || bitmap.IsEmpty()) {
      return false;
    }
#ifdef VSLAM_MIXVPR_CUDA_ENABLED
    if (used_gpu_ && d_nchw_ != nullptr && EncodeGpu(bitmap, desc)) {
      L2NormalizeDescriptor(desc);
      return true;
    }
#endif
    (void)bitmap;
    (void)desc;
    return false;
  }

 private:
#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
  void LoadTensorRt(const std::string& engine_path, int device_id) {
    LOG(INFO) << "MixVPR: loading TensorRT engine " << engine_path;
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
      throw std::runtime_error("Cannot open TensorRT engine " + engine_path);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<size_t>(size));
    if (!file.read(blob.data(), size)) {
      throw std::runtime_error("Cannot read TensorRT engine " + engine_path);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
    if (!runtime_) {
      throw std::runtime_error("createInferRuntime failed");
    }
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      throw std::runtime_error("deserializeCudaEngine failed");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("createExecutionContext failed");
    }

    const int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
      const char* name = engine_->getIOTensorName(i);
      const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        trt_input_name_ = name;
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        trt_output_name_ = name;
        const nvinfer1::Dims dims = engine_->getTensorShape(name);
        if (dims.nbDims > 0) {
          desc_dim_ = static_cast<int>(dims.d[dims.nbDims - 1]);
        }
      }
    }
    if (trt_input_name_.empty() || trt_output_name_.empty()) {
      throw std::runtime_error("TensorRT engine missing MixVPR IO tensors");
    }
    nvinfer1::Dims in_dims = engine_->getTensorShape(trt_input_name_.c_str());
    if (in_dims.nbDims != 4) {
      throw std::runtime_error("MixVPR TensorRT input must be NCHW 4D");
    }
    in_dims.d[0] = 1;
    in_dims.d[1] = 3;
    in_dims.d[2] = kMixVprInputSize;
    in_dims.d[3] = kMixVprInputSize;
    if (!context_->setInputShape(trt_input_name_.c_str(), in_dims)) {
      throw std::runtime_error("TensorRT setInputShape failed");
    }
    const nvinfer1::Dims out_dims =
        context_->getTensorShape(trt_output_name_.c_str());
    if (out_dims.nbDims > 0 && out_dims.d[out_dims.nbDims - 1] > 0) {
      desc_dim_ = static_cast<int>(out_dims.d[out_dims.nbDims - 1]);
    }
    AllocCuda(device_id);
    if (!context_->setTensorAddress(trt_input_name_.c_str(), d_nchw_) ||
        !context_->setTensorAddress(trt_output_name_.c_str(), d_out_)) {
      throw std::runtime_error("TensorRT setTensorAddress failed");
    }
    used_gpu_ = true;
    backend_ = "tensorrt_fp16";
    LOG(INFO) << "MixVPR: TensorRT ready input=" << trt_input_name_
              << " output=" << trt_output_name_ << " dim=" << desc_dim_;
  }
#endif

#ifdef VSLAM_MIXVPR_CUDA_ENABLED
  void AllocCuda(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    if (stream_ == nullptr) {
      err = cudaStreamCreate(&stream_);
      if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }
    const size_t nchw_bytes =
        static_cast<size_t>(3 * kMixVprInputSize * kMixVprInputSize) *
        sizeof(float);
    const size_t out_bytes = static_cast<size_t>(kMixVprDescDim) * sizeof(float);
    d_src_bytes_ = 8ull * 1024ull * 1024ull;
    err = cudaMalloc(&d_src_, d_src_bytes_);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_nchw_, nchw_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaMalloc(&d_out_, out_bytes);
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }
  }

  bool EncodeGpu(const Bitmap& bitmap, Eigen::VectorXf* desc) {
    const size_t src_bytes = static_cast<size_t>(bitmap.Pitch()) *
                             static_cast<size_t>(bitmap.Height());
    if (src_bytes > d_src_bytes_) {
      cudaFree(d_src_);
      d_src_ = nullptr;
      d_src_bytes_ = src_bytes;
      if (cudaMalloc(&d_src_, d_src_bytes_) != cudaSuccess) {
        return false;
      }
    }
    if (!MixVprPreprocessCuda(bitmap.RowMajorData().data(),
                              bitmap.Width(),
                              bitmap.Height(),
                              bitmap.Pitch(),
                              bitmap.Channels(),
                              d_src_,
                              d_src_bytes_,
                              d_nchw_,
                              kMixVprInputSize,
                              stream_)) {
      return false;
    }

#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
    if (context_ == nullptr) {
      return false;
    }
    if (!context_->enqueueV3(stream_)) {
      LOG(ERROR) << "MixVPR TensorRT enqueueV3 failed";
      return false;
    }
    desc->resize(desc_dim_);
    if (cudaMemcpyAsync(desc->data(),
                        d_out_,
                        static_cast<size_t>(desc_dim_) * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream_) != cudaSuccess) {
      return false;
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      return false;
    }
    return desc->size() == kMixVprDescDim;
#else
    (void)desc;
    return false;
#endif
  }
#endif  // VSLAM_MIXVPR_CUDA_ENABLED

  void ReleaseCuda() {
#ifdef VSLAM_MIXVPR_CUDA_ENABLED
    if (d_src_ != nullptr) {
      cudaFree(d_src_);
      d_src_ = nullptr;
    }
    if (d_nchw_ != nullptr) {
      cudaFree(d_nchw_);
      d_nchw_ = nullptr;
    }
    if (d_out_ != nullptr) {
      cudaFree(d_out_);
      d_out_ = nullptr;
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
#endif
  }

  bool used_gpu_ = false;
  int desc_dim_ = kMixVprDescDim;
  std::string backend_ = "none";
#ifdef VSLAM_MIXVPR_TENSORRT_ENABLED
  MixVprTrtLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string trt_input_name_;
  std::string trt_output_name_;
#endif
#ifdef VSLAM_MIXVPR_CUDA_ENABLED
  cudaStream_t stream_ = nullptr;
  uint8_t* d_src_ = nullptr;
  size_t d_src_bytes_ = 0;
  float* d_nchw_ = nullptr;
  float* d_out_ = nullptr;
#endif
};

MixVprEncoder::MixVprEncoder(const std::string& model_path,
                             bool use_gpu,
                             const std::string& gpu_index)
    : impl_(std::make_unique<Impl>(model_path, use_gpu, gpu_index)) {}

MixVprEncoder::~MixVprEncoder() = default;

bool MixVprEncoder::Encode(const std::string& image_abs_path,
                           Eigen::VectorXf* desc) {
  Bitmap bitmap;
  if (!bitmap.Read(image_abs_path, /*as_rgb=*/true)) {
    LOG(WARNING) << "MixVPR: failed to read image " << image_abs_path;
    return false;
  }
  return Encode(bitmap, desc);
}

bool MixVprEncoder::Encode(const Bitmap& bitmap, Eigen::VectorXf* desc) {
  return impl_->Encode(bitmap, desc);
}

bool MixVprEncoder::UsedGpu() const { return impl_->UsedGpu(); }

int MixVprEncoder::DescDim() const { return impl_->DescDim(); }

const std::string& MixVprEncoder::Backend() const { return impl_->Backend(); }

std::vector<std::pair<image_t, float>> RankMixVprMatches(
    const Eigen::VectorXf& query,
    const std::vector<std::pair<image_t, Eigen::VectorXf>>& entries,
    const std::function<bool(image_t)>& usable,
    const int knn) {
  std::vector<std::pair<image_t, float>> scored;
  if (query.size() == 0 || entries.empty()) {
    return scored;
  }
  scored.reserve(entries.size());
  for (const auto& [id, desc] : entries) {
    if (usable && !usable(id)) {
      continue;
    }
    if (desc.size() != query.size()) {
      continue;
    }
    scored.emplace_back(id, query.dot(desc));
  }
  std::sort(scored.begin(),
            scored.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) {
                return a.second > b.second;
              }
              return a.first < b.first;
            });
  if (knn > 0 && static_cast<int>(scored.size()) > knn) {
    scored.resize(static_cast<size_t>(knn));
  }
  return scored;
}

}  // namespace colmap
