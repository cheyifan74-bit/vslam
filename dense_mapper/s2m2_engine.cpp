#include "dense_mapper/s2m2_engine.h"

#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include "utils/print.h"

namespace vslam
{
namespace
{

class TrtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char *msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
    {
      PRINT_WARNING("[DENSE][TRT]: %s\n", msg);
    }
  }
};

void FreeCuda(void *ptr)
{
  if (ptr != nullptr)
  {
    cudaFree(ptr);
  }
}

} // namespace

struct S2M2Engine::Impl
{
  TrtLogger logger;
  nvinfer1::IRuntime *runtime = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;
  cudaStream_t stream = nullptr;
  void *left_device = nullptr;
  void *right_device = nullptr;
  void *disp_device = nullptr;
  void *occ_device = nullptr;
  void *conf_device = nullptr;
  bool input_is_float = true;
  size_t input_bytes = 0;
  size_t output_bytes = 0;

  ~Impl()
  {
    FreeCuda(left_device);
    FreeCuda(right_device);
    FreeCuda(disp_device);
    FreeCuda(occ_device);
    FreeCuda(conf_device);
    if (stream != nullptr)
    {
      cudaStreamDestroy(stream);
    }
    delete context;
    delete engine;
    delete runtime;
  }
};

S2M2Engine::S2M2Engine() = default;
S2M2Engine::~S2M2Engine() = default;

bool S2M2Engine::load(const std::string &engine_path, const int width, const int height)
{
  ready_ = false;
  impl_.reset();
  if (engine_path.empty() || width <= 0 || height <= 0)
  {
    return false;
  }
  std::ifstream file(engine_path, std::ios::binary);
  if (!file)
  {
    PRINT_WARNING("[DENSE]: S²M² engine not found at %s (dense fusion stays idle)\n",
                  engine_path.c_str());
    return false;
  }
  file.seekg(0, std::ios::end);
  const auto nbytes = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> blob(static_cast<size_t>(nbytes));
  file.read(blob.data(), nbytes);
  if (!file)
  {
    PRINT_ERROR("[DENSE]: failed to read engine %s\n", engine_path.c_str());
    return false;
  }

  auto impl = std::make_unique<Impl>();
  impl->runtime = nvinfer1::createInferRuntime(impl->logger);
  if (impl->runtime == nullptr)
  {
    PRINT_ERROR("[DENSE]: createInferRuntime failed\n");
    return false;
  }
  impl->engine = impl->runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (impl->engine == nullptr)
  {
    PRINT_ERROR("[DENSE]: deserializeCudaEngine failed. Rebuild the engine with this "
                "machine's TensorRT.\n");
    return false;
  }
  impl->context = impl->engine->createExecutionContext();
  if (impl->context == nullptr)
  {
    PRINT_ERROR("[DENSE]: createExecutionContext failed\n");
    return false;
  }

  auto has_tensor = [&](const char *name) {
    const int count = impl->engine->getNbIOTensors();
    for (int i = 0; i < count; ++i)
    {
      if (std::strcmp(impl->engine->getIOTensorName(i), name) == 0)
      {
        return true;
      }
    }
    return false;
  };
  const char *kLeft = "input_left";
  const char *kRight = "input_right";
  const char *kDisp = "output_disp";
  const char *kOcc = "output_occ";
  const char *kConf = "output_conf";
  if (!has_tensor(kLeft) || !has_tensor(kRight) || !has_tensor(kDisp) || !has_tensor(kOcc) ||
      !has_tensor(kConf))
  {
    PRINT_ERROR("[DENSE]: engine %s is not an S²M² engine "
                "(expected input_left/right and output_disp/occ/conf)\n",
                engine_path.c_str());
    return false;
  }
  const nvinfer1::Dims dims = impl->engine->getTensorShape(kLeft);
  if (dims.nbDims == 4 && dims.d[2] > 0 && dims.d[3] > 0 &&
      (dims.d[2] != height || dims.d[3] != width))
  {
    PRINT_ERROR("[DENSE]: engine %s is %dx%d but dense_rect is %dx%d\n", engine_path.c_str(),
                dims.d[3], dims.d[2], width, height);
    return false;
  }
  const nvinfer1::DataType input_type = impl->engine->getTensorDataType(kLeft);
  impl->input_is_float = input_type != nvinfer1::DataType::kUINT8;
  const int pixels = width * height;
  impl->input_bytes =
      static_cast<size_t>(pixels) * 3 * (impl->input_is_float ? sizeof(float) : sizeof(uint8_t));
  impl->output_bytes = static_cast<size_t>(pixels) * sizeof(float);
  auto alloc = [](size_t bytes, void **ptr) {
    return cudaMalloc(ptr, bytes) == cudaSuccess;
  };
  if (!alloc(impl->input_bytes, &impl->left_device) ||
      !alloc(impl->input_bytes, &impl->right_device) ||
      !alloc(impl->output_bytes, &impl->disp_device) ||
      !alloc(impl->output_bytes, &impl->occ_device) ||
      !alloc(impl->output_bytes, &impl->conf_device) ||
      cudaStreamCreate(&impl->stream) != cudaSuccess)
  {
    PRINT_ERROR("[DENSE]: CUDA allocation for the stereo engine failed\n");
    return false;
  }
  impl->context->setTensorAddress(kLeft, impl->left_device);
  impl->context->setTensorAddress(kRight, impl->right_device);
  impl->context->setTensorAddress(kDisp, impl->disp_device);
  impl->context->setTensorAddress(kOcc, impl->occ_device);
  impl->context->setTensorAddress(kConf, impl->conf_device);

  width_ = width;
  height_ = height;
  impl_ = std::move(impl);
  ready_ = true;
  PRINT_INFO("[DENSE]: loaded S²M² engine %s (%dx%d, %s input)\n", engine_path.c_str(), width,
             height, impl_->input_is_float ? "float" : "uint8");
  return true;
}

bool S2M2Engine::infer(const cv::Mat &left_rgb, const cv::Mat &right_rgb,
                       std::vector<float> *disparity, std::vector<float> *occlusion,
                       std::vector<float> *confidence)
{
  if (!ready_ || disparity == nullptr || occlusion == nullptr || confidence == nullptr ||
      left_rgb.cols != width_ || left_rgb.rows != height_ || left_rgb.type() != CV_8UC3 ||
      right_rgb.type() != CV_8UC3)
  {
    return false;
  }
  const int pixels = width_ * height_;
  std::vector<float> left_f;
  std::vector<float> right_f;
  std::vector<uint8_t> left_u8;
  std::vector<uint8_t> right_u8;
  auto pack = [&](const cv::Mat &rgb, std::vector<float> *as_float, std::vector<uint8_t> *as_u8) {
    if (impl_->input_is_float)
    {
      as_float->resize(static_cast<size_t>(pixels) * 3);
    }
    else
    {
      as_u8->resize(static_cast<size_t>(pixels) * 3);
    }
    for (int y = 0; y < height_; ++y)
    {
      const auto *row = rgb.ptr<cv::Vec3b>(y);
      for (int x = 0; x < width_; ++x)
      {
        const int base = (y * width_ + x);
        for (int c = 0; c < 3; ++c)
        {
          const uint8_t value = row[x][c];
          if (impl_->input_is_float)
          {
            (*as_float)[c * pixels + base] = static_cast<float>(value);
          }
          else
          {
            (*as_u8)[c * pixels + base] = value;
          }
        }
      }
    }
  };
  pack(left_rgb, &left_f, &left_u8);
  pack(right_rgb, &right_f, &right_u8);
  const void *left_host = impl_->input_is_float ? static_cast<const void *>(left_f.data())
                                                : static_cast<const void *>(left_u8.data());
  const void *right_host = impl_->input_is_float ? static_cast<const void *>(right_f.data())
                                                 : static_cast<const void *>(right_u8.data());
  if (cudaMemcpyAsync(impl_->left_device, left_host, impl_->input_bytes, cudaMemcpyHostToDevice,
                      impl_->stream) != cudaSuccess ||
      cudaMemcpyAsync(impl_->right_device, right_host, impl_->input_bytes, cudaMemcpyHostToDevice,
                      impl_->stream) != cudaSuccess)
  {
    PRINT_WARNING("[DENSE]: copy to S²M² inputs failed\n");
    return false;
  }
  if (!impl_->context->enqueueV3(impl_->stream))
  {
    PRINT_WARNING("[DENSE]: S²M² enqueueV3 failed\n");
    return false;
  }
  disparity->resize(static_cast<size_t>(pixels));
  occlusion->resize(static_cast<size_t>(pixels));
  confidence->resize(static_cast<size_t>(pixels));
  if (cudaMemcpyAsync(disparity->data(), impl_->disp_device, impl_->output_bytes,
                      cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess ||
      cudaMemcpyAsync(occlusion->data(), impl_->occ_device, impl_->output_bytes,
                      cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess ||
      cudaMemcpyAsync(confidence->data(), impl_->conf_device, impl_->output_bytes,
                      cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess)
  {
    PRINT_WARNING("[DENSE]: copy from S²M² outputs failed\n");
    return false;
  }
  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess)
  {
    PRINT_WARNING("[DENSE]: stereo engine synchronize failed\n");
    return false;
  }
  return true;
}

} // namespace vslam
