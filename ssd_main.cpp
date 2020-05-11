#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "BatchStream.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::vector;

const std::string gSampleName = "TensorRT.sample_ssd";
static samplesCommon::Args gArgs;

// Network details
const char* gNetworkName = "ssd";       // Network name
static const int kINPUT_C = 3;          // Input image channels
static const int kINPUT_H = 300;        // Input image height
static const int kINPUT_W = 300;        // Input image width
static const int kOUTPUT_CLS_SIZE = 21; // Number of classes
static const int kKEEP_TOPK = 200;      // Number of total bboxes to be kept per image after NMS step. It is same as detection_output_param.keep_top_k in prototxt file

enum MODE
{
    kFP32,
    kFP16,
    kINT8,
    kUNKNOWN
};

struct Param
{
    MODE modelType{MODE::kFP32}; // Default run FP32 precision
} params;

std::ostream& operator<<(std::ostream& o, MODE dt)
{
    switch (dt)
    {
        case kFP32: o << "FP32"; break;
        case kFP16: o << "FP16"; break;
        case kINT8: o << "INT8"; break;
        case kUNKNOWN: o << "UNKNOWN"; break;
    }
    return o;
}

// Data directory
static const std::vector<std::string> kDIRECTORIES{"/home/kun/TensorRT-5.1.5.0/data/samples/ssd/",
                                                   "/home/kun/TensorRT-5.1.5.0/data/ssd/",
                                                   "/home/kun/TensorRT-5.1.5.0/data/int8_samples/ssd/",
                                                   "/home/kun/TensorRT-5.1.5.0/int8/ssd/"};

const std::string gCLASSES[kOUTPUT_CLS_SIZE]{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}; // List of class labels

static const char* kINPUT_BLOB_NAME = "data";            // Input blob name
static const char* kOUTPUT_BLOB_NAME0 = "detection_out"; // Output blob name
static const char* kOUTPUT_BLOB_NAME1 = "keep_count";    // Output blob name

// INT8 calibration variables
static const int kCAL_BATCH_SIZE = 1;   // Batch size
static const int kFIRST_CAL_BATCH = 0;  // First batch
static const int kNB_CAL_BATCHES = 100; // Number of batches

#define CalibrationMode 1 //Set to '0' for Legacy calibrator and any other value for Entropy calibrator 2

// Visualization
const float kVISUAL_THRESHOLD = 0.6f;

class Int8LegacyCalibrator : public nvinfer1::IInt8LegacyCalibrator
{
public:
    Int8LegacyCalibrator(BatchStream& stream, int firstBatch, double cutoff, double quantile, const char* networkName, bool readCache = true)
            : mStream(stream)
            , mFirstBatch(firstBatch)
            , mReadCache(readCache)
            , mNetworkName(networkName)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        reset(cutoff, quantile);
    }

    virtual ~Int8LegacyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }
    double getQuantile() const override { return mQuantile; }
    double getRegressionCutoff() const override { return mCutoff; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;

        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

    const void* readHistogramCache(size_t& length) override
    {
        length = mHistogramCache.size();
        return length ? &mHistogramCache[0] : nullptr;
    }

    void writeHistogramCache(const void* cache, size_t length) override
    {
        mHistogramCache.clear();
        std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
    }

    void reset(double cutoff, double quantile)
    {
        mCutoff = cutoff;
        mQuantile = quantile;
        mStream.reset(mFirstBatch);
    }

private:
    std::string calibrationTableName()
    {
        assert(mNetworkName != nullptr);
        return std::string("CalibrationTable") + mNetworkName;
    }
    BatchStream mStream;
    int mFirstBatch;
    double mCutoff, mQuantile;
    bool mReadCache{true};
    const char* mNetworkName;
    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache, mHistogramCache;
};

class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(BatchStream& stream, int firstBatch, bool readCache = true)
            : mStream(stream)
            , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator2()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], kINPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable") + gNetworkName;
    }
    BatchStream mStream;
    size_t mInputCount;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

std::string locateFile(const std::string& input)
{
    return locateFile(input, kDIRECTORIES);
}

void caffeToTRTModel(const std::string& deployFile,           // Name for caffe prototxt
                     const std::string& modelFile,            // Name for model
                     const std::vector<std::string>& outputs, // Network outputs
                     unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                     MODE mode,                               // Precision mode
                     IHostMemory** trtModelStream)            // Output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    DataType dataType = DataType::kFLOAT;
    if (mode == kFP16)
        dataType = DataType::kHALF;
    gLogInfo << "Begin parsing model..." << std::endl;
    gLogInfo << mode << " mode running..." << std::endl;

    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              dataType);
    gLogInfo << "End parsing model..." << std::endl;

    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(36 << 20);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);
    builder->allowGPUFallback(true);
    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    ICudaEngine* engine;
    if (mode == kINT8)
    {
#if CalibrationMode == 0
        assert(args.useDLACore != -1 && "Legacy calibration mode not supported with DLA.");
        gLogInfo << "Using Legacy Calibrator" << std::endl;
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
        calibrator.reset(new Int8LegacyCalibrator(calibrationStream, 0, kCUTOFF, kQUANTILE, gNetworkName, true));
#else
        gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
        calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, kFIRST_CAL_BATCH));
#endif
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator.get());
    }
    else
    {
        builder->setFp16Mode(mode == kFP16);
    }
    gLogInfo << "Begin building engine..." << std::endl;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    gLogInfo << "End building engine..." << std::endl;

    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(kINPUT_BLOB_NAME),
            outputIndex0 = engine.getBindingIndex(kOUTPUT_BLOB_NAME0),
            outputIndex1 = engine.getBindingIndex(kOUTPUT_BLOB_NAME1);

    // Create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float))); // Data
    CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float)));               // Detection_out
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * sizeof(int)));                                  // KeepCount (BBoxs left for each batch)

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * kKEEP_TOPK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

void printHelp(const char* name)
{
    std::cout << "Usage: " << name << "\n"
              << "Optional Parameters:\n"
              << "  -h, --help        Display help information.\n"
              << "  --useDLACore=N    Specify the DLA engine to run on.\n"
              << "  --fp16            Specify to run in fp16 mode.\n"
              << "  --int8            Specify to run in int8 mode." << std::endl;
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help || !argsOK)
    {
        printHelp(argv[0]);
        return argsOK ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    params.modelType = kFP32;
    if (gArgs.runInFp16)
    {
        params.modelType = kFP16;
    }
    else if (gArgs.runInInt8)
    {
        params.modelType = kINT8;
    }


    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    IHostMemory* trtModelStream{nullptr};
    // Create a TensorRT model from the caffe model and serialize it to a stream

    const int N = 1; // Batch size

    caffeToTRTModel("../models/mobilenet-ssd/mobilenet-ssd.prototxt",
                    "../models/mobilenet-ssd/mobilenet-ssd.caffemodel",
                    std::vector<std::string>{kOUTPUT_BLOB_NAME0, kOUTPUT_BLOB_NAME1},
                    N, params.modelType, &trtModelStream);

    gLogInfo << "*** deserializing" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    // Deserialize the engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    trtModelStream->destroy();

    float pixelMean[3]{104.0f, 117.0f, 123.0f};
    float* data = new float[N * kINPUT_C * kINPUT_H * kINPUT_W];
    // Host memory for outputs
    float* detectionOut = new float[N * kKEEP_TOPK * 7];
    int* keepCount = new int[N];

    cv::VideoCapture cap("/home/kun/dataset/videos/ADAS_Video-20191120-151849.mp4");

    cv::Mat frame;

    bool pass = true;
    int wait_t = 1;
    while (cap.isOpened()){
        auto start_t = cv::getTickCount();
        cap >> frame;
        if(frame.empty())
            break;
        cv::resize(frame, frame, cv::Size(kINPUT_W, kINPUT_H));
        for(int c=0; c<kINPUT_C; c++){
            for(int h=0; h<kINPUT_H; h++){
                uchar* p = frame.ptr<uchar>(h);
                for(int w=0; w<kINPUT_W; w++){
//                    data[c*kINPUT_H*kINPUT_W+h*kINPUT_W+w] = float(p[w*kINPUT_C+c]) - pixelMean[c];
                    data[c*kINPUT_H*kINPUT_W+h*kINPUT_W+w] = float(p[w*kINPUT_C+c] - 127.5) / 127.5;
                }
            }
        }

        doInference(*context, data, detectionOut, keepCount, N);

        for (int p = 0; p < N; ++p)
        {
            int numDetections = 0;
            // is there at least one correct detection?
            bool correctDetection = false;
            for (int i = 0; i < keepCount[p]; ++i)
            {
                float* det = detectionOut + (p * kKEEP_TOPK + i) * 7;
                if (det[2] < kVISUAL_THRESHOLD)
                    continue;
                assert((int) det[1] < kOUTPUT_CLS_SIZE);

                numDetections++;
                if (gCLASSES[(int) det[1]] == "car")
                    correctDetection = true;
                auto xmin = int(det[3] * kINPUT_W);
                auto ymin = int(det[4] * kINPUT_H);
                auto xmax = int(det[5] * kINPUT_W);
                auto ymax = int(det[6] * kINPUT_H);
                cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0,0,255), 2, 8);
            }
            pass &= numDetections >= 1;
            pass &= correctDetection;
        }
        gLogInfo << "detection time:  " << double(cv::getTickCount()-start_t)/cv::getTickFrequency() << std::endl;
        cv::imshow("result", frame);
        int key = cv::waitKey(wait_t);
        if(27 == key)
            break;
        else if(32 == key)
            wait_t = 1 - wait_t;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    delete[] data;
    delete[] detectionOut;
    delete[] keepCount;

    // Note: Once you call shutdownProtobufLibrary, you cannot use the parsers anymore.
    shutdownProtobufLibrary();

    return gLogger.reportTest(sampleTest, pass);
}