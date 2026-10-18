// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLOv8n-det] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // YOLOv8 output shape after batch strip: [84, 6300]
    // inferDims: numDims=2, d[0]=84, d[1]=6300
    // Layout is [4+numClasses, numProposals] — need to transpose to iterate
    // proposals. Rows 0-3 = cx, cy, w, h in pixel coords of network input.
    // Rows 4..83 = class scores (post-sigmoid, already probabilities).

    const unsigned int dimFeatures = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 6300

    if (dimFeatures < 4u + numClasses) {
        fprintf(stderr, "[YOLOv8n-det] unexpected feature dim %u vs 4+%u\n",
                dimFeatures, numClasses);
        return false;
    }

    for (unsigned int i = 0; i < numProposals; ++i) {
        // Find best class score for proposal i.
        // data layout: data[featureIdx * numProposals + proposalIdx]
        float maxScore = 0.0f;
        unsigned int bestClassId = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = data[(4u + c) * numProposals + i];
            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }

        // Per-class threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.3f;
        if (maxScore < thr) continue;

        // cx, cy, w, h in network-input pixel coordinates
        float cx = data[0u * numProposals + i];
        float cy = data[1u * numProposals + i];
        float bw = data[2u * numProposals + i];
        float bh = data[3u * numProposals + i];

        // Convert centre+size to top-left corner
        float left = cx - bw * 0.5f;
        float top  = cy - bh * 0.5f;

        // Clamp to network input
        left = std::max(0.0f, std::min(left, netW));
        top  = std::max(0.0f, std::min(top,  netH));
        bw   = std::max(0.0f, std::min(bw, netW - left));
        bh   = std::max(0.0f, std::min(bh, netH - top));

        // Drop degenerate boxes
        if (bw <= 1.0f || bh <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.left = left;
        obj.top = top;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}