#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── COCO-17 skeleton topology ───────────────────────────────────────────────
static const int EDGES[][2] = {
    {0,1},{0,2},{1,3},{2,4},
    {5,6},{5,7},{7,9},{6,8},{8,10},
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};
static constexpr int NEDGE = (int)(sizeof(EDGES)/sizeof(EDGES[0]));
static constexpr int NKPT  = 17;

// ── Per-person decoded pose ─────────────────────────────────────────────────
struct DetPose {
    float x1, y1, x2, y2, score;
    float kx[NKPT], ky[NKPT], kc[NKPT];
};

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[pose_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // ── Find the [56, 8400] pose tensor ─────────────────────────────────────
    std::vector<TensorRef> ts = collect_tensors(in);
    const TensorRef *pt = find_tensor(ts, 56, 8400);
    if (!pt) {
        // No pose tensor this frame — just pass through
        return true;
    }
    const float *raw = pt->data;   // shape [56, 8400], row-major

    // ── Letterbox parameters (computed at runtime from actual surface) ───────
    const int NET = 640;
    const float scale = std::min<float>((float)NET / (float)W, (float)NET / (float)H);
    const int nw = (int)std::round((float)W * scale);
    const int nh = (int)std::round((float)H * scale);
    const float pad_x = (float)((NET - nw) / 2);
    const float pad_y = (float)((NET - nh) / 2);
    const float inv_scale = 1.0f / scale;

    const int C = 56;
    const int N = 8400;
    const float SCORE_THR = 0.25f;
    const float KPT_THR   = 0.30f;
    const float NMS_THR   = 0.45f;

    // ── Decode candidates ───────────────────────────────────────────────────
    // Tensor layout: row c, col i -> raw[c * N + i]
    // Row 0=cx, 1=cy, 2=w, 3=h, 4=score, 5..55 = 17*(kx,ky,kconf)
    std::vector<DetPose> dets;
    dets.reserve(256);

    for (int i = 0; i < N; ++i) {
        float sc = raw[4 * N + i];
        if (sc < SCORE_THR) continue;

        DetPose d;
        d.score = sc;

        float cx = raw[0 * N + i];
        float cy = raw[1 * N + i];
        float bw = raw[2 * N + i];
        float bh = raw[3 * N + i];

        float fx = (cx - pad_x) * inv_scale;
        float fy = (cy - pad_y) * inv_scale;
        float fw = bw * inv_scale;
        float fh = bh * inv_scale;

        d.x1 = fx - fw * 0.5f;
        d.y1 = fy - fh * 0.5f;
        d.x2 = fx + fw * 0.5f;
        d.y2 = fy + fh * 0.5f;

        for (int k = 0; k < NKPT; ++k) {
            int base = (5 + k * 3);
            float kx_raw = raw[base * N + i];
            float ky_raw = raw[(base + 1) * N + i];
            float kc_raw = raw[(base + 2) * N + i];
            d.kx[k] = (kx_raw - pad_x) * inv_scale;
            d.ky[k] = (ky_raw - pad_y) * inv_scale;
            d.kc[k] = kc_raw;
        }
        dets.push_back(d);
    }

    // ── NMS (greedy, score-descending, single class) ────────────────────────
    // Sort by score descending
    std::sort(dets.begin(), dets.end(), [](const DetPose &a, const DetPose &b) {
        return a.score > b.score;
    });

    std::vector<DetPose> kept;
    kept.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) {
        const DetPose &d = dets[i];
        bool suppress = false;
        for (size_t j = 0; j < kept.size(); ++j) {
            const DetPose &r = kept[j];
            float ix1 = std::max(d.x1, r.x1);
            float iy1 = std::max(d.y1, r.y1);
            float ix2 = std::min(d.x2, r.x2);
            float iy2 = std::min(d.y2, r.y2);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float ua = (d.x2 - d.x1) * (d.y2 - d.y1) +
                        (r.x2 - r.x1) * (r.y2 - r.y1) - inter;
            if (ua > 0.0f && (inter / ua) > NMS_THR) {
                suppress = true;
                break;
            }
        }
        if (!suppress) kept.push_back(d);
    }

    // ── Draw skeletons on frame ─────────────────────────────────────────────
    cv::Mat frame = read_nv12_bgr(in);

    render_meta::Builder sb(W, H);

    for (size_t p = 0; p < kept.size(); ++p) {
        const DetPose &d = kept[p];

        render_meta::Skeleton sk(NKPT);

        for (int k = 0; k < NKPT; ++k) {
            if (d.kc[k] >= KPT_THR) {
                sk.set((size_t)k, (double)d.kx[k], (double)d.ky[k]);
                int px = std::min(std::max((int)d.kx[k], 0), W - 1);
                int py = std::min(std::max((int)d.ky[k], 0), H - 1);
                cv::circle(frame, cv::Point(px, py), 4, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
            }
        }

        for (int e = 0; e < NEDGE; ++e) {
            int a = EDGES[e][0], b = EDGES[e][1];
            sk.link((size_t)a, (size_t)b);
            if (d.kc[a] >= KPT_THR && d.kc[b] >= KPT_THR) {
                int ax = std::min(std::max((int)d.kx[a], 0), W - 1);
                int ay = std::min(std::max((int)d.ky[a], 0), H - 1);
                int bx = std::min(std::max((int)d.kx[b], 0), W - 1);
                int by = std::min(std::max((int)d.ky[b], 0), H - 1);
                cv::line(frame, cv::Point(ax, ay), cv::Point(bx, by),
                         cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
        }

        sb.skeleton(sk, "person", -1, (double)d.score);
    }

    write_nv12(frame, in);
    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);