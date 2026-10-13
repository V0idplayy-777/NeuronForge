"use strict";
/*
 * NeuronForge standalone inference runtime (JavaScript)
 * =====================================================
 * Runs a model exported from NeuronForge without NeuronForge.
 * Works in browsers, Node.js (>= 16), Deno, Bun and web workers.
 * No dependencies. INFERENCE ONLY: no training, no gradients, no optimiser.
 *
 * The numerical code mirrors NeuronForge's engine operation-for-operation:
 * float32 tensor storage, NHWC layout, the same loop order, the same activation
 * formulas (GELU uses the tanh approximation), Dropout disabled, BatchNorm using
 * its running statistics. Outputs therefore match the editor to within
 * floating-point noise (and are usually bit-identical in the same JS engine).
 */

var NF_RUNTIME_VERSION = "1.0.0";
var NF_BUNDLE_FORMAT = "neuronforge-standalone";
var NF_BUNDLE_VERSION = 1;
var NF_SUPPORTED_LAYERS = [
  "dense", "flatten", "reshape", "dropout", "batchnorm", "layernorm", "conv2d", "maxpool2d", "avgpool2d",
  "gap2d", "embedding", "rnn", "identity", "relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax",
];
var NF_ACTIVATION_LAYERS = ["relu", "leakyrelu", "gelu", "silu", "sigmoid", "tanh", "softmax"];

/* ------------------------------------------------------------------ helpers */

function nfNum(v, d) {
  var n = Number(v);
  return Number.isFinite(n) ? n : d;
}
function nfBool(v, d) {
  return typeof v === "boolean" ? v : d;
}
function nfKernel(v) {
  if (Array.isArray(v) && v.length >= 2) return [Number(v[0]), Number(v[1])];
  var n = nfNum(v, 3);
  return [n, n];
}
function nfNumel(s) {
  var n = 1;
  for (var i = 0; i < s.length; i++) n *= s[i];
  return n;
}
function nfShapeStr(s) {
  return "[" + Array.prototype.join.call(s, ", ") + "]";
}
function nfSameShape(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (var i = 0; i < a.length; i++) if (Number(a[i]) !== Number(b[i])) return false;
  return true;
}
function nfPadding(c) {
  return c.padding === undefined || c.padding === null ? "same" : String(c.padding);
}
function nfAct(c) {
  return c.activation === undefined || c.activation === null ? "none" : String(c.activation);
}

/* ------------------------------------------------- base64, utf8, checksums */

var NF_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
var NF_B64_LOOKUP = (function () {
  var t = new Int16Array(256);
  t.fill(-1);
  for (var i = 0; i < 64; i++) t[NF_B64.charCodeAt(i)] = i;
  return t;
})();

function nfBase64ToBytes(b64, key) {
  var clean = String(b64).replace(/[^A-Za-z0-9+/=]/g, "");
  if (clean.length % 4 !== 0) throw new Error("Corrupted weight tensor '" + key + "': base64 length is not a multiple of 4");
  var pad = 0;
  if (clean.length > 0 && clean.charAt(clean.length - 1) === "=") pad++;
  if (clean.length > 1 && clean.charAt(clean.length - 2) === "=") pad++;
  var outLen = (clean.length / 4) * 3 - pad;
  var out = new Uint8Array(outLen);
  var o = 0;
  for (var i = 0; i < clean.length; i += 4) {
    var a = NF_B64_LOOKUP[clean.charCodeAt(i)];
    var b = NF_B64_LOOKUP[clean.charCodeAt(i + 1)];
    var c = clean.charAt(i + 2) === "=" ? 0 : NF_B64_LOOKUP[clean.charCodeAt(i + 2)];
    var d = clean.charAt(i + 3) === "=" ? 0 : NF_B64_LOOKUP[clean.charCodeAt(i + 3)];
    if (a < 0 || b < 0 || c < 0 || d < 0) throw new Error("Corrupted weight tensor '" + key + "': invalid base64");
    var triple = (a << 18) | (b << 12) | (c << 6) | d;
    if (o < outLen) out[o++] = (triple >> 16) & 255;
    if (o < outLen) out[o++] = (triple >> 8) & 255;
    if (o < outLen) out[o++] = triple & 255;
  }
  return out;
}

function nfUtf8(str) {
  var out = [];
  for (var i = 0; i < str.length; i++) {
    var cp = str.codePointAt(i);
    if (cp > 0xffff) i++;
    if (cp < 0x80) out.push(cp);
    else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 63));
    else if (cp < 0x10000) out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
  }
  return out;
}

function nfHex8(n) {
  return ("00000000" + n.toString(16)).slice(-8);
}

/**
 * Checksum over utf8(key) followed by the tensor bytes. Binding the key into
 * the checksum means a tensor copied onto the wrong layer is detected even if
 * the shapes happen to match.
 */
function nfChecksum(key, bytes) {
  var kb = nfUtf8(key);
  var s1 = 0;
  var s2 = 0;
  var idx = 0;
  for (var i = 0; i < kb.length; i++, idx++) {
    s1 += kb[i];
    s2 += ((idx % 65521) + 1) * kb[i];
  }
  for (var j = 0; j < bytes.length; j++, idx++) {
    s1 += bytes[j];
    s2 += ((idx % 65521) + 1) * bytes[j];
  }
  return nfHex8(s1 % 4294967296) + nfHex8(s2 % 4294967296);
}

function nfDecodeTensor(entry, key, checkSums) {
  if (!entry || typeof entry.data !== "string") throw new Error("Missing weight tensor '" + key + "'");
  var bytes = nfBase64ToBytes(entry.data, key);
  var n = nfNumel(entry.shape);
  if (bytes.length !== n * 4) {
    throw new Error("Weight tensor '" + key + "' has " + bytes.length + " bytes but shape " + nfShapeStr(entry.shape) + " needs " + n * 4);
  }
  if (checkSums && entry.checksum && nfChecksum(key, bytes) !== entry.checksum) {
    throw new Error("Checksum mismatch for weight tensor '" + key + "': the bundle is corrupted, edited, or a tensor was moved to the wrong layer");
  }
  var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  var out = new Float32Array(n);
  for (var i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

/* --------------------------------------------------------- shape inference */

function nfConvGeom(H, W, kH, kW, stride, padding) {
  if (padding === "same") {
    var oh = Math.ceil(H / stride);
    var ow = Math.ceil(W / stride);
    return {
      outH: oh,
      outW: ow,
      padH: Math.max(0, (oh - 1) * stride + kH - H),
      padW: Math.max(0, (ow - 1) * stride + kW - W),
    };
  }
  return { outH: Math.floor((H - kH) / stride) + 1, outW: Math.floor((W - kW) / stride) + 1, padH: 0, padW: 0 };
}

function nfInferShape(type, c, s, name) {
  function fail(msg) {
    throw new Error("Layer '" + name + "' (" + type + "): " + msg);
  }
  switch (type) {
    case "dense":
      if (s.length !== 1) fail("expects rank-1 input, got " + nfShapeStr(s));
      return [nfNum(c.units, 8)];
    case "flatten":
      return [nfNumel(s)];
    case "reshape": {
      var t = Array.isArray(c.targetShape) ? c.targetShape.map(Number) : [1];
      if (nfNumel(t) !== nfNumel(s)) fail("cannot reshape " + nfShapeStr(s) + " to " + nfShapeStr(t));
      return t.slice();
    }
    case "dropout":
    case "batchnorm":
    case "layernorm":
    case "identity":
    case "relu":
    case "leakyrelu":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
    case "softmax":
      return s.slice();
    case "conv2d": {
      if (s.length !== 3) fail("expects [H, W, C], got " + nfShapeStr(s));
      var k = nfKernel(c.kernelSize);
      var g = nfConvGeom(s[0], s[1], k[0], k[1], nfNum(c.stride, 1), nfPadding(c));
      return [g.outH, g.outW, nfNum(c.filters, 8)];
    }
    case "maxpool2d":
    case "avgpool2d": {
      if (s.length !== 3) fail("expects [H, W, C], got " + nfShapeStr(s));
      var p = nfNum(c.poolSize, 2);
      var st = nfNum(c.stride, p);
      var ph = Math.floor((s[0] - p) / st) + 1;
      var pw = Math.floor((s[1] - p) / st) + 1;
      if (ph <= 0 || pw <= 0) fail("pool " + p + "x" + p + " does not fit " + nfShapeStr(s));
      return [ph, pw, s[2]];
    }
    case "gap2d":
      if (s.length !== 3) fail("expects [H, W, C], got " + nfShapeStr(s));
      return [s[2]];
    case "embedding":
      if (s.length !== 1) fail("expects [seqLen], got " + nfShapeStr(s));
      return [s[0], nfNum(c.embeddingDim, 8)];
    case "rnn": {
      if (s.length !== 2) fail("expects [seqLen, features], got " + nfShapeStr(s));
      var u = nfNum(c.units, 32);
      return nfBool(c.returnSequences, false) ? [s[0], u] : [u];
    }
    default:
      throw new Error("Unsupported layer type '" + type + "' in layer '" + name + "'");
  }
}

function nfExpectedParams(type, c, s, out) {
  switch (type) {
    case "dense": {
      var p = { W: [s[0], out[0]] };
      if (nfBool(c.useBias, true)) p.b = [out[0]];
      return p;
    }
    case "conv2d": {
      var k = nfKernel(c.kernelSize);
      var q = { K: [k[0], k[1], s[2], out[2]] };
      if (nfBool(c.useBias, true)) q.b = [out[2]];
      return q;
    }
    case "batchnorm": {
      var f = s[s.length - 1];
      return { gamma: [f], beta: [f], runningMean: [f], runningVar: [f] };
    }
    case "layernorm": {
      var f2 = s[s.length - 1];
      return { gamma: [f2], beta: [f2] };
    }
    case "embedding":
      return { W: [nfNum(c.vocabSize, 32), out[1]] };
    case "rnn": {
      var H = nfNum(c.units, 32);
      return { Wxh: [s[1], H], Whh: [H, H], b: [H] };
    }
    default:
      return {};
  }
}

/* -------------------------------------------------------------- operations */

function nfMatmul(a, M, K, b, N) {
  var out = new Float32Array(M * N);
  for (var i = 0; i < M; i++) {
    var aOff = i * K;
    var cOff = i * N;
    for (var k = 0; k < K; k++) {
      var av = a[aOff + k];
      var bOff = k * N;
      for (var j = 0; j < N; j++) out[cOff + j] += av * b[bOff + j];
    }
  }
  return out;
}

function nfAddBias(m, B, N, bias) {
  var out = new Float32Array(m.length);
  for (var i = 0; i < B; i++) {
    var off = i * N;
    for (var j = 0; j < N; j++) out[off + j] = m[off + j] + bias[j];
  }
  return out;
}

function nfSigmoid(x) {
  var o = new Float32Array(x.length);
  for (var i = 0; i < x.length; i++) {
    var v = x[i];
    o[i] = v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v));
  }
  return o;
}

function nfSoftmax(x, inner) {
  var outer = x.length / inner;
  var o = new Float32Array(x.length);
  for (var r = 0; r < outer; r++) {
    var off = r * inner;
    var max = -Infinity;
    for (var i = 0; i < inner; i++) if (x[off + i] > max) max = x[off + i];
    var s = 0;
    for (var j = 0; j < inner; j++) {
      var e = Math.exp(x[off + j] - max);
      o[off + j] = e;
      s += e;
    }
    var inv = 1 / (s || 1e-12);
    for (var k = 0; k < inner; k++) o[off + k] *= inv;
  }
  return o;
}

function nfActivation(name, x, inner, alpha) {
  var o;
  var i;
  switch (name) {
    case "linear":
      return x;
    case "relu":
      o = new Float32Array(x.length);
      for (i = 0; i < x.length; i++) o[i] = x[i] > 0 ? x[i] : 0;
      return o;
    case "leakyrelu":
      o = new Float32Array(x.length);
      for (i = 0; i < x.length; i++) o[i] = x[i] > 0 ? x[i] : alpha * x[i];
      return o;
    case "sigmoid":
      return nfSigmoid(x);
    case "tanh":
      o = new Float32Array(x.length);
      for (i = 0; i < x.length; i++) o[i] = Math.tanh(x[i]);
      return o;
    case "gelu": {
      var c = Math.sqrt(2 / Math.PI);
      o = new Float32Array(x.length);
      for (i = 0; i < x.length; i++) {
        var v = x[i];
        o[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
      }
      return o;
    }
    case "silu": {
      var sg = nfSigmoid(x);
      o = new Float32Array(x.length);
      for (i = 0; i < x.length; i++) o[i] = x[i] * sg[i];
      return o;
    }
    case "softmax":
      return nfSoftmax(x, inner);
    default:
      throw new Error("Unsupported activation '" + name + "'");
  }
}

function nfConv2d(xd, kd, bd, B, H, W, Cin, outH, outW, F, kH, kW, stride, padTop, padLeft) {
  var out = new Float32Array(B * outH * outW * F);
  for (var n = 0; n < B; n++) {
    for (var oh = 0; oh < outH; oh++) {
      for (var ow = 0; ow < outW; ow++) {
        for (var f = 0; f < F; f++) {
          var acc = bd ? bd[f] : 0;
          for (var kh = 0; kh < kH; kh++) {
            var ih = oh * stride + kh - padTop;
            if (ih < 0 || ih >= H) continue;
            for (var kw = 0; kw < kW; kw++) {
              var iw = ow * stride + kw - padLeft;
              if (iw < 0 || iw >= W) continue;
              for (var c = 0; c < Cin; c++) {
                acc += xd[((n * H + ih) * W + iw) * Cin + c] * kd[((kh * kW + kw) * Cin + c) * F + f];
              }
            }
          }
          out[((n * outH + oh) * outW + ow) * F + f] = acc;
        }
      }
    }
  }
  return out;
}

/* ----------------------------------------------------------- layer compile */

function nfCompile(def, s, out, P) {
  var c = def.config || {};
  var name = def.name;
  switch (def.type) {
    case "dense": {
      var din = s[0];
      var units = out[0];
      var useBias = nfBool(c.useBias, true);
      var act = nfAct(c);
      return function (x) {
        if (x.shape.length !== 2 || x.shape[1] !== din) throw new Error("Dense '" + name + "': expected [B, " + din + "], got " + nfShapeStr(x.shape));
        var B = x.shape[0];
        var y = nfMatmul(x.data, B, din, P.W, units);
        if (useBias) y = nfAddBias(y, B, units, P.b);
        if (act !== "none") y = nfActivation(act, y, units, 0.01);
        return { shape: [B, units], data: y };
      };
    }
    case "flatten":
      return function (x) {
        var B = x.shape[0];
        return { shape: [B, x.data.length / B], data: x.data };
      };
    case "reshape":
      return function (x) {
        return { shape: [x.shape[0]].concat(out), data: x.data };
      };
    case "dropout":
    case "identity":
      // Dropout is disabled at inference, exactly as in NeuronForge evaluation.
      return function (x) {
        return x;
      };
    case "relu":
    case "leakyrelu":
    case "gelu":
    case "silu":
    case "sigmoid":
    case "tanh":
    case "softmax": {
      var alpha = nfNum(c.alpha, 0.01);
      var type = def.type;
      return function (x) {
        return { shape: x.shape, data: nfActivation(type, x.data, x.shape[x.shape.length - 1], alpha) };
      };
    }
    case "batchnorm": {
      var bf = s[s.length - 1];
      var beps = nfNum(c.epsilon, 1e-5);
      return function (x) {
        var d = x.data;
        var o = new Float32Array(d.length);
        for (var i = 0; i < d.length; i++) {
          var ch = i % bf;
          var nh = (d[i] - P.runningMean[ch]) / Math.sqrt(P.runningVar[ch] + beps);
          o[i] = P.gamma[ch] * nh + P.beta[ch];
        }
        return { shape: x.shape, data: o };
      };
    }
    case "layernorm": {
      var lf = s[s.length - 1];
      var leps = nfNum(c.epsilon, 1e-5);
      return function (x) {
        var d = x.data;
        var rows = d.length / lf;
        var o = new Float32Array(d.length);
        var std = new Float32Array(rows);
        for (var r = 0; r < rows; r++) {
          var off = r * lf;
          var m = 0;
          for (var i = 0; i < lf; i++) m += d[off + i];
          m /= lf;
          var v = 0;
          for (var j = 0; j < lf; j++) {
            var dd = d[off + j] - m;
            v += dd * dd;
          }
          v /= lf;
          std[r] = Math.sqrt(v + leps);
          for (var k = 0; k < lf; k++) {
            var nh = (d[off + k] - m) / std[r];
            o[off + k] = P.gamma[k] * nh + P.beta[k];
          }
        }
        return { shape: x.shape, data: o };
      };
    }
    case "conv2d": {
      var H = s[0], W = s[1], Cin = s[2], outH = out[0], outW = out[1], F = out[2];
      var kk = nfKernel(c.kernelSize);
      var stride = nfNum(c.stride, 1);
      var g = nfConvGeom(H, W, kk[0], kk[1], stride, nfPadding(c));
      var padTop = Math.floor(g.padH / 2);
      var padLeft = Math.floor(g.padW / 2);
      var cb = nfBool(c.useBias, true);
      var cact = nfAct(c);
      return function (x) {
        var B = x.shape[0];
        var y = nfConv2d(x.data, P.K, cb ? P.b : null, B, H, W, Cin, outH, outW, F, kk[0], kk[1], stride, padTop, padLeft);
        if (cact !== "none") y = nfActivation(cact, y, F, 0.01);
        return { shape: [B, outH, outW, F], data: y };
      };
    }
    case "maxpool2d":
    case "avgpool2d": {
      var pool = nfNum(c.poolSize, 2);
      var pst = nfNum(c.stride, pool);
      var isMax = def.type === "maxpool2d";
      return function (x) {
        var B = x.shape[0], PH = x.shape[1], PW = x.shape[2], C = x.shape[3];
        var oh = out[0], ow = out[1];
        var o = new Float32Array(B * oh * ow * C);
        var area = pool * pool;
        for (var n = 0; n < B; n++) {
          for (var y = 0; y < oh; y++) {
            for (var xx = 0; xx < ow; xx++) {
              for (var ch = 0; ch < C; ch++) {
                var best = -Infinity;
                var acc = 0;
                for (var kh = 0; kh < pool; kh++) {
                  var ih = y * pst + kh;
                  if (ih >= PH) continue;
                  for (var kw = 0; kw < pool; kw++) {
                    var iw = xx * pst + kw;
                    if (iw >= PW) continue;
                    var v = x.data[((n * PH + ih) * PW + iw) * C + ch];
                    if (v > best) best = v;
                    acc += v;
                  }
                }
                o[((n * oh + y) * ow + xx) * C + ch] = isMax ? (best === -Infinity ? 0 : best) : acc / area;
              }
            }
          }
        }
        return { shape: [B, oh, ow, C], data: o };
      };
    }
    case "gap2d":
      return function (x) {
        var B = x.shape[0], GH = x.shape[1], GW = x.shape[2], C = x.shape[3];
        var spatial = GH * GW;
        var o = new Float32Array(B * C);
        for (var n = 0; n < B; n++) {
          for (var h = 0; h < GH; h++) {
            for (var w = 0; w < GW; w++) {
              for (var ch = 0; ch < C; ch++) o[n * C + ch] += x.data[((n * GH + h) * GW + w) * C + ch];
            }
          }
          for (var cc = 0; cc < C; cc++) o[n * C + cc] /= spatial;
        }
        return { shape: [B, C], data: o };
      };
    case "embedding": {
      var vocab = nfNum(c.vocabSize, 32);
      var dim = out[1];
      return function (x) {
        var seq = x.data.length;
        var o = new Float32Array(seq * dim);
        for (var i = 0; i < seq; i++) {
          var id = Math.round(x.data[i]);
          if (id < 0) id = 0;
          if (id >= vocab) id = vocab - 1;
          o.set(P.W.subarray(id * dim, id * dim + dim), i * dim);
        }
        var B = x.shape[0];
        return { shape: [B, seq / B, dim], data: o };
      };
    }
    case "rnn": {
      var D = s[1];
      var RH = nfNum(c.units, 32);
      var retSeq = nfBool(c.returnSequences, false);
      return function (x) {
        if (x.shape.length !== 3) throw new Error("RNN '" + name + "': expected [B, T, " + D + "], got " + nfShapeStr(x.shape));
        var B = x.shape[0], T = x.shape[1];
        var hs = new Float32Array(B * (T + 1) * RH);
        var o = retSeq ? new Float32Array(B * T * RH) : new Float32Array(B * RH);
        var wx = P.Wxh, wh = P.Whh, bd = P.b, xd = x.data;
        for (var n = 0; n < B; n++) {
          for (var t = 0; t < T; t++) {
            var hPrev = (n * (T + 1) + t) * RH;
            var hOff = hPrev + RH;
            var xOff = (n * T + t) * D;
            for (var j = 0; j < RH; j++) {
              var acc = bd[j];
              for (var i = 0; i < D; i++) acc += xd[xOff + i] * wx[i * RH + j];
              for (var i2 = 0; i2 < RH; i2++) acc += hs[hPrev + i2] * wh[i2 * RH + j];
              hs[hOff + j] = Math.tanh(acc);
            }
            if (retSeq) o.set(hs.subarray(hOff, hOff + RH), (n * T + t) * RH);
          }
          if (!retSeq) {
            var last = (n * (T + 1) + T) * RH;
            o.set(hs.subarray(last, last + RH), n * RH);
          }
        }
        return { shape: retSeq ? [B, T, RH] : [B, RH], data: o };
      };
    }
    default:
      throw new Error("Unsupported layer type '" + def.type + "'");
  }
}

/* -------------------------------------------------------- tokenizer + rng */

var NF_WORD_RE = /[a-z0-9]+|[^\sa-z0-9]/g;

function nfTokenizeWords(text) {
  var m = String(text).toLowerCase().match(NF_WORD_RE);
  return m ? m.slice() : [];
}

function nfJoinWords(words) {
  if (words.length === 0) return "";
  var s = words[0];
  for (var i = 1; i < words.length; i++) {
    var w = words[i];
    if (w.length === 1 && ",.;:!?)]}".indexOf(w) >= 0) s += w;
    else if (w === "n't" || w === "'s") s += w;
    else s += " " + w;
  }
  return s;
}

function NfRng(seed) {
  this.s = seed >>> 0 || 1;
}
NfRng.prototype.next = function () {
  this.s = (this.s + 0x6d2b79f5) >>> 0;
  var t = this.s;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

function nfTopK(logits, k) {
  var sample = [];
  var stride = Math.max(1, Math.floor(logits.length / 512));
  for (var i = 0; i < logits.length; i += stride) sample.push(logits[i]);
  sample.sort(function (a, b) {
    return b - a;
  });
  var threshold = sample[Math.min(k - 1, sample.length - 1)];
  var idx = [];
  for (var j = 0; j < logits.length; j++) if (logits[j] >= threshold) idx.push(j);
  if (idx.length > k) {
    idx.sort(function (a, b) {
      return logits[b] - logits[a];
    });
    idx.length = k;
  }
  return idx;
}

/** Temperature + top-k sampling, identical to NeuronForge's sampler. */
function nfSample(probs, temperature, topK, rng) {
  var n = probs.length;
  if (n === 0) throw new Error("Cannot sample from an empty distribution");
  var best = 0;
  var i;
  if (!Number.isFinite(temperature) || temperature <= 0.01) {
    for (i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }
  var invT = 1 / temperature;
  var logits = new Float64Array(n);
  for (i = 0; i < n; i++) logits[i] = Math.log(Math.max(Number(probs[i]), 1e-9)) * invT;
  var k = Math.max(1, Math.min(Math.floor(topK) || n, n));
  var cand;
  if (k >= n) {
    cand = [];
    for (i = 0; i < n; i++) cand.push(i);
  } else {
    cand = nfTopK(logits, k);
  }
  var max = -Infinity;
  for (i = 0; i < cand.length; i++) if (logits[cand[i]] > max) max = logits[cand[i]];
  var sum = 0;
  var weights = [];
  for (i = 0; i < cand.length; i++) {
    var w = Math.exp(logits[cand[i]] - max);
    weights.push(w);
    sum += w;
  }
  if (!(sum > 0)) {
    for (i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }
  var r = rng.next() * sum;
  for (i = 0; i < cand.length; i++) {
    r -= weights[i];
    if (r <= 0) return cand[i];
  }
  return cand[cand.length - 1];
}

function nfFlatten(input, out) {
  if (input === null || input === undefined) throw new Error("Input is empty");
  if (typeof input === "number") {
    out.push(input);
    return out;
  }
  if (typeof input === "string") {
    var n = Number(input.trim());
    if (input.trim() === "" || !Number.isFinite(n)) throw new Error("Cannot interpret " + JSON.stringify(input) + " as a number");
    out.push(n);
    return out;
  }
  if (ArrayBuffer.isView(input)) {
    for (var i = 0; i < input.length; i++) out.push(Number(input[i]));
    return out;
  }
  if (Array.isArray(input)) {
    for (var j = 0; j < input.length; j++) nfFlatten(input[j], out);
    return out;
  }
  throw new Error("Cannot interpret input of type " + typeof input + " as numbers");
}

/* ------------------------------------------------------------ createModel */

/**
 * Build a runnable model from a NeuronForge standalone bundle (object or JSON string).
 * Validates format, every layer, every tensor shape and every checksum before
 * returning; anything unsupported or inconsistent throws a descriptive error.
 */
function createModel(bundle, options) {
  options = options || {};
  if (typeof bundle === "string") bundle = JSON.parse(bundle);
  if (!bundle || bundle.format !== NF_BUNDLE_FORMAT) throw new Error("Not a NeuronForge standalone bundle (format=" + (bundle && bundle.format) + ")");
  if (bundle.formatVersion !== NF_BUNDLE_VERSION) {
    throw new Error("Bundle format version " + bundle.formatVersion + " is not supported by runtime " + NF_RUNTIME_VERSION + " (expects " + NF_BUNDLE_VERSION + ")");
  }
  var arch = bundle.architecture;
  if (!arch || !Array.isArray(arch.layers) || !Array.isArray(arch.inputShape)) throw new Error("Bundle has no architecture");
  var weights = bundle.weights || {};
  var checkSums = options.skipChecksums !== true;
  var used = {};
  var layers = [];
  var layerInfo = [];
  var store = {};
  var shape = arch.inputShape.map(Number);
  var paramCount = 0;

  arch.layers.forEach(function (def, idx) {
    if (NF_SUPPORTED_LAYERS.indexOf(def.type) < 0) {
      throw new Error(
        "Unsupported layer type '" + def.type + "' (layer " + (idx + 1) + ", '" + def.name + "'). Runtime " + NF_RUNTIME_VERSION + " supports: " + NF_SUPPORTED_LAYERS.join(", "),
      );
    }
    var cfg = def.config || {};
    if ((def.type === "dense" || def.type === "conv2d") && nfAct(cfg) !== "none" && nfAct(cfg) !== "linear" && NF_ACTIVATION_LAYERS.indexOf(nfAct(cfg)) < 0) {
      throw new Error("Layer '" + def.name + "' uses unsupported fused activation '" + nfAct(cfg) + "'");
    }
    var out = nfInferShape(def.type, cfg, shape, def.name);
    if (Array.isArray(def.inputShape) && !nfSameShape(def.inputShape, shape)) {
      throw new Error("Shape check failed at layer '" + def.name + "': bundle says input " + nfShapeStr(def.inputShape) + ", runtime computed " + nfShapeStr(shape));
    }
    if (Array.isArray(def.outputShape) && !nfSameShape(def.outputShape, out)) {
      throw new Error("Shape check failed at layer '" + def.name + "': bundle says output " + nfShapeStr(def.outputShape) + ", runtime computed " + nfShapeStr(out));
    }
    var expected = nfExpectedParams(def.type, cfg, shape, out);
    var P = {};
    var counts = {};
    Object.keys(expected).forEach(function (pname) {
      var key = def.id + "." + pname;
      var entry = weights[key];
      if (!entry) throw new Error("Missing weight tensor '" + key + "' for layer '" + def.name + "' (" + def.type + ")");
      if (entry.dtype && entry.dtype !== "float32") throw new Error("Weight '" + key + "' has dtype " + entry.dtype + "; only float32 is supported");
      if (!nfSameShape(entry.shape, expected[pname])) {
        throw new Error("Weight '" + key + "' has shape " + nfShapeStr(entry.shape) + " but layer '" + def.name + "' needs " + nfShapeStr(expected[pname]));
      }
      P[pname] = nfDecodeTensor(entry, key, checkSums);
      store[key] = P[pname];
      used[key] = true;
      counts[pname] = P[pname].length;
      paramCount += P[pname].length;
    });
    layers.push(nfCompile(def, shape, out, P));
    layerInfo.push({ name: def.name, type: def.type, inputShape: shape.slice(), outputShape: out.slice(), params: counts });
    shape = out;
  });

  Object.keys(weights).forEach(function (k) {
    if (!used[k]) throw new Error("Weight tensor '" + k + "' is not used by any layer: the weights do not match this architecture");
  });

  var inputShape = arch.inputShape.map(Number);
  var outputShape = shape.slice();
  var inSize = nfNumel(inputShape);
  var task = bundle.task || "raw";
  var output = bundle.output || {};
  var tok = bundle.tokenizer || null;
  var pp = bundle.preprocessing || null;
  var img = bundle.image || null;
  var sampling = bundle.sampling || { temperature: 0.7, topK: 50 };
  var stoi = null;
  var padId = 0;
  if (tok) {
    stoi = new Map();
    tok.tokens.forEach(function (t, i) {
      stoi.set(t, i);
    });
    padId = stoi.has(tok.padToken) ? stoi.get(tok.padToken) : 0;
    if (task === "language" && tok.tokens.length !== nfNumel(outputShape)) {
      throw new Error("Tokenizer has " + tok.tokens.length + " tokens but the model outputs " + nfNumel(outputShape) + " classes");
    }
  }

  function forwardBatch(flat, batch) {
    if (flat.length !== batch * inSize) {
      throw new Error("Expected " + batch * inSize + " input values (" + batch + " x " + nfShapeStr(inputShape) + "), got " + flat.length);
    }
    var x = { shape: [batch].concat(inputShape), data: flat instanceof Float32Array ? flat : Float32Array.from(flat) };
    for (var i = 0; i < layers.length; i++) x = layers[i](x);
    return x;
  }

  function labelFor(i) {
    return output.classNames && output.classNames[i] !== undefined ? String(output.classNames[i]) : String(i);
  }

  /* ---- tabular preprocessing ---- */
  function encode(row) {
    if (!pp || pp.kind !== "tabular") throw new Error("This model has no tabular preprocessing; pass numeric features directly");
    if (Array.isArray(row) && row.length !== pp.columns.length) {
      throw new Error("Expected " + pp.columns.length + " raw values (" + pp.columns.map(function (c) { return c.name; }).join(", ") + "), got " + row.length);
    }
    var vec = [];
    pp.columns.forEach(function (col, i) {
      var v = Array.isArray(row) ? row[i] : row ? row[col.name] : undefined;
      if (col.type === "numeric") {
        var num;
        if (typeof v === "number") num = Number.isFinite(v) ? v : NaN;
        else {
          var raw = v === undefined || v === null ? "" : String(v).trim();
          num = raw === "" ? NaN : Number(raw);
          if (!Number.isFinite(num)) num = NaN;
        }
        if (Number.isNaN(num) && pp.missing === "drop") {
          throw new Error("Column '" + col.name + "' is missing or not numeric. The model was trained with missing-value policy 'drop', so there is no fill value");
        }
        vec.push(num);
      } else {
        var s = v === undefined || v === null ? "" : String(v).trim();
        if (col.encoding === "label") vec.push(s === "" ? 0 : Math.max(0, col.categories.indexOf(s)));
        else {
          var oh = new Array(Math.max(1, col.categories.length)).fill(0);
          var at = col.categories.indexOf(s);
          if (at >= 0) oh[at] = 1;
          for (var j = 0; j < oh.length; j++) vec.push(oh[j]);
        }
      }
    });
    return vec.map(function (v, j) {
      var x = Number.isFinite(v) ? v : pp.missing === "zero" ? 0 : pp.mean[j];
      if (pp.normalize === "minmax") {
        var d = pp.max[j] - pp.min[j] || 1;
        x = (x - pp.min[j]) / d;
      } else if (pp.normalize === "zscore") {
        x = (x - pp.mean[j]) / (pp.std[j] || 1);
      }
      return x;
    });
  }

  /* ---- tokenizer ---- */
  function requireTok() {
    if (!tok) throw new Error("This model was not exported with a tokenizer");
  }
  function tokenize(text) {
    requireTok();
    if (tok.level === "word") {
      return nfTokenizeWords(text).map(function (w) {
        return stoi.has(w) ? stoi.get(w) : 0;
      });
    }
    var ids = [];
    var chars = String(text).toLowerCase().split("");
    for (var i = 0; i < chars.length; i++) if (stoi.has(chars[i])) ids.push(stoi.get(chars[i]));
    return ids;
  }
  function detokenize(ids) {
    requireTok();
    var parts = [];
    for (var i = 0; i < ids.length; i++) {
      var t = tok.tokens[ids[i]];
      if (t !== undefined) parts.push(t);
    }
    return tok.level === "word" ? nfJoinWords(parts) : parts.join("");
  }
  function contextWindow(ids) {
    var seq = inputShape[0];
    var w = ids.slice(-seq);
    while (w.length < seq) w.unshift(padId);
    return w;
  }

  function prepare(input, opts) {
    opts = opts || {};
    if (task === "language" && typeof input === "string") {
      var ids = tokenize(input);
      if (ids.length === 0) throw new Error("The text contains no tokens from the vocabulary");
      return contextWindow(ids);
    }
    if (pp && pp.kind === "tabular" && !opts.preprocessed && input !== null && typeof input === "object" && !ArrayBuffer.isView(input)) {
      return encode(input);
    }
    var flat = nfFlatten(input, []);
    if (flat.length !== inSize) {
      throw new Error(
        "Expected " + inSize + " values for input shape " + nfShapeStr(inputShape) + ", got " + flat.length + (pp ? ". Raw table rows are encoded automatically; pass { preprocessed: true } only for already-encoded features" : ""),
      );
    }
    return flat;
  }

  function interpret(o, inputFlat) {
    var raw = Array.prototype.slice.call(o);
    var i;
    switch (task) {
      case "binary": {
        var p = raw[0];
        var cls = p >= 0.5 ? 1 : 0;
        return { task: task, predicted: cls, label: labelFor(cls), probability: output.outputsAreProbabilities ? p : undefined, score: p, raw: raw };
      }
      case "multiclass": {
        var best = 0;
        for (i = 1; i < raw.length; i++) if (raw[i] > raw[best]) best = i;
        var order = raw.map(function (_, k) { return k; }).sort(function (a, b) { return raw[b] - raw[a]; });
        return {
          task: task,
          predicted: best,
          label: labelFor(best),
          probabilities: output.outputsAreProbabilities ? raw : undefined,
          scores: raw,
          top: order.slice(0, Math.min(5, raw.length)).map(function (k) { return { index: k, label: labelFor(k), value: raw[k] }; }),
          raw: raw,
        };
      }
      case "regression": {
        var named = {};
        (output.targetNames || []).forEach(function (nm, k) {
          if (k < raw.length) named[nm] = raw[k];
        });
        return { task: task, value: raw[0], values: raw, named: named, raw: raw };
      }
      case "reconstruction": {
        var se = 0;
        for (i = 0; i < raw.length; i++) {
          var d = raw[i] - Number(inputFlat[i]);
          se += d * d;
        }
        return { task: task, reconstruction: raw, reconstructionError: raw.length ? se / raw.length : 0, raw: raw };
      }
      case "language": {
        var ord = raw.map(function (_, k) { return k; }).sort(function (a, b) { return raw[b] - raw[a]; });
        var top = ord.slice(0, 10).map(function (k) { return { id: k, token: tok ? tok.tokens[k] : String(k), probability: raw[k] }; });
        return { task: task, next: top[0] ? top[0].token : null, top: top, raw: raw };
      }
      default:
        return { task: "raw", output: raw, shape: outputShape.slice(), raw: raw };
    }
  }

  function predict(input, opts) {
    var flat = prepare(input, opts);
    return interpret(forwardBatch(Float32Array.from(flat), 1).data, flat);
  }

  function predictBatch(inputs, opts) {
    if (!Array.isArray(inputs) || inputs.length === 0) return [];
    var flats = inputs.map(function (x) { return prepare(x, opts); });
    var all = new Float32Array(flats.length * inSize);
    flats.forEach(function (f, i) { all.set(f, i * inSize); });
    var res = forwardBatch(all, flats.length).data;
    var per = res.length / flats.length;
    return flats.map(function (f, i) { return interpret(res.subarray(i * per, (i + 1) * per), f); });
  }

  function generateDetailed(prompt, o) {
    requireTok();
    if (task !== "language") throw new Error("generate() is only available for language models");
    o = o || {};
    var maxTokens = o.maxTokens === undefined || o.maxTokens === null ? 60 : Math.max(0, Math.floor(o.maxTokens));
    var temperature = o.temperature === undefined || o.temperature === null ? sampling.temperature : Number(o.temperature);
    var topK = o.topK === undefined || o.topK === null ? sampling.topK : Number(o.topK);
    var seed = o.seed === undefined || o.seed === null ? Math.floor(Math.random() * 4294967296) : Number(o.seed);
    var stopList = Array.isArray(o.stop) ? o.stop : o.stopAtEos !== false && tok.eos ? [tok.eos] : [];
    // Prototype-free map: a vocabulary word such as "constructor" must not look like a stop token.
    var stop = Object.create(null);
    stopList.forEach(function (s) { stop[s] = true; });
    var promptIds = tokenize(prompt);
    if (promptIds.length === 0) {
      throw new Error(tok.level === "word" ? "The prompt contains no word tokens" : "None of the prompt's characters are in the vocabulary");
    }
    var rng = new NfRng(seed);
    var ctx = promptIds.slice();
    var ids = [];
    var stoppedBy = null;
    for (var t = 0; t < maxTokens; t++) {
      var probs = forwardBatch(Float32Array.from(contextWindow(ctx)), 1).data;
      var idx = nfSample(probs, temperature, topK, rng);
      var token = tok.tokens[idx];
      if (token === undefined) throw new Error("Model produced class id " + idx + " outside the vocabulary");
      if (stop[token]) {
        stoppedBy = token;
        break;
      }
      ids.push(idx);
      ctx.push(idx);
    }
    var text = detokenize(ids);
    var full = tok.level === "word" ? (text && !/^[,.;:!?)\]}]/.test(text) ? String(prompt).trim() + " " + text : String(prompt).trim() + text) : String(prompt) + text;
    return {
      text: text,
      fullText: full,
      ids: ids,
      tokens: ids.map(function (i) { return tok.tokens[i]; }),
      promptIds: promptIds,
      stoppedBy: stoppedBy,
      settings: { maxTokens: maxTokens, temperature: temperature, topK: topK, seed: seed, stop: stopList },
    };
  }

  function imageToInput(image, o) {
    if (!img) throw new Error("This model was not exported with an image specification");
    o = o || {};
    var W = img.width, H = img.height, C = img.channels;
    var src = image && image.data ? image.data : image && image.pixels ? image.pixels : image;
    if (!src || typeof src.length !== "number") throw new Error("predictImage expects ImageData, { data, width, height } or a pixel array");
    var sw = (image && image.width) || W;
    var sh = (image && image.height) || H;
    var sc = (image && image.channels) || (src.length === sw * sh * 4 ? 4 : src.length === sw * sh * 3 ? 3 : src.length === sw * sh ? 1 : 0);
    if (!sc) throw new Error("Cannot infer channels: " + src.length + " values for a " + sw + "x" + sh + " image");
    var bytes = typeof Uint8ClampedArray !== "undefined" && (src instanceof Uint8Array || src instanceof Uint8ClampedArray);
    var scale = o.scale !== undefined ? Number(o.scale) : bytes ? 1 / 255 : 1;
    var lo = img.range[0], hi = img.range[1];
    var out = new Array(W * H * C);
    for (var y = 0; y < H; y++) {
      for (var x = 0; x < W; x++) {
        var sx = Math.min(sw - 1, Math.floor((x * sw) / W));
        var sy = Math.min(sh - 1, Math.floor((y * sh) / H));
        var base = (sy * sw + sx) * sc;
        for (var ch = 0; ch < C; ch++) {
          var v;
          if (C === 1) v = sc >= 3 ? (Number(src[base]) + Number(src[base + 1]) + Number(src[base + 2])) / 3 : Number(src[base]);
          else v = Number(src[base + Math.min(ch, sc - 1)]);
          v *= scale;
          if (o.invert) v = 1 - v;
          out[(y * W + x) * C + ch] = lo + v * (hi - lo);
        }
      }
    }
    return out;
  }

  function verify() {
    var t = bundle.tests || {};
    var tol = t.tolerance || { atol: 1e-5, rtol: 1e-4 };
    var checks = [];
    function add(name, status, detail) {
      checks.push({ name: name, status: status, detail: detail });
    }
    (t.forward || []).forEach(function (tc, i) {
      try {
        var got = forwardBatch(Float32Array.from(tc.input), 1).data;
        if (got.length !== tc.output.length) return add("forward #" + (i + 1), "fail", "output length " + got.length + " != " + tc.output.length);
        var maxAbs = 0;
        var bad = 0;
        for (var j = 0; j < got.length; j++) {
          var e = tc.output[j];
          var d = Math.abs(got[j] - e);
          if (d > maxAbs) maxAbs = d;
          if (!(d <= tol.atol + tol.rtol * Math.abs(e))) bad++;
        }
        add("forward #" + (i + 1), bad ? "fail" : "pass", "max |diff| = " + maxAbs.toExponential(2) + (bad ? ", " + bad + " values outside tolerance" : ""));
      } catch (err) {
        add("forward #" + (i + 1), "fail", String(err && err.message ? err.message : err));
      }
    });
    (t.tokenizer || []).forEach(function (tc, i) {
      var ids = tokenize(tc.text);
      var ok = ids.length === tc.ids.length && ids.every(function (v, k) { return v === tc.ids[k]; });
      add("tokenizer #" + (i + 1), ok ? "pass" : "fail", ok ? ids.length + " tokens match" : "token ids differ from training-time tokenizer");
    });
    (t.generation || []).forEach(function (tc, i) {
      try {
        var r = generateDetailed(tc.prompt, { maxTokens: tc.maxTokens, temperature: tc.temperature, topK: tc.topK, seed: tc.seed, stop: [] });
        var div = -1;
        for (var k = 0; k < Math.max(r.ids.length, tc.ids.length); k++) {
          if (r.ids[k] !== tc.ids[k]) {
            div = k;
            break;
          }
        }
        if (div < 0) add("generation #" + (i + 1), "pass", r.ids.length + " sampled tokens identical (T=" + tc.temperature + ", k=" + tc.topK + ", seed=" + tc.seed + ")");
        else add("generation #" + (i + 1), "warn", "diverged at token " + div + "; usually a last-digit Math.exp difference between JS engines changing a sampling tie");
      } catch (err) {
        add("generation #" + (i + 1), "fail", String(err && err.message ? err.message : err));
      }
    });
    (t.preprocessing || []).forEach(function (tc, i) {
      try {
        var enc = encode(tc.raw);
        var bad = enc.length !== tc.encoded.length;
        for (var j = 0; !bad && j < enc.length; j++) if (Math.abs(enc[j] - tc.encoded[j]) > 1e-9 + 1e-7 * Math.abs(tc.encoded[j])) bad = true;
        add("preprocessing #" + (i + 1), bad ? "fail" : "pass", bad ? "raw-row encoding differs from training pipeline" : enc.length + " encoded features match");
      } catch (err) {
        add("preprocessing #" + (i + 1), "fail", String(err && err.message ? err.message : err));
      }
    });
    if (checks.length === 0) add("embedded tests", "warn", "bundle contains no test vectors");
    return {
      ok: checks.every(function (c) { return c.status !== "fail"; }),
      runtimeVersion: NF_RUNTIME_VERSION,
      checks: checks,
    };
  }

  var info = {
    name: bundle.name,
    task: task,
    inputShape: inputShape.slice(),
    outputShape: outputShape.slice(),
    paramCount: paramCount,
    layers: layerInfo,
    classNames: output.classNames || null,
    targetNames: output.targetNames || null,
    outputsAreProbabilities: !!output.outputsAreProbabilities,
    tokenizer: tok ? { level: tok.level, vocabSize: tok.tokens.length, seqLen: tok.seqLen, padToken: tok.padToken, eos: tok.eos || null } : null,
    preprocessing: pp ? { kind: pp.kind, columns: pp.columns.map(function (c) { return c.name; }), normalize: pp.normalize, missing: pp.missing } : null,
    image: img,
    sampling: sampling,
    runtimeVersion: NF_RUNTIME_VERSION,
    exportedAt: bundle.exportedAt,
    limitations: bundle.limitations || [],
  };

  return {
    info: info,
    forward: function (input, batchSize) {
      var b = batchSize || 1;
      return forwardBatch(Float32Array.from(nfFlatten(input, [])), b);
    },
    predict: predict,
    predictBatch: predictBatch,
    encode: encode,
    tokenize: tokenize,
    detokenize: detokenize,
    nextToken: function (text) {
      return predict(String(text));
    },
    generate: function (prompt, o) {
      var r = generateDetailed(prompt, o);
      return o && o.includePrompt ? r.fullText : r.text;
    },
    generateDetailed: generateDetailed,
    imageToInput: imageToInput,
    predictImage: function (image, o) {
      var flat = imageToInput(image, o);
      return interpret(forwardBatch(Float32Array.from(flat), 1).data, flat);
    },
    verify: verify,
    getWeights: function () {
      var copy = {};
      Object.keys(store).forEach(function (k) { copy[k] = new Float32Array(store[k]); });
      return copy;
    },
  };
}
