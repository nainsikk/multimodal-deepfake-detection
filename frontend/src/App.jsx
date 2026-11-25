
import { useState } from "react";
import {
 ShieldCheck, History,
  UploadCloud, Image, FileText, Zap, Activity,
  CheckCircle2, AlertTriangle, XCircle, ChevronRight,
  Eye, ScanSearch, Download, Menu, X
} from "lucide-react";

import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

const mockResults = [
  {
    prediction: "REAL",
    confidence: 94,
    manipulation_type: "No manipulation detected",
    modality_scores: { text: 93, image: 95, fusion: 94 },
    evidence: {
      text_tokens: [],
      mismatch_score: 8
    },
    explanation:
      "The caption and image appear consistent. No significant cross-modal contradiction was detected."
  },
  {
    prediction: "FAKE",
    confidence: 91,
    manipulation_type: "Caption-image mismatch",
    modality_scores: { text: 89, image: 82, fusion: 91 },
    evidence: {
      text_tokens: [
        { token: "breaking", weight: 0.31 },
        { token: "flood", weight: 0.42 }
      ],
      mismatch_score: 87
    },
    explanation:
      "The caption describes a flood, but the image shows a dry street. This indicates a possible caption-image mismatch."
  },
  {
    prediction: "UNCERTAIN",
    confidence: 56,
    manipulation_type: "Uncertain evidence",
    modality_scores: { text: 59, image: 62, fusion: 56 },
    evidence: {
      text_tokens: [
        { token: "reportedly", weight: 0.18 }
      ],
      mismatch_score: 49
    },
    explanation:
      "The available text and image evidence is insufficient for a reliable decision. Further verification is recommended."
  }
];

// Convert backend decimal scores into percentages.
// Mock scores that are already percentages remain unchanged.
const toPercent = (value) => {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const number = Number(value);

  if (!Number.isFinite(number)) {
    return null;
  }

  const percentage = number <= 1 ? number * 100 : number;

  return Math.min(100, Math.max(0, percentage));
};


function App() {
  const [page, setPage] = useState("Analyze");
  const [caption, setCaption] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [useText, setUseText] = useState(true);
  const [useImage, setUseImage] = useState(true);
  const [heatmap, setHeatmap] = useState(false);
  const [history, setHistory] = useState([]);
  const [mobileMenu, setMobileMenu] = useState(false);

  const handleImage = (file) => {
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Please upload a JPG, PNG or WEBP image.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("Image size must be below 5 MB.");
      return;
    }

    setError("");
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const analyze = async () => {
  if (!useText && !useImage) {
    setError("Please select at least one modality.");
    return;
  }

  if (useText && !caption.trim()) {
    setError("Please enter a caption or text.");
    return;
  }

  if (useImage && !imageFile) {
    setError("Please upload an image.");
    return;
  }

  setError("");
  setLoading(true);
  setResult(null);

  try {
    const formData = new FormData();
    if (useText) formData.append("text", caption);
    if (useImage && imageFile) formData.append("image", imageFile);

    const modalities = [useText && "text", useImage && "image"]
      .filter(Boolean)
      .join(",");
    formData.append("modalities", modalities);

    const res = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    console.log("Backend response:", data);

    const newResult = {
      ...data,
      id: data.analysis_id || data.id || Date.now(),
      text: caption,
    };

    setResult(newResult);
    setHistory((previous) => [newResult, ...previous]);
  } catch (err) {
    console.error(err);
    setError("Could not reach the backend. Is it running on port 8000?");
  } finally {
    setLoading(false);
  }
};

  const runDemo = (index) => {
    setCaption([
      "A beautiful sunny day in the city.",
      "Breaking news: Massive flood hits the entire city.",
      "A viral claim is circulating on social media."
    ][index]);

    const mock = mockResults[index];

  setResult({ ...mock, id: `mock-${index}` });
    setError("");
    setPage("Analyze");
  };

  const downloadReport = () => {
    if (!result) return;

    const report = JSON.stringify(result, null, 2);
    const blob = new Blob([report], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "deepfake-analysis-report.json";
    link.click();

    URL.revokeObjectURL(url);
  };

  const navItems = [
    { name: "Analyze", icon: ScanSearch },
    { name: "History", icon: History },
    { name: "Sanity Check", icon: Activity },
    { name: "About / Ethics", icon: ShieldCheck }
  ];

  const predictionStyle = {
    REAL: {
      color: "text-emerald-400",
      bg: "bg-emerald-400/10",
      icon: CheckCircle2
    },
    FAKE: {
      color: "text-red-400",
      bg: "bg-red-400/10",
      icon: XCircle
    },
    UNCERTAIN: {
      color: "text-amber-400",
      bg: "bg-amber-400/10",
      icon: AlertTriangle
    }
  };

const confidencePercent = toPercent(result?.confidence);

const mismatchPercent = toPercent(
  result?.evidence?.mismatch_score
);


  return (
    <div className="min-h-screen bg-[#080d19] text-slate-100">

      {/* SIDEBAR */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 border-r border-slate-800 bg-[#0c1322] p-5 transition-transform md:translate-x-0 ${mobileMenu ? "translate-x-0" : "-translate-x-full"}`}>

        <div className="mb-12 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-violet-600 p-2.5">
              <ShieldCheck size={25} />
            </div>
            <div>
              <h1 className="text-lg font-bold">Multimodal Deepfake Detection</h1>
              <p className="text-xs text-slate-500">Content Verification</p>
            </div>
          </div>
          <button onClick={() => setMobileMenu(false)} className="md:hidden">
            <X />
          </button>
        </div>

        <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-slate-500">
          Workspace
        </p>

        <div className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                onClick={() => {
                  setPage(item.name);
                  setMobileMenu(false);
                }}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
                  page === item.name
                    ? "bg-violet-600/15 text-violet-400"
                    : "text-slate-400 hover:bg-slate-800 hover:text-white"
                }`}
              >
                <Icon size={19} />
                {item.name}
                {page === item.name && (
                  <ChevronRight className="ml-auto" size={16} />
                )}
              </button>
            );
          })}
        </div>

        <div className="absolute bottom-6 left-5 right-5 rounded-xl border border-slate-800 bg-slate-900 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
            <Zap size={17} className="text-violet-400" />
            AI Engine
          </div>
          <p className="text-xs text-slate-500">
            Connected to backend API
          </p>
          <div className="mt-3 h-1.5 rounded-full bg-slate-800">
            <div className="h-full w-full rounded-full bg-violet-500" />
          </div>
        </div>
      </aside>

      {/* MAIN AREA */}
      <main className="min-h-screen md:ml-64">

        {/* TOPBAR */}
        <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-slate-800 bg-[#080d19]/95 px-5 backdrop-blur md:px-10">

          <div className="flex items-center gap-3">
            <button onClick={() => setMobileMenu(true)} className="md:hidden">
              <Menu />
            </button>
            <div>
              <p className="text-sm font-semibold">{page}</p>
              <p className="text-xs text-slate-500">Deepfake Detection Platform</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400 sm:block">
              ● System Online
            </span>
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-violet-600 font-bold">
              M3
            </div>
          </div>
        </header>

        <div className="mx-auto max-w-7xl p-5 md:p-10">

          {/* ANALYZE PAGE */}
          {page === "Analyze" && (
            <>
              <div className="mb-9">
                <p className="mb-2 text-sm font-medium text-violet-400">
                  MULTIMODAL AI ANALYSIS
                </p>
                <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
                  Verify what's <span className="text-violet-400">real.</span>
                </h2>
                <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
                  Upload an image and its caption to detect possible
                  misinformation and cross-modal inconsistencies.
                </p>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">

                {/* TEXT INPUT */}
                <section className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                  <div className="mb-5 flex items-center gap-3">
                    <div className="rounded-lg bg-violet-500/10 p-2.5 text-violet-400">
                      <FileText size={21} />
                    </div>
                    <div>
                      <h3 className="font-semibold">Caption / Text</h3>
                      <p className="text-xs text-slate-500">Enter content to verify</p>
                    </div>
                  </div>

                  <textarea
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    placeholder="Paste your news caption, social media post or description here..."
                    className="h-52 w-full resize-none rounded-xl border border-slate-700 bg-[#0b1220] p-4 text-sm leading-6 text-white outline-none transition placeholder:text-slate-600 focus:border-violet-500"
                  />

                  <div className="mt-3 flex justify-between text-xs text-slate-500">
                    <span>Text analysis input</span>
                    <span>{caption.length} characters</span>
                  </div>
                </section>

                {/* IMAGE UPLOAD */}
                <section className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                  <div className="mb-5 flex items-center gap-3">
                    <div className="rounded-lg bg-blue-500/10 p-2.5 text-blue-400">
                      <Image size={21} />
                    </div>
                    <div>
                      <h3 className="font-semibold">Image Upload</h3>
                      <p className="text-xs text-slate-500">Upload an image for verification</p>
                    </div>
                  </div>

                  <label className="flex h-52 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-slate-700 bg-[#0b1220] transition hover:border-violet-500">

                    {preview ? (
                      <img
                        src={preview}
                        alt="Uploaded preview"
                        className="h-full w-full object-contain"
                      />
                    ) : (
                      <>
                        <UploadCloud size={38} className="mb-3 text-violet-400" />
                        <p className="text-sm font-medium">Drop your image here</p>
                        <p className="mt-2 text-xs text-slate-500">
                          or click to browse
                        </p>
                        <p className="mt-3 text-xs text-slate-600">
                          JPG, PNG, WEBP · Maximum 5 MB
                        </p>
                      </>
                    )}

                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      className="hidden"
                      onChange={(e) => handleImage(e.target.files?.[0])}
                    />
                  </label>

                  {imageFile && (
                    <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
                      <span>{imageFile.name}</span>
                      <button
                        onClick={() => {
                          setImageFile(null);
                          setPreview(null);
                        }}
                        className="text-red-400"
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </section>
              </div>

              {/* MODALITY SETTINGS */}
              <section className="mt-6 rounded-2xl border border-slate-800 bg-[#101827] p-5">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h3 className="text-sm font-semibold">Ablation Controls</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      Select modalities for analysis
                    </p>
                  </div>

                  <div className="flex gap-5">
                    <label className="flex cursor-pointer items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={useText}
                        onChange={(e) => setUseText(e.target.checked)}
                        className="accent-violet-500"
                      />
                      Use text
                    </label>

                    <label className="flex cursor-pointer items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={useImage}
                        onChange={(e) => setUseImage(e.target.checked)}
                        className="accent-violet-500"
                      />
                      Use image
                    </label>
                  </div>
                </div>
              </section>

              {error && (
                <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
                  <AlertTriangle className="mr-2 inline" size={16} />
                  {error}
                </p>
              )}

              <button
                onClick={analyze}
                disabled={loading}
                className="mt-6 flex w-full items-center justify-center gap-3 rounded-xl bg-violet-600 py-4 font-semibold transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto sm:px-12"
              >
                {loading ? (
                  <>
                    <Activity className="animate-spin" size={19} />
                    Analyzing content...
                  </>
                ) : (
                  <>
                    <ScanSearch size={19} />
                    Analyze Content
                  </>
                )}
              </button>

              {/* RESULT SECTION */}
              {result && (
                <section className="mt-12 space-y-6">

                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <p className="text-sm text-violet-400">ANALYSIS REPORT</p>
                      <h2 className="mt-1 text-2xl font-bold">Detection Results</h2>
                    </div>

                    <button
                      onClick={downloadReport}
                      className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-2.5 text-sm hover:bg-slate-800"
                    >
                      <Download size={17} />
                      Download Report
                    </button>
                  </div>

                  {/* PREDICTION */}
                  <div className="grid gap-5 md:grid-cols-2">

                    <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                      <p className="mb-4 text-sm text-slate-400">Prediction</p>

                      {(() => {
                        const style = predictionStyle[result.prediction];
                        const Icon = style.icon;

                        return (
                          <div className={`inline-flex items-center gap-3 rounded-xl px-5 py-3 ${style.bg} ${style.color}`}>
                            <Icon size={27} />
                            <span className="text-2xl font-bold">
                              {result.prediction}
                            </span>
                          </div>
                        );
                      })()}

                      <p className="mt-5 text-sm text-slate-400">
                        {result.manipulation_type}
                      </p>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                      <div className="flex justify-between">
                        <p className="text-sm text-slate-400">Confidence Score</p>
                        <span className="text-2xl font-bold text-violet-400">
                          
                        {confidencePercent === null
                          ? "N/A"
                          : `${confidencePercent.toFixed(2)}%`}

                        </span>
                      </div>

                      <div className="mt-6 h-3 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-violet-500 transition-all"
                         style={{ width: `${confidencePercent ?? 0}%` } }
                        />
                      </div>

                      <p className="mt-4 text-xs text-slate-500">
                        Model confidence · Not a guarantee of authenticity
                      </p>
                    </div>
                  </div>

                  {/* SCORE CHART */}
                  <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                    <h3 className="mb-6 font-semibold">Modality Scores</h3>

                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          
                          data={[
                            {
                              name: "Text",
                              score: toPercent(result.modality_scores?.text)
                            },
                            {
                              name: "Image",
                              score: toPercent(result.modality_scores?.image)
                            },
                            {
                              name: "Fusion",
                              score: toPercent(result.modality_scores?.fusion)
                            }
                          ].filter((item) => item.score !== null)}

                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#263246" />
                          <XAxis dataKey="name" stroke="#94a3b8" />
                          <YAxis domain={[0, 100]} stroke="#94a3b8" />
                          <Tooltip
                            contentStyle={{
                              background: "#101827",
                              border: "1px solid #334155",
                              borderRadius: "10px"
                            }}
                          />
                          <Bar dataKey="score" fill="#8b5cf6" radius={[7, 7, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* IMAGE EVIDENCE */}
                  <div className="grid gap-5 md:grid-cols-2">

                    <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                      <div className="mb-4 flex items-center justify-between">
                        <h3 className="font-semibold">Visual Evidence</h3>
                        <Eye size={19} className="text-violet-400" />
                      </div>

                      <div className="mb-4 flex gap-2">
                        <button
                          onClick={() => setHeatmap(false)}
                          className={`rounded-lg px-3 py-2 text-xs ${!heatmap ? "bg-violet-600" : "bg-slate-800"}`}
                        >
                          Original
                        </button>
                        <button
                          onClick={() => setHeatmap(true)}
                          className={`rounded-lg px-3 py-2 text-xs ${heatmap ? "bg-violet-600" : "bg-slate-800"}`}
                        >
                          Heatmap
                        </button>
                      </div>

                      <div className="flex h-52 items-center justify-center overflow-hidden rounded-xl bg-[#0b1220]">
                        {preview ? (
                          <img
                            src={preview}
                            alt="Analysis image"
                            className={`h-full w-full object-contain ${heatmap ? "brightness-75 saturate-150" : ""}`}
                          />
                        ) : (
                          <div className="text-center text-slate-500">
                            <Image size={38} className="mx-auto mb-2" />
                            <p className="text-xs">No image preview available</p>
                          </div>
                        )}
                      </div>

                      <p className="mt-3 text-xs text-slate-500">
                        {heatmap
                          ? "Demo view only. Actual Grad-CAM heatmap will be integrated later."
                          : "Original uploaded image"}
                      </p>
                    </div>

                    {/* MISMATCH */}
                    <div className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                      <h3 className="mb-5 font-semibold">Mismatch Analysis</h3>

                      <div className="flex items-center justify-center py-6">
                        <div className="relative flex h-36 w-36 items-center justify-center rounded-full border-10 border-slate-800">
                          <div
                            className="absolute -inset-2.5 rounded-full border-10 border-amber-400"
                            style={{
                              clipPath: `inset(0 ${100 - (result.evidence.mismatch_score ?? 0)}% 0 0)`
                            }}
                          />
                          <div className="text-center">
                            <p className="text-3xl font-bold">
                              {mismatchPercent ?? "N/A"}
                            </p>
                            <p className="text-xs text-slate-500">Mismatch</p>
                          </div>
                        </div>
                      </div>

                      <p className="text-center text-sm text-slate-400">
                        Text-image inconsistency score
                      </p>
                    </div>
                  </div>

                  {/* EXPLANATION */}
                  <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-6">
                    <div className="mb-4 flex items-center gap-3">
                      <ShieldCheck className="text-violet-400" size={22} />
                      <h3 className="font-semibold">AI Explanation</h3>
                    </div>

                    <p className="text-sm leading-7 text-slate-300">
                      {result.explanation}
                    </p>

                    {result.evidence.text_tokens.length > 0 && (
                      <div className="mt-5">
                        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                          Important Text Tokens
                        </p>

                        <div className="flex flex-wrap gap-2">
                          {result.evidence.text_tokens.map((item, index) => (
                            <span
                              key={index}
                              className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-sm text-amber-300"
                            >
                              {item.token} · {(item.weight * 100).toFixed(0)}%
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <p className="text-xs text-slate-500">
                   Result generated by the analysis backend.
                  </p>
                </section>
              )}
            </>
          )}

          {/* HISTORY PAGE */}
          {page === "History" && (
            <section>
              <h2 className="mb-3 text-3xl font-bold">Analysis History</h2>
              <p className="mb-8 text-sm text-slate-400">
                Previously analyzed content.
              </p>

              <div className="overflow-x-auto rounded-2xl border border-slate-800 bg-[#101827]">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-slate-800 text-slate-400">
                    <tr>
                      <th className="p-5">ID</th>
                      <th className="p-5">Prediction</th>
                      <th className="p-5">Confidence</th>
                      <th className="p-5">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item) => (
                      <tr key={item.id} className="border-b border-slate-800">
                        <td className="p-5">#{item.id.toString().slice(-5)}</td>
                        <td className="p-5">{item.prediction}</td>
                        <td className="p-5">{item.confidence}%</td>
                        <td className="p-5">
                          <button
                            onClick={() => {
                              setResult(item);
                              setPage("Analyze");
                            }}
                            className="text-violet-400"
                          >
                            View result
                          </button>
                        </td>
                      </tr>
                    ))}
                    {history.length === 0 && (
                      <tr>
                        <td colSpan="4" className="p-8 text-center text-slate-500">
                          No analyses yet. Run your first analysis.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* SANITY CHECK */}
          {page === "Sanity Check" && (
            <section>
              <h2 className="mb-3 text-3xl font-bold">Sanity Check Demo</h2>
              <p className="mb-8 text-sm text-slate-400">
                Test the application using three preset examples.
              </p>

              <div className="grid gap-5 md:grid-cols-3">
                {[
                  ["Clearly Real", "Consistent caption and image", "REAL"],
                  ["Clearly Fake", "Possible misleading content", "FAKE"],
                  ["Mismatched", "Caption-image contradiction", "UNCERTAIN"]
                ].map((item, index) => (
                  <div key={item[0]} className="rounded-2xl border border-slate-800 bg-[#101827] p-6">
                    <Activity className="mb-5 text-violet-400" size={27} />
                    <h3 className="mb-2 font-semibold">{item[0]}</h3>
                    <p className="mb-5 text-sm text-slate-400">{item[1]}</p>
                    <p className="mb-5 text-xs text-slate-500">
                      Expected category: {item[2]}
                    </p>
                    <button
                      onClick={() => runDemo(index)}
                      className="w-full rounded-lg bg-violet-600 py-3 text-sm font-semibold hover:bg-violet-500"
                    >
                      Run Demo
                    </button>
                  </div>
                ))}
              </div>
              <p className="mt-5 text-xs text-slate-500">
               Demo cards load sample inputs.
              </p>
            </section>
          )}

          {/* ABOUT PAGE */}
          {page === "About / Ethics" && (
            <section className="max-w-3xl">
              <h2 className="mb-4 text-3xl font-bold">About Multimodal Deepfake Detection</h2>
              <p className="mb-8 leading-7 text-slate-400">
                Multimodal Deepfake Detection is a multimodal content verification prototype
                that analyzes text and images together to identify possible
                misinformation and inconsistencies.
              </p>

              {[
                ["Responsible Use", "The platform is designed to support content verification, not replace human judgment."],
                ["Privacy", "Uploaded files should be handled securely and deleted according to the backend retention policy."],
                ["Limitations", "AI predictions can be wrong. REAL, FAKE and UNCERTAIN labels should not be treated as absolute proof."],
                ["Explainability", "Visual and textual evidence helps users understand which features influenced a prediction."]
              ].map((item) => (
                <div key={item[0]} className="mb-4 rounded-xl border border-slate-800 bg-[#101827] p-6">
                  <h3 className="mb-2 font-semibold text-violet-400">{item[0]}</h3>
                  <p className="text-sm leading-6 text-slate-400">{item[1]}</p>
                </div>
              ))}
            </section>
          )}

        </div>

        <footer className="border-t border-slate-800 px-6 py-6 text-center text-xs text-slate-600">
         Multimodal Deepfake Detection  · Multimodal NLP Deepfake Detection · College Project
        </footer>

      </main>
    </div>
  );
}

export default App;
