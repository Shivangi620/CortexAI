import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Layout } from "./components/Layout.jsx";
import { ROUTES, getInitialRoute } from "./config/routes.js";
import { createApiClient } from "./lib/api.js";
import { safeParseJson, toCsvArray } from "./lib/format.js";
import { usePolling } from "./hooks/usePolling.js";
import { OverviewPage } from "./pages/OverviewPage.jsx";
import { DataPage } from "./pages/DataPage.jsx";
import { TrainingPage } from "./pages/TrainingPage.jsx";
import { ResultsPage } from "./pages/ResultsPage.jsx";
import { TrackingPage } from "./pages/TrackingPage.jsx";
import { MonitoringPage } from "./pages/MonitoringPage.jsx";
import { ToolsPage } from "./pages/ToolsPage.jsx";

const STORAGE_KEYS = {
  datasetId: "codin_dataset_id",
  jobId: "codin_job_id",
  theme: "codin_theme",
};

export function App() {
  const [currentPath, setCurrentPath] = useState(getInitialRoute);

  const [theme, setTheme] = useState(() => localStorage.getItem(STORAGE_KEYS.theme) || "dark");
  const [loading, setLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [workspaces, setWorkspaces] = useState([]);
  const [workspaceLatest, setWorkspaceLatest] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState(() => localStorage.getItem(STORAGE_KEYS.datasetId) || "");
  const [selectedJobId, setSelectedJobId] = useState(() => localStorage.getItem(STORAGE_KEYS.jobId) || "");

  const [datasetProfile, setDatasetProfile] = useState(null);
  const [datasetHealth, setDatasetHealth] = useState(null);
  const [datasetDetect, setDatasetDetect] = useState(null);
  const [datasetLeakage, setDatasetLeakage] = useState(null);
  const [datasetTimeline, setDatasetTimeline] = useState(null);
  const [datasetLineage, setDatasetLineage] = useState(null);
  const [datasetVersions, setDatasetVersions] = useState(null);

  const [forecast, setForecast] = useState(null);
  const [trainMessage, setTrainMessage] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [repairMessage, setRepairMessage] = useState("");

  const [jobStatus, setJobStatus] = useState(null);
  const [resultsAssets, setResultsAssets] = useState({
    shapSummary: null,
    permutationSummary: null,
    pipelineGraph: null,
    featureLineage: null,
    calibration: null,
    thresholds: null,
    trustHeatmap: null,
    recommendations: null,
    narration: null,
  });

  const [trackingState, setTrackingState] = useState({
    compareMessage: "",
    diffMessage: "",
    notesMessage: "",
    compareResult: null,
    diffResult: null,
    notesResult: null,
  });

  const [monitoringState, setMonitoringState] = useState({
    driftResult: null,
    driftHistory: null,
    driftTimeline: null,
    driftSchedule: null,
    goalSeeker: null,
    goalSeekerMessage: "",
    detectMessage: "",
    scheduleMessage: "",
    retrainMessage: "",
    predictResult: null,
    predictMessage: "",
    futureResult: null,
    futureMessage: "",
  });

  const [toolsState, setToolsState] = useState({
    messages: {},
    outputs: {},
    metaStatus: null,
  });

  const [mergeState, setMergeState] = useState({
    preview: null,
    message: "",
  });

  const [forms, setForms] = useState({
    uploadPdfMode: "text",
    repairTarget: "",
    targetColumn: "",
    selectedFeatureNames: [],
    trainGoal: "Balanced",
    trainMode: "Balanced",
    trainMetric: "",
    trainCvFolds: 5,
    trainPcaMode: "auto",
    trainPcaComponents: 0,
    trainAutoClean: true,
    trainHandleImbalance: false,
    compareExperimentIds: "",
    diffRunA: "",
    diffRunB: "",
    noteEntityType: "job",
    noteEntityId: "",
    noteText: "",
    driftEnabled: true,
    driftFrequencyDays: 7,
    driftWarningThreshold: 0.1,
    driftCriticalThreshold: 0.2,
    driftFeature: "",
    predictPayload: {},
    baseFeatures: {},
    sweepFeature: "",
    goalSeekTarget: "",
    predictJson: "{}",
    futureBaseJson: "{}",
    futureFeature: "",
    futureValues: "0,1,2",
    quicktrainTarget: "",
    quicktrainSelectedFeatures: [],
    quicktrainModels: "Decision Tree,Random Forest,Logistic Regression",
    nlPrompt: "",
    chatPrompt: "",
    workspaceName: "",
    trainPreset: "Balanced",
    exportModel: true,
    exportCode: true,
    exportReport: true,
    connectorType: "PostgreSQL",
    connectorUri: "",
    connectorQuery: "SELECT * FROM your_table LIMIT 5000",
    importMode: "Upload File",
    showArchived: false,
    mergeLeftId: "",
    mergeRightId: "",
    mergeLeftKey: "",
    mergeRightKey: "",
    mergeJoinType: "inner",
    ocrText: "",
  });

  const [uploadPreview, setUploadPreview] = useState([]);
  const [ingestSummary, setIngestSummary] = useState({});

  const uploadRef = useRef(null);
  const driftUploadRef = useRef(null);
  const retrainUploadRef = useRef(null);
  const contractUploadRef = useRef(null);
  const batchUploadRef = useRef(null);

  const api = useMemo(() => createApiClient(), []);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === selectedDatasetId) || null,
    [datasets, selectedDatasetId]
  );

  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) || null,
    [jobs, selectedJobId]
  );

  const overviewStats = useMemo(() => {
    const completed = jobs.filter((job) => job.status === "completed").length;
    const training = jobs.filter((job) => job.status === "training").length;
    return { completed, training };
  }, [jobs]);

  const patchForm = useCallback((keyOrObject, value) => {
    if (typeof keyOrObject === "object" && keyOrObject !== null) {
      setForms((current) => ({ ...current, ...keyOrObject }));
    } else {
      setForms((current) => ({ ...current, [keyOrObject]: value }));
    }
  }, []);

  const patchTools = useCallback((messageKey, message, output) => {
    setToolsState((current) => ({
      ...current,
      messages: { ...current.messages, [messageKey]: message },
      outputs: output === undefined ? current.outputs : { ...current.outputs, [messageKey]: output },
    }));
  }, []);

  const refreshCoreData = useCallback(async () => {
    setLoading(true);
    setGlobalError("");
    try {
      const [datasetsPayload, jobsPayload, experimentsPayload, notificationsPayload, workspacesPayload, latestWorkspacePayload, metaStatusPayload] =
        await Promise.all([
          api("/api/datasets"),
          api("/api/jobs"),
          api("/api/experiments"),
          api("/api/notifications"),
          api("/api/workspaces"),
          api("/api/workspace/latest").catch(() => null),
          api("/api/meta/status").catch(() => null),
        ]);

      const datasetRows = datasetsPayload.datasets || [];
      const jobRows = jobsPayload || [];
      setDatasets(datasetRows);
      setJobs(jobRows);
      setExperiments(experimentsPayload || []);
      setNotifications(notificationsPayload.notifications || []);
      setWorkspaces(workspacesPayload.workspaces || []);
      setWorkspaceLatest(latestWorkspacePayload);
      setToolsState((current) => ({ ...current, metaStatus: metaStatusPayload }));

      if (!selectedDatasetId && datasetRows.length) setSelectedDatasetId(datasetRows[0].id);
      if (!selectedJobId && jobRows.length) setSelectedJobId(jobRows[0].id);
    } catch (error) {
      setGlobalError(error.message);
    } finally {
      setLoading(false);
    }
  }, [api, selectedDatasetId, selectedJobId]);

  const loadDatasetContext = useCallback(async () => {
    if (!selectedDatasetId) return;
    setLoading(true);
    try {
      const [profile, health, detect, leakage, timeline, lineage, versions] = await Promise.all([
        api(`/api/dataset/${selectedDatasetId}`),
        api(`/api/health/${selectedDatasetId}`).catch(() => null),
        api("/api/detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset_id: selectedDatasetId, target_column: forms.targetColumn }),
        }).catch(() => null),
        api(`/api/leakage/${selectedDatasetId}${forms.targetColumn ? `?target_column=${encodeURIComponent(forms.targetColumn)}` : ""}`).catch(() => null),
        api(`/api/dataset/${selectedDatasetId}/timeline`).catch(() => null),
        api(`/api/dataset/${selectedDatasetId}/lineage-graph`).catch(() => null),
        api(`/api/dataset/${selectedDatasetId}/versions`).catch(() => null),
      ]);
      setDatasetProfile(profile);
      setDatasetHealth(health);
      setDatasetDetect(detect);
      setDatasetLeakage(leakage);
      setDatasetTimeline(timeline);
      setDatasetLineage(lineage);
      setDatasetVersions(versions);

      const profileColumns = profile?.columns || [];
      const suggestedTarget = profile?.suggested_target || "";
      setForms((current) => {
        const currentTargetIsValid = current.targetColumn && profileColumns.includes(current.targetColumn);
        const nextTarget = currentTargetIsValid ? current.targetColumn : suggestedTarget || "";
        const nextRepairTarget =
          current.repairTarget && profileColumns.includes(current.repairTarget)
            ? current.repairTarget
            : nextTarget;
        const nextQuicktrainTarget =
          current.quicktrainTarget && profileColumns.includes(current.quicktrainTarget)
            ? current.quicktrainTarget
            : nextTarget;
        const nextSelectedFeatureNames = Array.isArray(current.selectedFeatureNames)
          ? current.selectedFeatureNames.filter((feature) => profileColumns.includes(feature) && feature !== nextTarget)
          : [];
        const nextQuicktrainSelectedFeatures = Array.isArray(current.quicktrainSelectedFeatures)
          ? current.quicktrainSelectedFeatures.filter((feature) => profileColumns.includes(feature) && feature !== nextQuicktrainTarget)
          : [];
        const nextFutureFeature =
          current.futureFeature && profileColumns.includes(current.futureFeature)
            ? current.futureFeature
            : "";

        if (
          nextTarget === current.targetColumn &&
          nextRepairTarget === current.repairTarget &&
          nextQuicktrainTarget === current.quicktrainTarget &&
          nextFutureFeature === current.futureFeature &&
          JSON.stringify(nextSelectedFeatureNames) === JSON.stringify(current.selectedFeatureNames || []) &&
          JSON.stringify(nextQuicktrainSelectedFeatures) === JSON.stringify(current.quicktrainSelectedFeatures || [])
        ) {
          return current;
        }

        return {
          ...current,
          targetColumn: nextTarget,
          repairTarget: nextRepairTarget,
          quicktrainTarget: nextQuicktrainTarget,
          selectedFeatureNames: nextSelectedFeatureNames,
          quicktrainSelectedFeatures: nextQuicktrainSelectedFeatures,
          futureFeature: nextFutureFeature,
        };
      });
    } catch (error) {
      setGlobalError(error.message);
    } finally {
      setLoading(false);
    }
  }, [api, selectedDatasetId, forms.targetColumn]);

  const loadJobContext = useCallback(async () => {
    if (!selectedJobId) return;
    try {
      const status = await api(`/api/status/${selectedJobId}`);
      const isCompleted = status?.status === "completed";
      const [shapSummary, permutationSummary, pipelineGraph, featureLineage, calibration, thresholds, trustHeatmap, recommendations, narration, driftHistory, driftSchedule] =
        await Promise.all([
          isCompleted ? api(`/api/shap/${selectedJobId}`).catch(() => null) : Promise.resolve(null),
          isCompleted ? api(`/api/permutation/${selectedJobId}`).catch(() => null) : Promise.resolve(null),
          api(`/api/pipeline/${selectedJobId}`).catch(() => null),
          api(`/api/lineage/${selectedJobId}`).catch(() => null),
          api(`/api/calibration/${selectedJobId}`).catch(() => null),
          api(`/api/thresholds/${selectedJobId}`).catch(() => null),
          api(`/api/trust/${selectedJobId}`).catch(() => null),
          isCompleted ? api(`/api/recommend/${selectedJobId}`).catch(() => null) : Promise.resolve(null),
          isCompleted ? api(`/api/narrate/${selectedJobId}`).catch(() => null) : Promise.resolve(null),
          api(`/api/drift/${selectedJobId}/history`).catch(() => null),
          api(`/api/drift/${selectedJobId}/schedule`).catch(() => null),
        ]);

      setJobStatus(status);
      setResultsAssets({
        shapSummary,
        permutationSummary,
        pipelineGraph,
        featureLineage,
        calibration,
        thresholds,
        trustHeatmap,
        recommendations,
        narration,
      });
      setMonitoringState((current) => ({
        ...current,
        driftHistory,
        driftSchedule,
      }));
      if (driftSchedule) {
        setForms((current) => ({
          ...current,
          driftEnabled: Boolean(driftSchedule.enabled),
          driftFrequencyDays: driftSchedule.frequency_days ?? current.driftFrequencyDays,
          driftWarningThreshold: driftSchedule.warning_threshold ?? current.driftWarningThreshold,
          driftCriticalThreshold: driftSchedule.critical_threshold ?? current.driftCriticalThreshold,
        }));
      }
    } catch (error) {
      setGlobalError(error.message);
    }
  }, [api, selectedJobId]);

  const loadDriftTimeline = useCallback(async () => {
    if (!selectedJobId) return;
    try {
      const suffix = forms.driftFeature ? `?feature=${encodeURIComponent(forms.driftFeature)}` : "";
      const payload = await api(`/api/drift/${selectedJobId}/feature-timeline${suffix}`);
      setMonitoringState((current) => ({ ...current, driftTimeline: payload }));
    } catch {
      // keep quiet for filtered timeline misses
    }
  }, [api, selectedJobId, forms.driftFeature]);

  useEffect(() => {
    const onPopState = () => setCurrentPath(getInitialRoute());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (window.location.pathname !== currentPath) {
      window.history.replaceState({}, "", currentPath);
    }
  }, [currentPath]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.datasetId, selectedDatasetId || "");
  }, [selectedDatasetId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.jobId, selectedJobId || "");
  }, [selectedJobId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.theme, theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    refreshCoreData();
  }, []);

  useEffect(() => {
    loadDatasetContext();
  }, [selectedDatasetId, forms.targetColumn]);

  useEffect(() => {
    loadJobContext();
  }, [selectedJobId]);

  useEffect(() => {
    loadDriftTimeline();
  }, [selectedJobId, forms.driftFeature]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    const timer = window.setTimeout(async () => {
      try {
        const payload = await api("/api/train/forecast", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_id: selectedDatasetId,
            target_column: forms.targetColumn,
            goal: forms.trainGoal,
            mode: forms.trainMode,
            eval_metric: forms.trainMetric,
            selected_features: forms.selectedFeatureNames,
            auto_clean: forms.trainAutoClean,
            handle_imbalance: forms.trainHandleImbalance,
            cv_folds: Number(forms.trainCvFolds) || 0,
            pca_mode: forms.trainPcaMode,
            pca_components: Number(forms.trainPcaComponents) || 0,
          }),
        });
        setForecast(payload);
      } catch {
        setForecast(null);
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [
    api,
    selectedDatasetId,
    forms.targetColumn,
    forms.trainGoal,
    forms.trainMode,
    forms.trainMetric,
    forms.trainAutoClean,
    forms.trainHandleImbalance,
    forms.trainCvFolds,
    forms.trainPcaMode,
    forms.trainPcaComponents,
  ]);

  usePolling(refreshCoreData, 6000, true);
  usePolling(loadJobContext, selectedJob?.status === "training" ? 2500 : 0, Boolean(selectedJobId));

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);


  async function handleUpload(event) {
    if (event && event.preventDefault) event.preventDefault();
    patchForm("ocrText", "");
    const file = uploadRef.current?.files?.[0];
    if (!file) {
      setUploadMessage("Choose a file first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("pdf_mode", forms.uploadPdfMode);
    try {
      const payload = await api("/api/upload", { method: "POST", body: formData });
      if (payload.error) {
        setUploadMessage(payload.error);
      } else {
        setUploadMessage(`Dataset ready: ${payload.dataset_id}`);
        setUploadPreview(payload.preview_records || []);
        setIngestSummary(payload.ingest_summary || {});
        await refreshCoreData();
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
        if (payload.imported_job_id) setSelectedJobId(payload.imported_job_id);
      }
    } catch (error) {
      setUploadMessage(error.message);
    }
  }

  async function handleImportSource(event) {
    event.preventDefault();
    patchForm("ocrText", "");
    try {
      const payload = await api("/api/import-source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: forms.connectorType.toLowerCase(),
          connection_uri: forms.connectorUri,
          query: forms.connectorQuery,
        }),
      });
      if (payload.error) {
        setUploadMessage(payload.error);
      } else {
        setUploadMessage(`Imported dataset: ${payload.dataset_id}`);
        setUploadPreview(payload.preview_records || []);
        setIngestSummary(payload.ingest_summary || {});
        await refreshCoreData();
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
      }
    } catch (error) {
      setUploadMessage(error.message);
    }
  }

  async function handleArchive(datasetId, archive = true) {
    try {
      await api(`/api/dataset/${datasetId}/${archive ? "archive" : "unarchive"}`, { method: "POST" });
      await refreshCoreData();
    } catch (error) {
      setGlobalError(`Archive failed: ${error.message}`);
    }
  }

  async function handleDeleteDataset(datasetId) {
    try {
      await api(`/api/dataset/${datasetId}`, { method: "DELETE" });
      if (selectedDatasetId === datasetId) setSelectedDatasetId("");
      await refreshCoreData();
    } catch (error) {
      setGlobalError(`Delete failed: ${error.message}`);
    }
  }

  async function handleResumeLastRun() {
    try {
      const payload = await api("/api/workspaces/resume");
      if (payload.job_id) {
        setSelectedJobId(payload.job_id);
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
        setGlobalMessage("Resumed last run context.");
      } else {
        setGlobalError(payload.error || "No run available to resume.");
      }
    } catch (error) {
      setGlobalError(`Resume failed: ${error.message}`);
    }
  }

  async function repairPreview() {
    if (!selectedDatasetId) {
      setRepairMessage("Select a dataset first.");
      return;
    }
    setLoading(true);
    try {
      const payload = await api("/api/repair-preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          target_column: forms.repairTarget || forms.targetColumn,
        }),
      });
      setRepairMessage(payload.error || `Repair preview: ${payload.after_rows} rows after cleaning.`);
    } catch (error) {
      setRepairMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function repairApply() {
    if (!selectedDatasetId) {
      setRepairMessage("Select a dataset first.");
      return;
    }
    setLoading(true);
    try {
      const payload = await api("/api/repair-apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          target_column: forms.repairTarget || forms.targetColumn,
        }),
      });
      setRepairMessage(payload.error || `Repaired dataset created: ${payload.dataset_id}`);
      if (payload.dataset_id) {
        await refreshCoreData();
        setSelectedDatasetId(payload.dataset_id);
      }
    } catch (error) {
      setRepairMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleOcrReview() {
    if (!selectedDatasetId) return;
    setLoading(true);
    try {
      const payload = await api(`/api/dataset/${selectedDatasetId}/ocr-review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: forms.ocrText }),
      });
      if (payload.error) {
        setUploadMessage(payload.error);
      } else {
        setUploadMessage("OCR-reviewed dataset created.");
        setUploadPreview(payload.preview_records || []);
        setIngestSummary(payload.ingest_summary || {});
        await refreshCoreData();
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
      }
    } catch (error) {
      setUploadMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleTrain(event) {
    event.preventDefault();
    if (!selectedDatasetId) {
      setTrainMessage("Select a dataset first.");
      return;
    }
    try {
      const payload = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          target_column: forms.targetColumn,
          goal: forms.trainGoal,
          mode: forms.trainMode,
          eval_metric: forms.trainMetric,
          selected_features: forms.selectedFeatureNames,
          auto_clean: forms.trainAutoClean,
          handle_imbalance: forms.trainHandleImbalance,
          cv_folds: Number(forms.trainCvFolds) || 0,
          pca_mode: forms.trainPcaMode,
          pca_components: Number(forms.trainPcaComponents) || 0,
          workspace_name: forms.workspaceName,
          preset_name: forms.trainPreset,
          export_model: forms.exportModel,
          export_code: forms.exportCode,
          export_report: forms.exportReport,
        }),
      });
      setTrainMessage(payload.error || `Training started: ${payload.job_id}`);
      if (payload.job_id) {
        setSelectedJobId(payload.job_id);
        setCurrentPath("/training");
        await refreshCoreData();
      }
    } catch (error) {
      setTrainMessage(error.message);
    }
  }

  async function compareExperiments() {
    const ids = toCsvArray(forms.compareExperimentIds);
    if (!ids.length) {
      setTrackingState((current) => ({ ...current, compareResult: null, compareMessage: "Select at least one run to compare." }));
      return;
    }
    try {
      const payload = await api(`/api/experiments/compare?ids=${encodeURIComponent(ids.join(","))}`);
      setTrackingState((current) => ({ ...current, compareResult: payload, compareMessage: "Comparison ready." }));
    } catch (error) {
      setTrackingState((current) => ({ ...current, compareResult: null, compareMessage: error.message }));
    }
  }

  async function diffExperiments() {
    if (!forms.diffRunA || !forms.diffRunB) {
      setTrackingState((current) => ({ ...current, diffResult: null, diffMessage: "Choose two runs before calculating a diff." }));
      return;
    }
    try {
      const payload = await api(`/api/experiments/diff?run_a=${encodeURIComponent(forms.diffRunA)}&run_b=${encodeURIComponent(forms.diffRunB)}`);
      setTrackingState((current) => ({ ...current, diffResult: payload, diffMessage: "Diff ready." }));
    } catch (error) {
      setTrackingState((current) => ({ ...current, diffResult: null, diffMessage: error.message }));
    }
  }

  async function fetchNotes() {
    if (!forms.noteEntityType || !forms.noteEntityId) {
      setTrackingState((current) => ({ ...current, notesResult: null, notesMessage: "Choose an entity type and ID first." }));
      return;
    }
    try {
      const payload = await api(`/api/notes/${forms.noteEntityType}/${forms.noteEntityId}`);
      setTrackingState((current) => ({ ...current, notesResult: payload, notesMessage: "Notes loaded." }));
    } catch (error) {
      setTrackingState((current) => ({ ...current, notesResult: null, notesMessage: error.message }));
    }
  }

  async function saveNote(event) {
    event.preventDefault();
    if (!forms.noteEntityType || !forms.noteEntityId || !String(forms.noteText || "").trim()) {
      setTrackingState((current) => ({ ...current, notesMessage: "Entity type, entity ID, and note text are required." }));
      return;
    }
    try {
      await api(`/api/notes/${forms.noteEntityType}/${forms.noteEntityId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note: forms.noteText }),
      });
      setTrackingState((current) => ({ ...current, notesMessage: "Note saved." }));
      fetchNotes();
    } catch (error) {
      setTrackingState((current) => ({ ...current, notesMessage: error.message }));
    }
  }

  async function driftDetect(event) {
    event.preventDefault();
    const file = driftUploadRef.current?.files?.[0];
    if (!selectedJobId || !file) {
      setMonitoringState((current) => ({ ...current, detectMessage: "Select a job and file first." }));
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const payload = await api(`/api/drift/${selectedJobId}`, { method: "POST", body: formData });
      setMonitoringState((current) => ({ ...current, driftResult: payload, detectMessage: "Drift report completed." }));
      await loadJobContext();
      await loadDriftTimeline();
    } catch (error) {
      setMonitoringState((current) => ({ ...current, detectMessage: error.message }));
    }
  }

  async function driftScheduleSave(event) {
    event.preventDefault();
    if (!selectedJobId) {
      setMonitoringState((current) => ({ ...current, scheduleMessage: "Select a run before saving a policy." }));
      return;
    }
    try {
      const params = new URLSearchParams({
        enabled: String(Boolean(forms.driftEnabled)),
        frequency_days: String(forms.driftFrequencyDays),
        warning_threshold: String(forms.driftWarningThreshold),
        critical_threshold: String(forms.driftCriticalThreshold),
      });
      const payload = await api(`/api/drift/${selectedJobId}/schedule?${params.toString()}`, { method: "POST" });
      setMonitoringState((current) => ({ ...current, driftSchedule: payload, scheduleMessage: "Schedule saved." }));
    } catch (error) {
      setMonitoringState((current) => ({ ...current, scheduleMessage: error.message }));
    }
  }

  async function driftRetrain(event) {
    event.preventDefault();
    const file = retrainUploadRef.current?.files?.[0];
    if (!selectedJobId || !file) {
      setMonitoringState((current) => ({ ...current, retrainMessage: "Select a job and file first." }));
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const payload = await api(`/api/drift/${selectedJobId}/retrain`, { method: "POST", body: formData });
      setMonitoringState((current) => ({ ...current, retrainMessage: payload.error || "Retrain launched." }));
      if (payload.job_id) setSelectedJobId(payload.job_id);
      if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
      await refreshCoreData();
    } catch (error) {
      setMonitoringState((current) => ({ ...current, retrainMessage: error.message }));
    }
  }

  async function handlePredict(event, overridePayload = null) {
    if (event && event.preventDefault) event.preventDefault();
    if (!selectedJobId) {
      setMonitoringState((curr) => ({ ...curr, predictMessage: "Select a run first." }));
      return;
    }
    setLoading(true);
    try {
      const payload = await api(`/api/predict/${selectedJobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features:
            overridePayload && typeof overridePayload === "object"
              ? overridePayload
              : safeParseJson(
                  typeof forms.predictPayload === "string"
                    ? forms.predictPayload
                    : JSON.stringify(forms.predictPayload || {}),
                  {},
                ),
        }),
      });
      setMonitoringState((curr) => ({ ...curr, predictResult: payload, predictMessage: "Prediction ready." }));
    } catch (error) {
      setMonitoringState((curr) => ({ ...curr, predictMessage: error.message }));
    } finally {
      setLoading(false);
    }
  }

  async function handleFutureSweep(event) {
    if (event && event.preventDefault) event.preventDefault();
    if (!selectedJobId) {
      setMonitoringState((curr) => ({ ...curr, futureMessage: "Select a run first." }));
      return;
    }
    setLoading(true);
    try {
      const payload = await api("/api/future", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: selectedJobId,
          base_features: safeParseJson(
            typeof forms.baseFeatures === "string"
              ? forms.baseFeatures
              : JSON.stringify(forms.baseFeatures || {}),
            {},
          ),
          sweep_feature: forms.sweepFeature,
          sweep_values: toCsvArray(forms.futureValues),
        }),
      });
      setMonitoringState((curr) => ({ ...curr, futureResult: payload, futureMessage: "Future sweep ready." }));
    } catch (error) {
      setMonitoringState((curr) => ({ ...curr, futureMessage: error.message }));
    } finally {
      setLoading(false);
    }
  }

  async function predict(event) {
    event.preventDefault();
    try {
      const payload = await api(`/api/predict/${selectedJobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: safeParseJson(forms.predictPayload, {}) }),
      });
      patchTools("predict", "Prediction ready.", payload);
    } catch (error) {
      patchTools("predict", error.message);
    }
  }

  async function futurePredict(event) {
    event.preventDefault();
    try {
      const payload = await api("/api/future", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: selectedJobId,
          base_features: safeParseJson(forms.baseFeatures, {}),
          sweep_feature: forms.sweepFeature,
          sweep_values: toCsvArray(forms.futureValues),
        }),
      });
      patchTools("future", "Future sweep ready.", payload);
    } catch (error) {
      patchTools("future", error.message);
    }
  }

  async function contractCheck(event) {
    event.preventDefault();
    const file = contractUploadRef.current?.files?.[0];
    if (!selectedJobId || !file) {
      patchTools("contractCheck", "Select a job and file first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    try {
      const payload = await api(`/api/contract-check/${selectedJobId}`, { method: "POST", body: formData });
      patchTools("contractCheck", "Contract check ready.", payload);
    } catch (error) {
      patchTools("contractCheck", error.message);
    }
  }

  async function batchPredict(event) {
    event.preventDefault();
    const file = batchUploadRef.current?.files?.[0];
    if (!selectedJobId || !file) {
      patchTools("batchPredict", "Select a job and file first.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const payload = await api(`/api/batch-predict/${selectedJobId}`, { method: "POST", body: formData });
      patchTools("batchPredict", "Batch inference completed.", payload);
    } catch (error) {
      patchTools("batchPredict", error.message);
    } finally {
      setLoading(false);
    }
  }

  async function getGoalSeeker() {
    if (!selectedJobId) {
      setMonitoringState((curr) => ({ ...curr, goalSeekerMessage: "Select a run first." }));
      return;
    }
    setLoading(true);
    try {
      const payload = await api(`/api/counterfactual/${selectedJobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          payload: safeParseJson(
            typeof forms.predictPayload === "string"
              ? forms.predictPayload
              : JSON.stringify(forms.predictPayload || {}),
            {},
          ),
          target_prediction: parseFloat(forms.goalSeekTarget) || 0,
        }),
      });
      setMonitoringState((curr) => ({
        ...curr,
        goalSeeker: payload,
        goalSeekerMessage: payload.message || "Goal seeker analysis ready.",
      }));
    } catch (error) {
      setMonitoringState((curr) => ({
        ...curr,
        goalSeeker: { error: error.message },
        goalSeekerMessage: error.message,
      }));
    } finally {
      setLoading(false);
    }
  }

  function downloadModelCard() {
    if (!selectedJobId) return;
    window.open(`/api/report/${selectedJobId}/model-card`, "_blank");
  }

  async function quicktrain(event) {
    event.preventDefault();
    try {
      const payload = await api("/api/quicktrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: selectedDatasetId,
          target_column: forms.quicktrainTarget || forms.targetColumn,
          selected_features: forms.quicktrainSelectedFeatures,
          selected_models: toCsvArray(forms.quicktrainModels),
        }),
      });
      patchTools("quicktrain", "Quicktrain completed.", payload);
    } catch (error) {
      patchTools("quicktrain", error.message);
    }
  }

  async function syntheticExpand() {
    try {
      const payload = await api(`/api/synthetic/${selectedDatasetId}`, { method: "POST" });
      patchTools("synthetic", payload.error || "Synthetic dataset created.", payload);
      if (payload.new_dataset_id) {
        refreshCoreData();
        setSelectedDatasetId(payload.new_dataset_id);
      }
    } catch (error) {
      patchTools("synthetic", error.message);
    }
  }

  async function syntheticJudge() {
    try {
      const payload = await api(`/api/synthetic/judge/${selectedDatasetId}`);
      patchTools("synthetic", "Synthetic judge completed.", payload);
    } catch (error) {
      patchTools("synthetic", error.message);
    }
  }

  async function zeroShot() {
    try {
      const payload = await api(`/api/zeroshot/${selectedDatasetId}`);
      patchTools("zeroshot", "Zero-shot recommendation ready.", payload);
    } catch (error) {
      patchTools("zeroshot", error.message);
    }
  }

  async function metaInsights() {
    try {
      const payload = await api(`/api/meta/insights/${selectedDatasetId}`);
      patchTools("metaInsights", "Meta insights ready.", payload);
    } catch (error) {
      patchTools("metaInsights", error.message);
    }
  }

  async function parseIntent(event) {
    event.preventDefault();
    try {
      const payload = await api("/api/nl/intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: forms.nlPrompt, dataset_id: selectedDatasetId }),
      });
      patchTools("intent", "Intent parsed.", payload);
    } catch (error) {
      patchTools("intent", error.message);
    }
  }

  async function chat(event) {
    event.preventDefault();
    try {
      const payload = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: selectedJobId, prompt: forms.chatPrompt }),
      });
      patchTools("chat", "Chat reply ready.", payload);
    } catch (error) {
      patchTools("chat", error.message);
    }
  }

  async function runLeakageScan() {
    if (!selectedDatasetId) return;
    setLoading(true);
    try {
      const url = `/api/leakage/${selectedDatasetId}${forms.targetColumn ? `?target_column=${encodeURIComponent(forms.targetColumn)}` : ""}`;
      const payload = await api(url);
      setDatasetLeakage(payload);
    } catch (error) {
      setGlobalError(`Leakage scan failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleMergePreview() {
    if (!forms.mergeLeftId || !forms.mergeRightId || !forms.mergeLeftKey || !forms.mergeRightKey) {
      setMergeState((curr) => ({ ...curr, message: "Select both datasets and keys first." }));
      return;
    }
    setLoading(true);
    try {
      const payload = await api("/api/merge-studio/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_dataset_id: forms.mergeLeftId,
          right_dataset_id: forms.mergeRightId,
          join_key_left: forms.mergeLeftKey,
          join_key_right: forms.mergeRightKey,
          join_type: forms.mergeJoinType,
        }),
      });
      setMergeState({ preview: payload, message: payload.error || "Join preview ready." });
    } catch (error) {
      setMergeState((curr) => ({ ...curr, message: error.message }));
    } finally {
      setLoading(false);
    }
  }

  async function handleMergeApply() {
    if (!forms.mergeLeftId || !forms.mergeRightId || !forms.mergeLeftKey || !forms.mergeRightKey) {
      setMergeState((curr) => ({ ...curr, message: "Select both datasets and keys first." }));
      return;
    }
    setLoading(true);
    try {
      const payload = await api("/api/merge-studio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          left_dataset_id: forms.mergeLeftId,
          right_dataset_id: forms.mergeRightId,
          join_key_left: forms.mergeLeftKey,
          join_key_right: forms.mergeRightKey,
          join_type: forms.mergeJoinType,
        }),
      });
      if (payload.error) {
        setMergeState((curr) => ({ ...curr, message: payload.error }));
      } else {
        setMergeState({ preview: null, message: "Datasets merged successfully." });
        await refreshCoreData();
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
      }
    } catch (error) {
      setMergeState((curr) => ({ ...curr, message: error.message }));
    } finally {
      setLoading(false);
    }
  }

  let page = null;
  if (currentPath === "/overview") {
    page = (
      <OverviewPage
        notifications={notifications}
        jobs={jobs}
        experiments={experiments}
        workspaces={workspaces}
        overviewStats={overviewStats}
        selectedJob={selectedJob}
        workspaceLatest={workspaceLatest}
        datasets={datasets}
        selectedDatasetId={selectedDatasetId}
        setSelectedDatasetId={setSelectedDatasetId}
        datasetProfile={datasetProfile}
        datasetDetect={datasetDetect}
        loading={loading}
        forecast={forecast}
        forms={forms}
        patchForm={patchForm}
        uploadRef={uploadRef}
        handleUpload={handleUpload}
        uploadMessage={uploadMessage}
        uploadPreview={uploadPreview}
        ingestSummary={ingestSummary}
        handleImportSource={handleImportSource}
        handleArchive={handleArchive}
        handleDeleteDataset={handleDeleteDataset}
        handleResumeLastRun={handleResumeLastRun}
        handleTrain={handleTrain}
        trainMessage={trainMessage}
        handleMergePreview={handleMergePreview}
        handleMergeApply={handleMergeApply}
        mergeState={mergeState}
        handleOcrReview={handleOcrReview}
      />
    );

  } else if (currentPath === "/data") {
    page = (
      <DataPage
        datasets={datasets}
        selectedDataset={selectedDataset}
        datasetProfile={datasetProfile}
        datasetHealth={datasetHealth}
        datasetDetect={datasetDetect}
        datasetLeakage={datasetLeakage}
        datasetTimeline={datasetTimeline}
        datasetLineage={datasetLineage}
        datasetVersions={datasetVersions}
        loading={loading}
        forms={forms}
        patchForm={patchForm}
        uploadRef={uploadRef}
        handleUpload={handleUpload}
        handleImportSource={handleImportSource}
        uploadMessage={uploadMessage}
        repairPreview={repairPreview}
        repairApply={repairApply}
        repairMessage={repairMessage}
        runLeakageScan={runLeakageScan}
        handleMergePreview={handleMergePreview}
        handleMergeApply={handleMergeApply}
        mergeState={mergeState}
      />
    );
  } else if (currentPath === "/training") {
    page = (
      <TrainingPage
        forms={forms}
        patchForm={patchForm}
        forecast={forecast}
        trainMessage={trainMessage}
        handleTrain={handleTrain}
        jobs={jobs}
        jobStatus={jobStatus}
        datasetColumns={datasetProfile?.columns || []}
      />
    );
  } else if (currentPath === "/results") {
    page = (
      <ResultsPage
        results={jobStatus?.results}
        selectedJobId={selectedJobId}
        {...resultsAssets}
      />
    );
  } else if (currentPath === "/tracking") {
    page = (
      <TrackingPage
        experiments={experiments}
        workspaces={workspaces}
        forms={forms}
        patchForm={patchForm}
        compareExperiments={compareExperiments}
        diffExperiments={diffExperiments}
        fetchNotes={fetchNotes}
        saveNote={saveNote}
        datasets={datasets}
        selectedDatasetId={selectedDatasetId}
        setSelectedDatasetId={setSelectedDatasetId}
        handleArchive={handleArchive}
        handleDeleteDataset={handleDeleteDataset}
        datasetLineage={datasetLineage}
        datasetTimeline={datasetTimeline}
        datasetVersions={datasetVersions}
        {...trackingState}
      />
    );
  } else if (currentPath === "/monitoring") {
    page = (
      <MonitoringPage
        driftUploadRef={driftUploadRef}
        retrainUploadRef={retrainUploadRef}
        driftDetect={driftDetect}
        driftScheduleSave={driftScheduleSave}
        driftRetrain={driftRetrain}
        handlePredict={handlePredict}
        getGoalSeeker={getGoalSeeker}
        goalSeeker={monitoringState.goalSeeker}
        handleFutureSweep={handleFutureSweep}
        loading={loading}
        forms={forms}
        patchForm={patchForm}
        datasetProfile={datasetProfile}
        {...monitoringState}
      />
    );
  } else {
    page = (
      <ToolsPage
        forms={forms}
        patchForm={patchForm}
        predict={predict}
        futurePredict={futurePredict}
        contractCheck={contractCheck}
        batchPredict={batchPredict}
        downloadModelCard={downloadModelCard}
        quicktrain={quicktrain}
        syntheticExpand={syntheticExpand}
        syntheticJudge={syntheticJudge}
        zeroShot={zeroShot}
        metaInsights={metaInsights}
        parseIntent={parseIntent}
        chat={chat}
        uploadRefs={{ contractUploadRef, batchUploadRef }}
        messages={toolsState.messages}
        outputs={toolsState.outputs}
        metaStatus={toolsState.metaStatus}
        datasetColumns={datasetProfile?.columns || []}
        trainedFeatureNames={jobStatus?.results?.feature_names || []}
        selectedJobId={selectedJobId}
        loading={loading}
        predictResult={monitoringState.predictResult}
        predictMessage={monitoringState.predictMessage}
        handlePredict={handlePredict}
        futureResult={monitoringState.futureResult}
        futureMessage={monitoringState.futureMessage}
        handleFutureSweep={handleFutureSweep}
        goalSeeker={monitoringState.goalSeeker}
        goalSeekerMessage={monitoringState.goalSeekerMessage}
        getGoalSeeker={getGoalSeeker}
        isClassification={jobStatus?.results?.is_classification}
      />
    );
  }

  return (
    <Layout
      currentPath={currentPath}
      datasets={datasets}
      jobs={jobs}
      selectedDataset={selectedDataset}
      selectedJob={selectedJob}
      selectedDatasetId={selectedDatasetId}
      selectedJobId={selectedJobId}
      setSelectedDatasetId={setSelectedDatasetId}
      setSelectedJobId={setSelectedJobId}
      refresh={refreshCoreData}
      loading={loading}
      metaStatus={toolsState.metaStatus}
      theme={theme}
      toggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
    >
      {globalError ? <div className="message message--warning">{globalError}</div> : null}
      {page}
    </Layout>
  );
}
