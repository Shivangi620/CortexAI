import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Layout } from "./components/Layout.jsx";
import { getInitialRoute } from "./config/routes.js";
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
  const [workspaces, setWorkspaces] = useState([]);
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
  const [trainingRegistryPreview, setTrainingRegistryPreview] = useState(null);
  const [trainMessage, setTrainMessage] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadLoadingLabel, setUploadLoadingLabel] = useState("");
  const [uploadSelectedName, setUploadSelectedName] = useState("");
  const [uploadInspection, setUploadInspection] = useState(null);
  const [sourceInspection, setSourceInspection] = useState(null);
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
    scenarioContext: null,
    scenarioResult: null,
    scenarioMessage: "",
    scenarioPacks: [],
    scenarioPacksMessage: "",
    ensembleResult: null,
    ensembleMessage: "",
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
    uploadSheetName: "",
    uploadSqliteTable: "",
    uploadArchiveMember: "",
    uploadTextChunkSize: 0,
    repairTarget: "",
    targetColumn: "",
    selectedFeatureNames: [],
    trainGoal: "Balanced",
    trainMode: "Balanced",
    trainTaskType: "",
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
    runOriginFilter: "All",
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
    syntheticRowCount: "",
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
    connectorHttpMethod: "GET",
    connectorHeadersJson: "{}",
    connectorBodyJson: "{}",
    connectorSheetName: "",
    connectorSqliteTable: "",
    connectorArchiveMember: "",
    importMode: "Upload File",
    showArchived: false,
    mergeLeftId: "",
    mergeRightId: "",
    mergeLeftKey: "",
    mergeRightKey: "",
    mergeJoinType: "inner",
    ocrText: "",
    scenarioMode: "payload",
    scenarioRowIndex: 0,
    scenarioFilters: "[]",
    scenarioDefinitions: JSON.stringify(
      [
        {
          name: "Upside Case",
          description: "Increase one or two growth levers and compare delta.",
          adjustments: { delta: {} },
        },
        {
          name: "Downside Case",
          description: "Stress-test the riskiest operating assumptions.",
          adjustments: { delta: {} },
        },
      ],
      null,
      2,
    ),
    scenarioSweepFeature: "",
    scenarioSweepValues: "0,1,2,3",
    scenarioPackName: "",
    scenarioPackDescription: "",
    scenarioApprovedIds: "",
    scenarioMaxChangePct: 35,
    scenarioHardBounds: true,
    scenarioBlockedFeatures: "",
    ensembleJobIds: "",
    ensembleStrategy: "voting",
  });

  const [uploadPreview, setUploadPreview] = useState([]);
  const [ingestSummary, setIngestSummary] = useState({});

  const uploadRef = useRef(null);
  const driftUploadRef = useRef(null);
  const retrainUploadRef = useRef(null);
  const contractUploadRef = useRef(null);
  const batchUploadRef = useRef(null);

  const api = useMemo(() => createApiClient(), []);

  const selectedJob = useMemo(() => jobs.find((job) => job.id === selectedJobId) || null, [jobs, selectedJobId]);

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
      const [datasetsPayload, jobsPayload, experimentsPayload, workspacesPayload, metaStatusPayload] =
        await Promise.all([
          api("/api/datasets"),
          api("/api/jobs"),
          api("/api/experiments"),
          api("/api/workspaces"),
          api("/api/meta/status").catch(() => null),
        ]);

      const datasetRows = datasetsPayload.datasets || [];
      const jobRows = jobsPayload || [];
      setDatasets(datasetRows);
      setJobs(jobRows);
      setExperiments(experimentsPayload || []);
      setWorkspaces(workspacesPayload.workspaces || []);
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
        api(
          `/api/leakage/${selectedDatasetId}${forms.targetColumn ? `?target_column=${encodeURIComponent(forms.targetColumn)}` : ""}`,
        ).catch(() => null),
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
          current.repairTarget && profileColumns.includes(current.repairTarget) ? current.repairTarget : nextTarget;
        const nextQuicktrainTarget =
          current.quicktrainTarget && profileColumns.includes(current.quicktrainTarget)
            ? current.quicktrainTarget
            : nextTarget;
        const nextSelectedFeatureNames = Array.isArray(current.selectedFeatureNames)
          ? current.selectedFeatureNames.filter((feature) => profileColumns.includes(feature) && feature !== nextTarget)
          : [];
        const nextQuicktrainSelectedFeatures = Array.isArray(current.quicktrainSelectedFeatures)
          ? current.quicktrainSelectedFeatures.filter(
              (feature) => profileColumns.includes(feature) && feature !== nextQuicktrainTarget,
            )
          : [];
        const nextFutureFeature =
          current.futureFeature && profileColumns.includes(current.futureFeature) ? current.futureFeature : "";

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
      const [
        shapSummary,
        permutationSummary,
        pipelineGraph,
        featureLineage,
        calibration,
        thresholds,
        trustHeatmap,
        recommendations,
        narration,
        driftHistory,
        driftSchedule,
      ] = await Promise.all([
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

  const loadScenarioContext = useCallback(async () => {
    if (!selectedJobId) return;
    try {
      const payload = await api(`/api/scenario/context/${selectedJobId}`);
      setMonitoringState((current) => ({
        ...current,
        scenarioContext: payload,
      }));
    } catch {
      setMonitoringState((current) => ({
        ...current,
        scenarioContext: null,
      }));
    }
  }, [api, selectedJobId]);

  const loadScenarioPacks = useCallback(async () => {
    if (!selectedJobId) return;
    try {
      const payload = await api(`/api/scenario-packs/${selectedJobId}`);
      setMonitoringState((current) => ({
        ...current,
        scenarioPacks: payload.packs || [],
      }));
    } catch (error) {
      setMonitoringState((current) => ({
        ...current,
        scenarioPacks: [],
        scenarioPacksMessage: error.message,
      }));
    }
  }, [api, selectedJobId]);

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
    loadScenarioContext();
  }, [selectedJobId]);

  useEffect(() => {
    loadScenarioPacks();
  }, [selectedJobId]);

  useEffect(() => {
    loadDriftTimeline();
  }, [selectedJobId, forms.driftFeature]);

  useEffect(() => {
    if (!selectedDatasetId) {
      setForecast(null);
      setTrainingRegistryPreview(null);
      return;
    }
    const timer = window.setTimeout(async () => {
      const payload = {
        dataset_id: selectedDatasetId,
        target_column: forms.targetColumn,
        goal: forms.trainGoal,
        mode: forms.trainMode,
        task_type: forms.trainTaskType,
        eval_metric: forms.trainMetric,
        selected_features: forms.selectedFeatureNames,
        auto_clean: forms.trainAutoClean,
        handle_imbalance: forms.trainHandleImbalance,
        cv_folds: Number(forms.trainCvFolds) || 0,
        pca_mode: forms.trainPcaMode,
        pca_components: Number(forms.trainPcaComponents) || 0,
      };
      try {
        const [forecastResult, registryResult] = await Promise.allSettled([
          api("/api/train/forecast", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          }),
          api("/api/train/model-registry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          }),
        ]);
        setForecast(forecastResult.status === "fulfilled" ? forecastResult.value : null);
        setTrainingRegistryPreview(registryResult.status === "fulfilled" ? registryResult.value : null);
      } catch {
        setForecast(null);
        setTrainingRegistryPreview(null);
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [
    api,
    selectedDatasetId,
    forms.targetColumn,
    forms.trainGoal,
    forms.trainMode,
    forms.trainTaskType,
    forms.trainMetric,
    forms.selectedFeatureNames,
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
    formData.append("sheet_name", forms.uploadSheetName);
    formData.append("sqlite_table", forms.uploadSqliteTable);
    formData.append("archive_member", forms.uploadArchiveMember);
    formData.append("text_chunk_size", String(forms.uploadTextChunkSize || 0));
    setUploadLoading(true);
    setUploadLoadingLabel(`Ingesting ${file.name} into Mission Control...`);
    setUploadMessage("");
    try {
      const payload = await api("/api/upload", { method: "POST", body: formData });
      if (payload.error) {
        setUploadMessage(payload.error);
      } else {
        setUploadLoadingLabel("Analyzing the uploaded data and refreshing the workspace...");
        setUploadMessage(`Dataset ready: ${payload.dataset_id}`);
        setUploadPreview(payload.preview_records || []);
        setIngestSummary(payload.ingest_summary || {});
        await refreshCoreData();
        if (payload.dataset_id) setSelectedDatasetId(payload.dataset_id);
        if (payload.imported_job_id) setSelectedJobId(payload.imported_job_id);
      }
    } catch (error) {
      setUploadMessage(error.message);
    } finally {
      setUploadLoading(false);
      setUploadLoadingLabel("");
    }
  }

  async function handleInspectUpload(event) {
    if (event && event.preventDefault) event.preventDefault();
    const file = uploadRef.current?.files?.[0];
    if (!file) {
      setUploadSelectedName("");
      setUploadInspection(null);
      setUploadMessage("Choose a file first.");
      return;
    }
    setUploadSelectedName(file.name || "");
    const formData = new FormData();
    formData.append("file", file);
    setUploadLoading(true);
    setUploadLoadingLabel(`Inspecting ${file.name} for sheets, tables, and archive members...`);
    setUploadMessage("");
    try {
      const payload = await api("/api/inspect-upload", { method: "POST", body: formData });
      if (payload.error) {
        setUploadInspection(null);
        setUploadMessage(payload.error);
      } else {
        setUploadInspection(payload);
        patchForm({
          uploadSheetName: payload.recommended?.sheet_name || "",
          uploadSqliteTable: payload.recommended?.sqlite_table || "",
          uploadArchiveMember: payload.recommended?.archive_member || "",
        });
        setUploadMessage(`Inspection ready for ${file.name}. Adjust options if needed, then ingest.`);
      }
    } catch (error) {
      setUploadInspection(null);
      setUploadMessage(error.message);
    } finally {
      setUploadLoading(false);
      setUploadLoadingLabel("");
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
          http_method: forms.connectorHttpMethod,
          headers_json: forms.connectorHeadersJson,
          body_json: forms.connectorBodyJson,
          sheet_name: forms.connectorSheetName,
          sqlite_table: forms.connectorSqliteTable,
          archive_member: forms.connectorArchiveMember,
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

  async function handleInspectSource(event) {
    if (event && event.preventDefault) event.preventDefault();
    try {
      const payload = await api("/api/inspect-source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: forms.connectorType.toLowerCase(),
          connection_uri: forms.connectorUri,
          http_method: forms.connectorHttpMethod,
          headers_json: forms.connectorHeadersJson,
          body_json: forms.connectorBodyJson,
        }),
      });
      if (payload.error) {
        setSourceInspection(null);
        setUploadMessage(payload.error);
      } else {
        setSourceInspection(payload);
        patchForm({
          connectorSheetName: payload.recommended?.sheet_name || "",
          connectorSqliteTable: payload.recommended?.sqlite_table || "",
          connectorArchiveMember: payload.recommended?.archive_member || "",
        });
        setUploadMessage("Remote source inspected. Review the discovered options before importing.");
      }
    } catch (error) {
      setSourceInspection(null);
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
    await startTrainingForDataset(selectedDatasetId);
  }

  async function startTrainingForDataset(datasetIdOverride, sourceLabel = "") {
    const datasetId = datasetIdOverride || selectedDatasetId;
    if (!datasetId) {
      setTrainMessage("Select a dataset first.");
      return;
    }
    if (!forms.targetColumn) {
      setTrainMessage("Choose a target column before training.");
      setCurrentPath("/overview");
      return;
    }
    try {
      const payload = await api("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          target_column: forms.targetColumn,
          goal: forms.trainGoal,
          mode: forms.trainMode,
          task_type: forms.trainTaskType,
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
      setTrainMessage(payload.error || `${sourceLabel ? `${sourceLabel}: ` : ""}Training started: ${payload.job_id}`);
      if (payload.job_id) {
        setSelectedDatasetId(datasetId);
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
      setTrackingState((current) => ({
        ...current,
        compareResult: null,
        compareMessage: "Select at least one run to compare.",
      }));
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
      setTrackingState((current) => ({
        ...current,
        diffResult: null,
        diffMessage: "Choose two runs before calculating a diff.",
      }));
      return;
    }
    try {
      const payload = await api(
        `/api/experiments/diff?run_a=${encodeURIComponent(forms.diffRunA)}&run_b=${encodeURIComponent(forms.diffRunB)}`,
      );
      setTrackingState((current) => ({ ...current, diffResult: payload, diffMessage: "Diff ready." }));
    } catch (error) {
      setTrackingState((current) => ({ ...current, diffResult: null, diffMessage: error.message }));
    }
  }

  async function fetchNotes() {
    if (!forms.noteEntityType || !forms.noteEntityId) {
      setTrackingState((current) => ({
        ...current,
        notesResult: null,
        notesMessage: "Choose an entity type and ID first.",
      }));
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
      setTrackingState((current) => ({
        ...current,
        notesMessage: "Entity type, entity ID, and note text are required.",
      }));
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
    const retrainPlan = monitoringState.driftResult?.retrain_recommendation || null;
    const formData = new FormData();
    formData.append("file", file);
    if (retrainPlan?.recommended_goal) formData.append("goal_override", retrainPlan.recommended_goal);
    if (retrainPlan?.recommended_mode) formData.append("mode_override", retrainPlan.recommended_mode);
    formData.append(
      "launch_context_json",
      JSON.stringify({
        source: "drift_recommendation",
        source_job_id: selectedJobId,
        message: retrainPlan?.message || "",
        current_model: retrainPlan?.current_model || "",
        historical_winner: retrainPlan?.historical_winner || "",
        candidate_models: retrainPlan?.candidate_models || [],
        memory_confidence: retrainPlan?.memory_confidence ?? 0,
        memory_applied: Boolean(retrainPlan?.memory_applied),
        parent_score: retrainPlan?.current_score ?? null,
        parent_validation_gap: retrainPlan?.current_validation_gap ?? null,
        metric_name: retrainPlan?.metric_name || "",
      }),
    );
    try {
      const payload = await api(`/api/drift/${selectedJobId}/retrain`, { method: "POST", body: formData });
      const launchMessage =
        payload.error ||
        `Retrain launched using ${payload.goal || retrainPlan?.recommended_goal || "Balanced"} / ${payload.mode || retrainPlan?.recommended_mode || "Balanced"}.`;
      setMonitoringState((current) => ({ ...current, retrainMessage: launchMessage }));
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
            typeof forms.baseFeatures === "string" ? forms.baseFeatures : JSON.stringify(forms.baseFeatures || {}),
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

  async function runScenarioSimulator(eventOrPayload) {
    if (eventOrPayload && eventOrPayload.preventDefault) eventOrPayload.preventDefault();
    if (!selectedJobId) {
      setMonitoringState((curr) => ({ ...curr, scenarioMessage: "Select a completed run first." }));
      return;
    }

    setLoading(true);
    try {
      const overridePayload =
        eventOrPayload && !eventOrPayload.preventDefault && typeof eventOrPayload === "object" ? eventOrPayload : null;
      const payload = await api(`/api/scenarios/${selectedJobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_mode: overridePayload?.base_mode || forms.scenarioMode,
          row_index: overridePayload?.row_index ?? (Number(forms.scenarioRowIndex) || 0),
          base_payload:
            overridePayload?.base_payload ||
            safeParseJson(
              typeof forms.predictPayload === "string"
                ? forms.predictPayload
                : JSON.stringify(forms.predictPayload || {}),
              {},
            ),
          filters: overridePayload?.filters || safeParseJson(forms.scenarioFilters, []),
          scenarios: overridePayload?.scenarios || safeParseJson(forms.scenarioDefinitions, []),
          sweep_feature: overridePayload?.sweep_feature || forms.scenarioSweepFeature,
          sweep_values: overridePayload?.sweep_values || toCsvArray(forms.scenarioSweepValues),
          approval_policy: {
            max_numeric_delta_ratio: Math.max(parseFloat(forms.scenarioMaxChangePct || 35) || 35, 0) / 100,
            hard_bounds: Boolean(forms.scenarioHardBounds),
            blocked_features: toCsvArray(forms.scenarioBlockedFeatures),
          },
          approved_scenarios: toCsvArray(forms.scenarioApprovedIds),
          enforce_guardrails: true,
        }),
      });
      const reviewCount = Number(payload?.guardrail_summary?.review_required || 0);
      const blockedCount = Number(payload?.guardrail_summary?.blocked || 0);
      setMonitoringState((curr) => ({
        ...curr,
        scenarioResult: payload,
        scenarioMessage:
          blockedCount > 0
            ? `${blockedCount} scenario${blockedCount === 1 ? "" : "s"} blocked by guardrails.${reviewCount ? ` ${reviewCount} still need approval.` : ""}`
            : reviewCount > 0
              ? `${reviewCount} scenario${reviewCount === 1 ? "" : "s"} need approval before execution.`
              : "Scenario simulator updated in realtime.",
      }));
    } catch (error) {
      setMonitoringState((curr) => ({
        ...curr,
        scenarioResult: null,
        scenarioMessage: error.message,
      }));
    } finally {
      setLoading(false);
    }
  }

  async function saveScenarioPack() {
    if (!selectedJobId) {
      setMonitoringState((curr) => ({ ...curr, scenarioPacksMessage: "Select a completed run first." }));
      return;
    }

    try {
      const payload = await api(`/api/scenario-packs/${selectedJobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: forms.scenarioPackName,
          description: forms.scenarioPackDescription,
          base_mode: forms.scenarioMode,
          row_index: forms.scenarioMode === "row" ? Number(forms.scenarioRowIndex) || 0 : null,
          base_payload: safeParseJson(
            typeof forms.predictPayload === "string"
              ? forms.predictPayload
              : JSON.stringify(forms.predictPayload || {}),
            {},
          ),
          filters: safeParseJson(forms.scenarioFilters, []),
          scenarios: safeParseJson(forms.scenarioDefinitions, []),
          sweep_feature: forms.scenarioSweepFeature,
          sweep_values: toCsvArray(forms.scenarioSweepValues),
          approval_policy: {
            max_numeric_delta_ratio: Math.max(parseFloat(forms.scenarioMaxChangePct || 35) || 35, 0) / 100,
            hard_bounds: Boolean(forms.scenarioHardBounds),
            blocked_features: toCsvArray(forms.scenarioBlockedFeatures),
          },
          approved_scenarios: toCsvArray(forms.scenarioApprovedIds),
        }),
      });
      setMonitoringState((curr) => ({
        ...curr,
        scenarioPacksMessage: `Saved pack "${payload.name}".`,
      }));
      await loadScenarioPacks();
    } catch (error) {
      setMonitoringState((curr) => ({
        ...curr,
        scenarioPacksMessage: error.message,
      }));
    }
  }

  function applyScenarioPack(pack) {
    if (!pack) return;
    patchForm({
      scenarioPackName: pack.name || "",
      scenarioPackDescription: pack.description || "",
      scenarioMode: pack.base_mode || "payload",
      scenarioRowIndex: pack.row_index ?? 0,
      predictPayload: pack.base_payload || {},
      scenarioFilters: JSON.stringify(pack.filters || [], null, 2),
      scenarioDefinitions: JSON.stringify(pack.scenarios || [], null, 2),
      scenarioSweepFeature: pack.sweep_feature || "",
      scenarioSweepValues: Array.isArray(pack.sweep_values) ? pack.sweep_values.join(",") : "",
      scenarioMaxChangePct: pack.approval_policy?.max_numeric_delta_ratio
        ? String(Math.round(Number(pack.approval_policy.max_numeric_delta_ratio) * 100))
        : "35",
      scenarioHardBounds: pack.approval_policy?.hard_bounds !== false,
      scenarioBlockedFeatures: Array.isArray(pack.approval_policy?.blocked_features)
        ? pack.approval_policy.blocked_features.join(",")
        : "",
      scenarioApprovedIds: Array.isArray(pack.approved_scenarios) ? pack.approved_scenarios.join(",") : "",
    });
    setMonitoringState((curr) => ({
      ...curr,
      scenarioPacksMessage: `Loaded pack "${pack.name}".`,
    }));
  }

  function applySuggestedScenarios() {
    const suggestions = monitoringState.scenarioResult?.auto_suggestions || [];
    if (!suggestions.length) return;
    patchForm(
      "scenarioDefinitions",
      JSON.stringify(
        suggestions.map((item) => item.scenario),
        null,
        2,
      ),
    );
    setMonitoringState((curr) => ({
      ...curr,
      scenarioMessage: "Applied auto-generated uplift and low-risk scenarios.",
    }));
  }

  function approveScenarioReviews() {
    const ids = (monitoringState.scenarioResult?.scenarios || [])
      .filter((item) => item.approval_status === "review_required")
      .map((item) => item.id)
      .filter(Boolean);
    patchForm("scenarioApprovedIds", ids.join(","));
    setMonitoringState((curr) => ({
      ...curr,
      scenarioMessage: ids.length
        ? "Review-required scenarios marked approved. Run the simulator again to execute them."
        : "No review approvals are pending.",
    }));
  }

  async function buildEnsemble() {
    const selectedIds = Array.from(
      new Set(toCsvArray(forms.ensembleJobIds).concat(selectedJobId ? [selectedJobId] : [])),
    ).filter(Boolean);

    if (selectedIds.length < 2) {
      setMonitoringState((curr) => ({
        ...curr,
        ensembleMessage: "Pick at least two completed jobs for the ensemble.",
      }));
      return;
    }

    setLoading(true);
    try {
      const payload = await api("/api/ensemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_ids: selectedIds,
          strategy: forms.ensembleStrategy,
          dataset_id: selectedDatasetId || undefined,
        }),
      });
      setMonitoringState((curr) => ({
        ...curr,
        ensembleResult: payload.error ? null : payload,
        ensembleMessage: payload.error || "Ensemble built and registered as a completed run.",
      }));
      if (payload.job_id) {
        setSelectedJobId(payload.job_id);
      }
      await refreshCoreData();
    } catch (error) {
      setMonitoringState((curr) => ({
        ...curr,
        ensembleResult: null,
        ensembleMessage: error.message,
      }));
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
    if (!selectedDatasetId) {
      patchTools("synthetic", "Select a dataset first.");
      return;
    }
    setLoading(true);
    try {
      const requestedRows = String(forms.syntheticRowCount || "").trim();
      const query = requestedRows ? `?n_rows=${encodeURIComponent(requestedRows)}` : "";
      const payload = await api(`/api/synthetic/${selectedDatasetId}${query}`, { method: "POST" });
      if (payload.error) {
        patchTools("synthetic", payload.error, payload);
        return;
      }

      let mergedPayload = payload;
      let message = "Synthetic dataset created.";

      if (payload.new_dataset_id) {
        setSelectedDatasetId(payload.new_dataset_id);
        await refreshCoreData();

        try {
          const judgePayload = await api(`/api/synthetic/judge/${payload.new_dataset_id}`);
          mergedPayload = {
            ...payload,
            ...judgePayload,
          };
          message = judgePayload.error
            ? `Synthetic dataset created, but judge review is unavailable: ${judgePayload.error}`
            : "Synthetic dataset created and judged automatically.";
        } catch (judgeError) {
          message = `Synthetic dataset created, but judge review failed: ${judgeError.message}`;
        }
      }

      patchTools("synthetic", message, mergedPayload);
    } catch (error) {
      patchTools("synthetic", error.message);
    } finally {
      setLoading(false);
    }
  }

  async function syntheticJudge() {
    if (!selectedDatasetId) {
      patchTools("synthetic", "Select a dataset first.");
      return;
    }
    try {
      const payload = await api(`/api/synthetic/judge/${selectedDatasetId}`);
      setToolsState((current) => ({
        ...current,
        messages: { ...current.messages, synthetic: "Synthetic judge completed." },
        outputs: {
          ...current.outputs,
          synthetic: {
            ...(current.outputs.synthetic || {}),
            ...payload,
          },
        },
      }));
    } catch (error) {
      patchTools("synthetic", error.message);
    }
  }

  async function trainSyntheticDataset() {
    const syntheticDatasetId = toolsState.outputs.synthetic?.new_dataset_id || selectedDatasetId;
    await startTrainingForDataset(syntheticDatasetId, "Synthetic dataset");
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
    if (!String(forms.nlPrompt || "").trim()) {
      patchTools("intent", "Describe the ML goal first.");
      return;
    }
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
    if (!selectedJobId) {
      patchTools("chat", "Select a run first.");
      return;
    }
    if (!String(forms.chatPrompt || "").trim()) {
      patchTools("chat", "Ask a question first.");
      return;
    }
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
        jobs={jobs}
        selectedDatasetId={selectedDatasetId}
        datasetProfile={datasetProfile}
        datasetDetect={datasetDetect}
        loading={loading}
        uploadLoading={uploadLoading}
        uploadLoadingLabel={uploadLoadingLabel}
        forecast={forecast}
        trainingRegistryPreview={trainingRegistryPreview}
        forms={forms}
        patchForm={patchForm}
        uploadRef={uploadRef}
        uploadSelectedName={uploadSelectedName}
        uploadInspection={uploadInspection}
        sourceInspection={sourceInspection}
        handleInspectUpload={handleInspectUpload}
        handleUpload={handleUpload}
        uploadMessage={uploadMessage}
        uploadPreview={uploadPreview}
        ingestSummary={ingestSummary}
        handleInspectSource={handleInspectSource}
        handleImportSource={handleImportSource}
        handleTrain={handleTrain}
        trainMessage={trainMessage}
        handleOcrReview={handleOcrReview}
      />
    );
  } else if (currentPath === "/data") {
    page = (
      <DataPage
        datasets={datasets}
        datasetProfile={datasetProfile}
        datasetHealth={datasetHealth}
        datasetDetect={datasetDetect}
        datasetLeakage={datasetLeakage}
        datasetLineage={datasetLineage}
        datasetVersions={datasetVersions}
        loading={loading}
        forms={forms}
        patchForm={patchForm}
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
        forecast={forecast}
        trainingRegistryPreview={trainingRegistryPreview}
        jobs={jobs}
        jobStatus={jobStatus}
      />
    );
  } else if (currentPath === "/results") {
    page = (
      <ResultsPage
        results={jobStatus?.results}
        launchContext={jobStatus?.config?.launch_context}
        jobs={jobs}
        selectedJobId={selectedJobId}
        scenarioResult={monitoringState.scenarioResult}
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
        trainSyntheticDataset={trainSyntheticDataset}
        zeroShot={zeroShot}
        metaInsights={metaInsights}
        parseIntent={parseIntent}
        chat={chat}
        uploadRefs={{ contractUploadRef, batchUploadRef }}
        messages={toolsState.messages}
        outputs={toolsState.outputs}
        metaStatus={toolsState.metaStatus}
        selectedDatasetId={selectedDatasetId}
        datasetProfile={datasetProfile}
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
        scenarioContext={monitoringState.scenarioContext}
        loadScenarioContext={loadScenarioContext}
        runScenarioSimulator={runScenarioSimulator}
        scenarioResult={monitoringState.scenarioResult}
        scenarioMessage={monitoringState.scenarioMessage}
        scenarioPacks={monitoringState.scenarioPacks}
        scenarioPacksMessage={monitoringState.scenarioPacksMessage}
        saveScenarioPack={saveScenarioPack}
        applyScenarioPack={applyScenarioPack}
        applySuggestedScenarios={applySuggestedScenarios}
        approveScenarioReviews={approveScenarioReviews}
        goalSeeker={monitoringState.goalSeeker}
        goalSeekerMessage={monitoringState.goalSeekerMessage}
        getGoalSeeker={getGoalSeeker}
        buildEnsemble={buildEnsemble}
        ensembleResult={monitoringState.ensembleResult}
        ensembleMessage={monitoringState.ensembleMessage}
        jobs={jobs}
        isClassification={jobStatus?.results?.is_classification}
      />
    );
  }

  return (
    <Layout currentPath={currentPath} datasets={datasets} jobs={jobs} theme={theme} toggleTheme={toggleTheme}>
      {globalError ? <div className="message message--warning">{globalError}</div> : null}
      {page}
    </Layout>
  );
}
