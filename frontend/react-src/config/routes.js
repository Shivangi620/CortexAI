export const ROUTES = [
  {
    path: "/overview",
    label: "Mission Control",
    step: "01",
    description: "Workspace, auth, live ops pulse, recent activity",
  },
  { path: "/data", label: "Data DNA", step: "02", description: "Ingest, QA, target detection, repair, lineage" },
  {
    path: "/training",
    label: "Training Deck",
    step: "03",
    description: "Forecast, configure, launch, watch reasoning",
  },
  { path: "/results", label: "Decision Room", step: "04", description: "Scores, explainability, trust, exports" },
  { path: "/tracking", label: "History", step: "05", description: "Experiments, diffs, notes, workspaces" },
  { path: "/monitoring", label: "Ops Watch", step: "06", description: "Drift checks, thresholds, retraining" },
  { path: "/tools", label: "Advanced Lab", step: "07", description: "Inference, synthetic data, quicktrain, AI tools" },
];

export function getInitialRoute() {
  const current = window.location.pathname;
  if (ROUTES.some((route) => route.path === current)) return current;
  return "/overview";
}

export function navigateTo(path) {
  if (window.location.pathname !== path) {
    window.history.pushState({}, "", path);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
}
