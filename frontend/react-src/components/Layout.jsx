import React from "react";
import { ROUTES, navigateTo } from "../config/routes.js";
import { Badge } from "./ui.jsx";

export function Layout({ currentPath, datasets, jobs, theme, toggleTheme, children }) {
  const current = ROUTES.find((route) => route.path === currentPath) || ROUTES[0];
  const completedJobs = jobs.filter((job) => job.status === "completed").length;

  return (
    <div className="studio-shell">
      <div className="studio-shell__ambient studio-shell__ambient--one" />
      <div className="studio-shell__ambient studio-shell__ambient--two" />

      <aside className="studio-sidebar">
        <div className="brand-card">
          <span className="brand-card__label">Inferyx Neural Studio</span>
          <h1>Inferyx</h1>
          <p>
            AutoML orchestration for ingestion, training, simulation, monitoring, and export of machine learning models.
          </p>
          <div className="brand-card__tags">
            <Badge tone="success">{datasets.length} datasets</Badge>
            <Badge>{jobs.length} jobs</Badge>
            <Badge tone="warning">{completedJobs} completed</Badge>
          </div>
        </div>

        <nav className="studio-nav">
          {ROUTES.map((route) => (
            <button
              key={route.path}
              type="button"
              className={`studio-nav__item ${route.path === currentPath ? "studio-nav__item--active" : ""}`}
              onClick={() => navigateTo(route.path)}
            >
              <span className="studio-nav__step">{route.step}</span>
              <span>
                <strong>{route.label}</strong>
                <small>{route.description}</small>
              </span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="studio-main">
        <header className="studio-topbar">
          <div>
            <span className="eyebrow">{current.step} / Studio Flow</span>
            <h2>{current.label}</h2>
            <p>{current.description}</p>
          </div>
          <div className="studio-topbar__actions">
            <div className="status-pill">
              <span>Theme</span>
              <strong>{theme === "dark" ? "Dark" : "Light"}</strong>
            </div>
            <button type="button" className="button button--secondary" onClick={toggleTheme}>
              Toggle Theme
            </button>
          </div>
        </header>
        {children}
      </main>
    </div>
  );
}
