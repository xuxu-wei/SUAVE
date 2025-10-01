"""Browser-based helper for interactive schema editing.

The utilities in this module are optional and only activated when the
Flask dependency is available. They provide a lightweight web interface
that runs locally, making it possible to review and adjust the schema
suggested by the SchemaInferencer.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..schema_inference import (
    SCHEMA_INFERENCE_MODES,
    SchemaInferenceResult,
    SchemaInferencer,
)
from ..types import Schema

try:  # pragma: no cover - optional dependency
    from flask import Flask, jsonify, request
    from werkzeug.serving import make_server
except Exception:  # pragma: no cover - executed only when Flask is absent
    Flask = None  # type: ignore
    jsonify = None  # type: ignore
    request = None  # type: ignore
    make_server = None  # type: ignore


_SUPPORTED_TYPES: tuple[str, ...] = ("real", "pos", "count", "cat", "ordinal")
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


class SchemaBuilderError(RuntimeError):
    """Error raised when the interactive schema builder cannot start."""


@dataclass
class _ColumnState:
    """JSON serialisable representation of a single column."""

    name: str
    type: str
    note: str
    dtype: str
    nunique: int
    missing: int
    sample: List[object]
    n_classes: Optional[int]
    y_dim: Optional[int]
    confidence: str
    position: int
    flagged: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "type": self.type,
            "note": self.note,
            "dtype": self.dtype,
            "nunique": self.nunique,
            "missing": self.missing,
            "sample": self.sample,
            "n_classes": self.n_classes,
            "y_dim": self.y_dim,
            "confidence": self.confidence,
            "position": self.position,
            "flagged": self.flagged,
        }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SUAVE Schema Builder</title>
<style>
:root {
    color-scheme: light;
    font-family: "Segoe UI", Arial, sans-serif;
}
body {
    margin: 0;
    background: #f5f5f5;
    color: #1f1f1f;
}
main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
}
header {
    margin-bottom: 24px;
}
h1 {
    margin: 0 0 8px;
    font-size: 1.8rem;
}
p.lead {
    margin: 0 0 16px;
    color: #333;
}
.table-wrapper {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    padding: 16px;
    overflow: visible;
}
.table-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    font-size: 0.95rem;
}
.table-controls label {
    display: flex;
    align-items: center;
    gap: 8px;
}
.table-controls select {
    padding: 4px 8px;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.2);
    background: #ffffff;
}
.sort-button {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
}
.table-scroll {
    overflow-x: auto;
}
.table-scroll table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 10px;
    vertical-align: top;
}
th {
    background: #fafafa;
    position: sticky;
    top: 0;
    z-index: 1;
}
tbody tr[data-confidence="low"] td {
    background-color: #FBBDB8;
}
tbody tr[data-confidence="medium"] td {
    background-color: #FFE483;
}
tbody tr[data-confidence="high"] td {
    background-color: #AAD781;
}
tbody tr td {
    transition: background-color 0.2s ease-in-out;
}
tbody tr:hover td {
    box-shadow: inset 0 0 0 9999px rgba(0, 0, 0, 0.05);
}
button.primary {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 10px 18px;
    font-size: 1rem;
    cursor: pointer;
}
button.primary[disabled] {
    opacity: 0.6;
    cursor: wait;
}
button.secondary {
    padding: 6px 12px;
    border: 1px solid #1f6feb;
    background: #ffffff;
    color: #1f6feb;
    border-radius: 4px;
    cursor: pointer;
}
#messages .message {
    margin-bottom: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    background: #eef3ff;
    border: 1px solid #d0defb;
}
#messages .message.error {
    background: #fde7e9;
    border-color: #f5b0b7;
}
.legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    font-size: 0.9rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.legend-color {
    display: inline-block;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.15);
}
.tooltip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #1f6feb;
    color: #ffffff;
    font-size: 0.8rem;
    font-weight: bold;
    margin-left: 6px;
    position: relative;
    cursor: pointer;
}
.tooltip.alert {
    background: #d93025;
}
.tooltip-content {
    display: none;
}
.tooltip-floating {
    position: fixed;
    z-index: 2000;
    max-width: min(320px, 90vw);
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
    padding: 12px 14px;
    color: #1f1f1f;
    font-size: 0.85rem;
    line-height: 1.4;
    display: none;
}
.tooltip-floating div + div {
    margin-top: 6px;
}
.column-name {
    display: flex;
    align-items: center;
    gap: 6px;
}
.summary-note {
    margin-top: 4px;
    color: #5f6368;
    font-size: 0.82rem;
}
.modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    z-index: 3000;
}
.modal-overlay[hidden] {
    display: none;
}
.modal-content {
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.2);
    max-width: min(520px, 95vw);
    width: 100%;
    padding: 24px;
    max-height: 90vh;
    overflow-y: auto;
}
.modal-content h2 {
    margin-top: 0;
    margin-bottom: 8px;
}
.modal-content p {
    margin-top: 0;
}
.summary-list {
    margin-top: 12px;
}
.summary-entry + .summary-entry {
    margin-top: 12px;
}
.summary-entry-title {
    font-weight: 600;
}
.summary-entry-body {
    margin-top: 4px;
    font-size: 0.9rem;
    color: #424242;
}
.code-container {
    margin-top: 16px;
}
.code-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #5f6368;
    margin-bottom: 6px;
}
.code-block {
    background: #1f1f1f;
    color: #f5f5f5;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: "SFMono-Regular", "Consolas", "Liberation Mono", monospace;
    font-size: 0.85rem;
    white-space: pre;
    overflow-x: auto;
}
.modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-top: 24px;
    flex-wrap: wrap;
}
.is-hidden {
    display: none !important;
}
</style>
</head>
<body>
<main>
  <header>
    <h1>SUAVE Schema Builder</h1>
    <p class="lead">Review the automatically inferred schema, adjust column types as needed, and click <strong>Save schema</strong> when you are satisfied.</p>
    <div class="legend">
      <div class="legend-item"><span class="legend-color" style="background:#FBBDB8;"></span><span>Low confidence (please double-check)</span></div>
      <div class="legend-item"><span class="legend-color" style="background:#FFE483;"></span><span>Medium confidence (review recommended)</span></div>
      <div class="legend-item"><span class="legend-color" style="background:#AAD781;"></span><span>High confidence</span></div>
    </div>
  </header>
  <div id="messages"></div>
  <div class="table-wrapper">
    <div class="table-controls">
      <label for="sort-key">Sort by
        <select id="sort-key">
          <option value="position">Column order</option>
          <option value="confidence">Confidence</option>
          <option value="flagged">Flagged status</option>
        </select>
      </label>
      <button id="sort-direction" class="secondary sort-button" type="button">Asc ↑</button>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Column</th>
            <th>Type <span class="tooltip" tabindex="0">?<span class="tooltip-content">
              <div><strong>real</strong> Continuous numeric feature modelled with a Gaussian mean/variance; values can take any real number. Example: z-scored blood pressure with an approximately bell-shaped spread.</div>
              <div><strong>pos</strong> Non-negative continuous feature (zero allowed) transformed with log1p and modelled as log-normal, so training data must stay above -1 and generated samples remain near that range. Example: right-skewed lab results such as lactate.</div>
              <div><strong>count</strong> Non-negative integer feature with a Poisson rate; fit and sampling cover 0, 1, 2, … with no preset upper bound. Example: number of prior hospital admissions.</div>
              <div><strong>cat</strong> Unordered categorical feature; the decoder learns independent class probabilities for each level. Example: ward identifier or blood type.</div>
              <div><strong>ordinal</strong> Ordered categorical feature using cumulative logit thresholds across a fixed number of ranks (0 to K-1), so samples never exceed the declared highest level—unlike counts, which assume an unbounded integer range. Example: triage acuity scores.</div>
            </span></span></th>
            <th>n_classes</th>
            <th>y_dim</th>
            <th>Distribution</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody id="schema-body"></tbody>
      </table>
    </div>
  </div>
  <div style="margin-top:16px;">
    <button id="finalize" class="primary">Save schema</button>
  </div>
  <div
    id="summary-modal"
    class="modal-overlay"
    role="dialog"
    aria-modal="true"
    aria-labelledby="summary-title"
    tabindex="-1"
    hidden
  >
    <div class="modal-content" role="document">
      <h2 id="summary-title">Schema saved</h2>
      <p id="summary-intro">Here is a summary of the adjustments you made.</p>
      <div id="summary-details" class="summary-list"></div>
      <div id="summary-code-container" class="code-container is-hidden">
        <div class="code-label">Python snippet</div>
        <pre id="summary-code" class="code-block"></pre>
      </div>
      <div class="modal-actions">
        <button id="copy-summary" class="secondary" type="button">Copy code</button>
        <button id="close-summary" class="primary" type="button">Close</button>
      </div>
    </div>
  </div>
</main>
<script>
const TYPE_OPTIONS = ["real", "pos", "count", "cat", "ordinal"];
const CONFIDENCE_ORDER = { low: 0, medium: 1, high: 2 };
let cancellationNotified = false;
let currentColumns = [];
let sortConfig = { key: "position", direction: "asc" };
let floatingTooltip = null;
let activeTooltipTarget = null;
const summaryModal = document.getElementById("summary-modal");
const summaryIntro = document.getElementById("summary-intro");
const summaryDetails = document.getElementById("summary-details");
const summaryCodeContainer = document.getElementById("summary-code-container");
const summaryCodeBlock = document.getElementById("summary-code");
const copySummaryButton = document.getElementById("copy-summary");
const closeSummaryButton = document.getElementById("close-summary");
let latestPythonSnippet = "";
let previousBodyOverflow = "";

const sortKeySelect = document.getElementById("sort-key");
const sortDirectionButton = document.getElementById("sort-direction");

function ensureFloatingTooltip() {
    if (!floatingTooltip) {
        floatingTooltip = document.createElement("div");
        floatingTooltip.className = "tooltip-floating";
        document.body.appendChild(floatingTooltip);
    }
    return floatingTooltip;
}

function updateSortDirectionButton() {
    if (!sortDirectionButton) {
        return;
    }
    const label = sortConfig.direction === "asc" ? "Asc ↑" : "Desc ↓";
    sortDirectionButton.textContent = label;
    sortDirectionButton.setAttribute(
        "aria-label",
        sortConfig.direction === "asc" ? "Ascending order" : "Descending order",
    );
}

function formatSummaryValue(value) {
    if (value === undefined) {
        return "—";
    }
    if (value === null) {
        return "null";
    }
    if (typeof value === "object") {
        return JSON.stringify(value);
    }
    return String(value);
}

function describeChange(change) {
    const before = change.before || {};
    const after = change.after || {};
    const keys = new Set([
        ...Object.keys(before || {}),
        ...Object.keys(after || {}),
    ]);
    const details = [];
    keys.forEach((key) => {
        const beforeValue = before ? before[key] : undefined;
        const afterValue = after ? after[key] : undefined;
        if (JSON.stringify(beforeValue) === JSON.stringify(afterValue)) {
            return;
        }
        details.push(
            `${key}: ${formatSummaryValue(beforeValue)} → ${formatSummaryValue(afterValue)}`,
        );
    });
    return details.join("; ");
}

function clearSummaryModal() {
    if (summaryDetails) {
        summaryDetails.innerHTML = "";
    }
    if (summaryCodeBlock) {
        summaryCodeBlock.textContent = "";
    }
    if (copySummaryButton) {
        copySummaryButton.textContent = "Copy code";
    }
}

function openSummaryModal(payload) {
    if (!summaryModal) {
        return;
    }
    clearSummaryModal();
    const hasPayload = typeof payload === "object" && payload !== null;
    const changes = hasPayload && Array.isArray(payload.changes) ? payload.changes : [];
    latestPythonSnippet = hasPayload && typeof payload.python === "string" ? payload.python : "";

    if (summaryIntro) {
        summaryIntro.textContent = changes.length
            ? "Here is a summary of the adjustments you made."
            : "No columns were modified. The inferred schema was saved as-is.";
    }

    if (summaryDetails && changes.length) {
        changes.forEach((change) => {
            const container = document.createElement("div");
            container.className = "summary-entry";
            const title = document.createElement("div");
            title.className = "summary-entry-title";
            title.textContent = change.column;
            container.appendChild(title);
            const body = document.createElement("div");
            body.className = "summary-entry-body";
            body.textContent = describeChange(change) || "Column updated.";
            container.appendChild(body);
            summaryDetails.appendChild(container);
        });
    }

    if (summaryCodeContainer) {
        if (latestPythonSnippet) {
            summaryCodeContainer.classList.remove("is-hidden");
        } else {
            summaryCodeContainer.classList.add("is-hidden");
        }
    }

    if (summaryCodeBlock && latestPythonSnippet) {
        summaryCodeBlock.textContent = latestPythonSnippet;
    }

    if (copySummaryButton) {
        copySummaryButton.disabled = latestPythonSnippet.length === 0;
        copySummaryButton.textContent = "Copy code";
    }

    previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    summaryModal.removeAttribute("hidden");
    summaryModal.focus({ preventScroll: true });
    if (closeSummaryButton) {
        closeSummaryButton.focus({ preventScroll: true });
    }
}

function closeSummaryModal() {
    if (!summaryModal) {
        return;
    }
    summaryModal.setAttribute("hidden", "hidden");
    document.body.style.overflow = previousBodyOverflow;
}

async function copySummaryCode() {
    if (!latestPythonSnippet) {
        return;
    }
    if (!copySummaryButton) {
        return;
    }
    try {
        if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
            await navigator.clipboard.writeText(latestPythonSnippet);
        } else {
            const helper = document.createElement("textarea");
            helper.value = latestPythonSnippet;
            helper.setAttribute("readonly", "");
            helper.style.position = "fixed";
            helper.style.opacity = "0";
            document.body.appendChild(helper);
            helper.select();
            document.execCommand("copy");
            document.body.removeChild(helper);
        }
        copySummaryButton.textContent = "Copied!";
        setTimeout(() => {
            if (copySummaryButton) {
                copySummaryButton.textContent = "Copy code";
            }
        }, 2000);
    } catch (error) {
        alert("Unable to copy code automatically. Please copy it manually.");
    }
}

if (sortKeySelect) {
    sortKeySelect.addEventListener("change", () => {
        sortConfig.key = sortKeySelect.value;
        refreshTable();
    });
}

if (sortDirectionButton) {
    sortDirectionButton.addEventListener("click", () => {
        sortConfig.direction = sortConfig.direction === "asc" ? "desc" : "asc";
        updateSortDirectionButton();
        refreshTable();
    });
    updateSortDirectionButton();
}

if (copySummaryButton) {
    copySummaryButton.addEventListener("click", () => {
        void copySummaryCode();
    });
}

if (closeSummaryButton) {
    closeSummaryButton.addEventListener("click", () => {
        closeSummaryModal();
    });
}

if (summaryModal) {
    summaryModal.addEventListener("click", (event) => {
        if (event.target === summaryModal) {
            closeSummaryModal();
        }
    });
}

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && summaryModal && !summaryModal.hasAttribute("hidden")) {
        closeSummaryModal();
    }
});

function createOption(option, selected) {
    const choice = document.createElement("option");
    choice.value = option;
    choice.textContent = option;
    if (option === selected) {
        choice.selected = true;
    }
    return choice;
}

function formatSampleValue(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
        const rounded = Math.round(value * 100) / 100;
        if (Number.isInteger(rounded)) {
            return String(Object.is(rounded, -0) ? 0 : rounded);
        }
        return (Object.is(rounded, -0) ? 0 : rounded)
            .toFixed(2)
            .replace(/0+$/, "")
            .replace(/\.$/, "");
    }
    if (value === null) {
        return "null";
    }
    if (typeof value === "string") {
        return JSON.stringify(value);
    }
    return JSON.stringify(value);
}

function formatSampleArray(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return "[]";
    }
    const formatted = values.map((value) => formatSampleValue(value));
    return `[${formatted.join(", ")}]`;
}

function renderMessages(messages) {
    const container = document.getElementById("messages");
    container.innerHTML = "";
    if (!messages.length) {
        return;
    }
    messages.forEach((msg) => {
        const div = document.createElement("div");
        div.className = "message";
        div.textContent = msg;
        container.appendChild(div);
    });
}

function showError(message) {
    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message error";
    div.textContent = message;
    container.appendChild(div);
}

async function loadState() {
    try {
        const response = await fetch("/api/state");
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        const payload = await response.json();
        renderMessages(payload.messages || []);
        setColumns(payload.columns || []);
    } catch (error) {
        showError(`Unable to load initial state: ${error.message}`);
    }
}

function setColumns(columns) {
    currentColumns = (columns || []).map((column, index) => ({
        ...column,
        position:
            typeof column.position === "number"
                ? column.position
                : index,
        confidence: column.confidence || "high",
        flagged: Boolean(column.flagged),
    }));
    refreshTable();
}

function applyConfidence(row, confidence) {
    const level = (confidence || "high").toLowerCase();
    row.dataset.confidence = ["low", "medium", "high"].includes(level) ? level : "high";
}

function syncNClassState(select, nClassesInput) {
    const needsClasses = select.value === "cat" || select.value === "ordinal";
    nClassesInput.disabled = !needsClasses;
    if (!needsClasses) {
        nClassesInput.value = "";
    }
}

function getSortValue(column) {
    switch (sortConfig.key) {
        case "confidence": {
            const key = String(column.confidence || "").toLowerCase();
            return key in CONFIDENCE_ORDER ? CONFIDENCE_ORDER[key] : Number.MAX_SAFE_INTEGER;
        }
        case "flagged":
            return column.flagged ? 1 : 0;
        case "position":
        default:
            return typeof column.position === "number" ? column.position : 0;
    }
}

function getSortedColumns() {
    const sorted = [...currentColumns];
    sorted.sort((a, b) => {
        const aValue = getSortValue(a);
        const bValue = getSortValue(b);
        if (aValue === bValue) {
            const aPos = typeof a.position === "number" ? a.position : 0;
            const bPos = typeof b.position === "number" ? b.position : 0;
            return aPos - bPos;
        }
        if (sortConfig.direction === "asc") {
            return aValue - bValue;
        }
        return bValue - aValue;
    });
    return sorted;
}

function createFlagIndicator(note) {
    const indicator = document.createElement("span");
    indicator.className = "tooltip alert";
    indicator.tabIndex = 0;
    indicator.setAttribute("role", "button");
    indicator.setAttribute("aria-label", "Flagged for review");
    indicator.textContent = "!";
    const tooltipContent = document.createElement("span");
    tooltipContent.className = "tooltip-content";
    tooltipContent.textContent = note || "Flagged for review.";
    indicator.appendChild(tooltipContent);
    return indicator;
}

function refreshTable() {
    const body = document.getElementById("schema-body");
    if (!body) {
        return;
    }
    hideFloatingTooltip();
    body.innerHTML = "";
    getSortedColumns().forEach((column) => {
        const row = document.createElement("tr");
        applyConfidence(row, column.confidence);

        const nameCell = document.createElement("td");
        const nameWrapper = document.createElement("div");
        nameWrapper.className = "column-name";
        const nameText = document.createElement("span");
        nameText.textContent = column.name;
        nameWrapper.appendChild(nameText);
        if (column.flagged) {
            nameWrapper.appendChild(createFlagIndicator(column.note));
        }
        nameCell.appendChild(nameWrapper);
        row.appendChild(nameCell);

        const typeCell = document.createElement("td");
        const select = document.createElement("select");
        TYPE_OPTIONS.forEach((option) => select.appendChild(createOption(option, column.type)));
        const nClassesInput = document.createElement("input");
        const yDimInput = document.createElement("input");
        syncNClassState(select, nClassesInput);
        select.addEventListener("change", () => {
            syncNClassState(select, nClassesInput);
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        typeCell.appendChild(select);
        row.appendChild(typeCell);

        const nClassesCell = document.createElement("td");
        nClassesInput.type = "number";
        nClassesInput.min = "2";
        nClassesInput.placeholder = "cat / ordinal only";
        nClassesInput.value = column.n_classes === null || column.n_classes === undefined ? "" : column.n_classes;
        nClassesInput.disabled = !(column.type === "cat" || column.type === "ordinal");
        nClassesInput.addEventListener("change", () => {
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        nClassesCell.appendChild(nClassesInput);
        row.appendChild(nClassesCell);

        const yDimCell = document.createElement("td");
        yDimInput.type = "number";
        yDimInput.min = "1";
        yDimInput.placeholder = "optional";
        yDimInput.value = column.y_dim === null || column.y_dim === undefined ? "" : column.y_dim;
        yDimInput.addEventListener("change", () => {
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        yDimCell.appendChild(yDimInput);
        row.appendChild(yDimCell);

        const distributionCell = document.createElement("td");
        const distributionButton = document.createElement("button");
        distributionButton.type = "button";
        distributionButton.className = "secondary";
        distributionButton.textContent = "View";
        distributionButton.addEventListener("click", () => showDistribution(column.name));
        distributionCell.appendChild(distributionButton);
        row.appendChild(distributionCell);

        const summaryCell = document.createElement("td");
        const summary = document.createElement("div");
        summary.innerHTML = `<strong>dtype:</strong> ${column.dtype} · <strong>nunique:</strong> ${column.nunique} · <strong>missing:</strong> ${column.missing}`;
        const sample = document.createElement("div");
        sample.className = "summary-note";
        sample.textContent = `sample: ${formatSampleArray(column.sample)}`;
        summaryCell.appendChild(summary);
        summaryCell.appendChild(sample);
        if (column.note) {
            const note = document.createElement("div");
            note.className = "summary-note";
            note.textContent = column.note;
            summaryCell.appendChild(note);
        }
        row.appendChild(summaryCell);

        body.appendChild(row);
    });
    attachTooltipHandlers(body);
}

function attachTooltipHandlers(root = document) {
    const tooltipElements = root.querySelectorAll(".tooltip");
    tooltipElements.forEach((tooltip) => {
        if (tooltip.dataset.tooltipBound === "true") {
            return;
        }
        tooltip.dataset.tooltipBound = "true";
        tooltip.addEventListener("mouseenter", handleTooltipEnter);
        tooltip.addEventListener("mouseleave", handleTooltipLeave);
        tooltip.addEventListener("focus", handleTooltipEnter);
        tooltip.addEventListener("blur", handleTooltipLeave);
        tooltip.addEventListener("keydown", handleTooltipKeydown);
    });
}

function handleTooltipEnter(event) {
    const target = event.currentTarget;
    target.setAttribute("aria-expanded", "true");
    showFloatingTooltip(target);
}

function handleTooltipLeave(event) {
    const target = event.currentTarget;
    if (activeTooltipTarget === target) {
        hideFloatingTooltip();
    } else {
        target.removeAttribute("aria-expanded");
    }
}

function handleTooltipKeydown(event) {
    if (event.key === "Escape") {
        hideFloatingTooltip();
        event.currentTarget.blur();
    }
}

function showFloatingTooltip(target) {
    const tooltipContent = target.querySelector(".tooltip-content");
    if (!tooltipContent) {
        return;
    }
    const tooltip = ensureFloatingTooltip();
    tooltip.innerHTML = "";
    if (tooltipContent.childElementCount > 0) {
        tooltip.innerHTML = tooltipContent.innerHTML;
    } else {
        const text = tooltipContent.textContent || "";
        if (!text.trim()) {
            tooltip.style.display = "none";
            return;
        }
        tooltip.textContent = text;
    }
    tooltip.style.display = "block";
    tooltip.style.visibility = "hidden";
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    activeTooltipTarget = target;
    requestAnimationFrame(() => {
        positionFloatingTooltip(target);
        tooltip.style.visibility = "visible";
    });
}

function hideFloatingTooltip() {
    if (activeTooltipTarget) {
        activeTooltipTarget.removeAttribute("aria-expanded");
    }
    const tooltip = floatingTooltip;
    if (!tooltip) {
        activeTooltipTarget = null;
        return;
    }
    tooltip.style.display = "none";
    tooltip.style.visibility = "hidden";
    activeTooltipTarget = null;
}

function positionFloatingTooltip(target) {
    const tooltip = floatingTooltip;
    if (!tooltip || tooltip.style.display === "none") {
        return;
    }
    const rect = target.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const margin = 8;
    let top = rect.bottom + margin;
    let left = rect.left + rect.width / 2 - tooltipRect.width / 2;
    if (left < margin) {
        left = margin;
    }
    const maxLeft = window.innerWidth - tooltipRect.width - margin;
    if (left > maxLeft) {
        left = maxLeft;
    }
    if (top + tooltipRect.height > window.innerHeight - margin) {
        top = rect.top - tooltipRect.height - margin;
        if (top < margin) {
            top = Math.min(window.innerHeight - tooltipRect.height - margin, rect.bottom + margin);
        }
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
}

function dismissFloatingTooltip() {
    if (activeTooltipTarget) {
        hideFloatingTooltip();
    }
}

function preparePayload(column, type, nClasses, yDim) {
    const payload = { column, type };
    if (nClasses !== "" && nClasses !== null) {
        payload.n_classes = Number(nClasses);
    }
    if (yDim !== "" && yDim !== null) {
        payload.y_dim = Number(yDim);
    }
    return payload;
}

async function sendUpdate(column, type, nClasses, yDim) {
    const payload = preparePayload(column, type, nClasses, yDim);
    try {
        const response = await fetch("/api/schema", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Server responded with ${response.status}`);
        }
        await loadState();
    } catch (error) {
        alert(`Failed to update column: ${error.message}`);
    }
}

async function showDistribution(column) {
    try {
        const response = await fetch(`/api/distribution?column=${encodeURIComponent(column)}`);
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        const payload = await response.json();
        if (payload.error) {
            alert(payload.error);
            return;
        }
        openDistributionWindow(column, payload);
    } catch (error) {
        alert(`Failed to load distribution: ${error.message}`);
    }
}

function openDistributionWindow(column, payload) {
    const win = window.open('', '_blank', 'width=720,height=520');
    if (!win) {
        alert('Unable to open a new window for the distribution plot.');
        return;
    }

    const doc = win.document;
    doc.open();
    doc.write(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Distribution for ${column}</title>
<style>
body { font-family: "Segoe UI", Arial, sans-serif; margin: 16px; }
#plot { width: 100%; min-height: 200px; display: flex; align-items: center; justify-content: center; }
#plot img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); border: 1px solid rgba(0, 0, 0, 0.08); }
#plot .message { font-size: 1rem; color: #333; text-align: center; }
</style>
</head>
<body>
<h2>Distribution for ${column}</h2>
<div id="plot"></div>
</body>
</html>`);
    doc.close();

    const plotContainer = doc.getElementById('plot');
    if (!plotContainer) {
        return;
    }

    if (payload.type === "empty") {
        const message = doc.createElement('p');
        message.className = 'message';
        message.textContent = payload.message || 'No data available.';
        plotContainer.appendChild(message);
        return;
    }

    if (payload.image_data) {
        const image = doc.createElement('img');
        image.src = payload.image_data;
        image.alt = payload.alt_text || `Distribution for ${column}`;
        plotContainer.appendChild(image);
        return;
    }

    plotContainer.innerText = 'Distribution image could not be rendered.';
}

async function finalizeSchema() {
    const button = document.getElementById("finalize");
    if (!button) {
        return;
    }
    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "Saving...";
    let success = false;
    try {
        const response = await fetch("/api/finalize", { method: "POST" });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Server responded with ${response.status}`);
        }
        window.__schemaFinalized = true;
        success = true;
        openSummaryModal(data);
    } catch (error) {
        alert(`Failed to save schema: ${error.message}`);
    } finally {
        button.textContent = originalText;
        if (!success) {
            button.disabled = false;
        }
    }
}

document.getElementById("finalize").addEventListener("click", finalizeSchema);

function notifyCancellation(reason) {
    if (cancellationNotified) {
        return;
    }
    cancellationNotified = true;
    const payload = JSON.stringify({ reason, timestamp: Date.now() });
    if (navigator.sendBeacon) {
        navigator.sendBeacon('/api/cancel', payload);
    } else {
        fetch('/api/cancel', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
            keepalive: true,
        }).catch(() => {});
    }
}

function handlePageExit() {
    if (window.__schemaFinalized) {
        return;
    }
    notifyCancellation('page_exit');
}

window.addEventListener("beforeunload", handlePageExit);
window.addEventListener("pagehide", handlePageExit);
window.addEventListener("scroll", dismissFloatingTooltip, true);
window.addEventListener("resize", dismissFloatingTooltip);
document.addEventListener("pointerdown", (event) => {
    const tooltip = floatingTooltip;
    if (tooltip && tooltip.contains(event.target)) {
        return;
    }
    dismissFloatingTooltip();
});

attachTooltipHandlers();

loadState();
</script>
</body>
</html>
"""


def launch_schema_builder(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    open_browser: bool = True,
    mode: str = "info",
    inferencer: Optional[SchemaInferencer] = None,
) -> Schema:
    """Launch a local web UI to review and edit a schema.

    Parameters
    ----------
    df:
        Data frame containing the candidate feature columns.
    feature_columns:
        Optional subset of column names to review. When omitted, all columns in
        df are considered.
    host, port:
        Address where the temporary web server should listen.
    open_browser:
        If True (default) the default system browser is opened
        automatically.
    mode:
        Inference mode forwarded to :class:`SchemaInferencer`. ``"info"`` works
        well for interactive use because it includes diagnostic messages.
    inferencer:
        Optional pre-configured inferencer. When None a fresh instance with
        default settings is created.

    Returns
    -------
    Schema
        The final schema selected through the web interface.

    Raises
    ------
    SchemaBuilderError
        If Flask (or one of its dependencies) is not available, if the server
        cannot bind to host:port, or if the inferred schema cannot be
        validated.

    Examples
    --------
    >>> import pandas as pd
    >>> from suave.interactive import launch_schema_builder
    >>> df = pd.DataFrame({"age": [25, 30], "sex": ["F", "M"]})
    >>> schema = launch_schema_builder(df, open_browser=False)  # doctest: +SKIP
    >>> schema.to_dict()  # doctest: +SKIP
    {'age': {'type': 'real'}, 'sex': {'type': 'cat', 'n_classes': 2}}
    """

    if Flask is None or jsonify is None or request is None or make_server is None:
        raise SchemaBuilderError(
            "The interactive schema builder requires Flask. Install the optional "
            "dependency via 'pip install flask' or 'pip install suave[schema-ui]'"
        )

    inferencer = inferencer or SchemaInferencer()
    mode_normalised = str(mode).lower()
    if mode_normalised not in SCHEMA_INFERENCE_MODES:
        raise ValueError(
            "mode must be one of {'silent', 'info', 'interactive'}; "
            f"got {mode!r}"
        )
    result = inferencer.infer(df, feature_columns, mode=mode_normalised)
    builder = _SchemaBuilder(
        df=df,
        inference=result,
        host=host,
        port=port,
        open_browser=open_browser,
    )
    return builder.run()


class _SchemaBuilder:
    """Internal helper orchestrating the Flask server lifecycle."""

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        inference: SchemaInferenceResult,
        host: str,
        port: int,
        open_browser: bool,
    ) -> None:
        self._df = df
        self._host = host
        self._port = port
        self._open_browser = open_browser
        self._schema_dict: Dict[str, MutableMapping[str, object]] = json.loads(
            json.dumps(inference.schema.to_dict())
        )
        self._initial_schema: Dict[str, Dict[str, object]] = json.loads(
            json.dumps(self._schema_dict)
        )
        self._notes = dict(inference.column_notes)
        self._messages = [
            message
            for message in inference.messages
            if "flagged for review" not in message.lower()
        ]
        self._confidence_map: Dict[str, str] = {}
        for name in self._schema_dict:
            raw_confidence = inference.column_confidence.get(name, "high")
            value = getattr(raw_confidence, "value", raw_confidence)
            self._confidence_map[name] = str(value)
        if not self._messages:
            self._messages = [
                "Schema inferred automatically; adjust columns as needed."
            ]
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cancelled = False
        self._result_schema: Optional[Schema] = None

    def run(self) -> Schema:
        app = Flask("suave_schema_builder")
        self._register_routes(app)

        try:
            server = make_server(self._host, self._port, app)
        except OSError as error:
            raise SchemaBuilderError(
                f"Unable to start schema builder on {self._host}:{self._port}: {error}"
            ) from error

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        if self._open_browser:
            url = f"http://{self._host}:{self._port}/"
            threading.Thread(
                target=self._launch_browser, args=(url,), daemon=True
            ).start()

        self._stop_event.wait()
        server.shutdown()
        server_thread.join(timeout=2.0)

        if self._cancelled:
            raise SchemaBuilderError("Schema builder cancelled by user.")
        if self._result_schema is None:
            raise SchemaBuilderError("Schema builder terminated without a result.")
        return self._result_schema

    def _launch_browser(self, url: str) -> None:
        time.sleep(0.5)
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - best effort only
            pass

    def _register_routes(self, app: Flask) -> None:
        @app.get("/")
        def index():  # pragma: no cover - exercised via integration
            return HTML_TEMPLATE

        @app.get("/api/state")
        def state():
            with self._lock:
                payload = {
                    "columns": [
                        self._column_state(name, index).to_dict()
                        for index, name in enumerate(self._schema_dict)
                    ],
                    "messages": list(self._messages),
                }
            return jsonify(payload)

        @app.post("/api/schema")
        def update_schema():
            data = request.get_json(silent=True) or {}
            column = data.get("column")
            new_type = data.get("type")
            raw_n_classes = data.get("n_classes")
            raw_y_dim = data.get("y_dim")

            if column not in self._schema_dict:
                return jsonify({"error": f"Unknown column '{column}'."}), 400
            if new_type not in _SUPPORTED_TYPES:
                return jsonify({"error": f"Unsupported type '{new_type}'."}), 400

            try:
                n_classes = _coerce_optional_positive_int(raw_n_classes)
                y_dim = _coerce_optional_positive_int(raw_y_dim)
            except ValueError as error:
                return jsonify({"error": str(error)}), 400

            updated_spec: MutableMapping[str, object] = {"type": new_type}
            if new_type in {"cat", "ordinal"}:
                if n_classes is None:
                    return (
                        jsonify(
                            {
                                "error": (
                                    f"Column '{column}' is '{new_type}' and requires 'n_classes' > 1."
                                )
                            }
                        ),
                        400,
                    )
                updated_spec["n_classes"] = int(n_classes)
            elif n_classes is not None:
                return (
                    jsonify(
                        {
                            "error": "'n_classes' only applies to categorical/ordinal types."
                        }
                    ),
                    400,
                )

            if y_dim is not None:
                updated_spec["y_dim"] = int(y_dim)

            with self._lock:
                self._schema_dict[column] = updated_spec
                position = list(self._schema_dict).index(column)
                payload = self._column_state(column, position).to_dict()
            return jsonify(payload)

        @app.get("/api/distribution")
        def distribution():
            column = request.args.get("column", "")
            if column not in self._schema_dict:
                return jsonify({"error": f"Unknown column '{column}'."}), 400

            payload = _distribution_payload(self._df[column], column)
            return jsonify(payload)

        @app.post("/api/finalize")
        def finalize():
            with self._lock:
                try:
                    self._result_schema = Schema(dict(self._schema_dict))
                except ValueError as error:
                    return jsonify({"error": str(error)}), 400
                changes, changed_specs = _summarise_schema_changes(
                    self._initial_schema,
                    self._schema_dict,
                )
                python_code = _format_python_update(changed_specs)
                response_payload = {
                    "status": "ok",
                    "changes": changes,
                    "python": python_code,
                }
                self._stop_event.set()
            return jsonify(response_payload)

        @app.post("/api/cancel")
        def cancel():
            with self._lock:
                self._cancelled = True
                self._stop_event.set()
            return jsonify({"status": "cancelled"})

    def _column_state(self, column: str, position: int) -> _ColumnState:
        spec = self._schema_dict[column]
        series = self._df[column]
        summary = _summarise_series(series)
        note = self._notes.get(column, "")
        return _ColumnState(
            name=column,
            type=str(spec.get("type", "")),
            note=note,
            dtype=summary["dtype"],
            nunique=summary["nunique"],
            missing=summary["missing"],
            sample=summary["sample"],
            n_classes=spec.get("n_classes"),
            y_dim=spec.get("y_dim"),
            confidence=self._confidence_map.get(column, "high"),
            position=position,
            flagged=column in self._notes,
        )


def _distribution_payload(series: pd.Series, column: str) -> Dict[str, object]:
    """Return data suitable for plotting the distribution of series."""

    non_null = series.dropna()
    if non_null.empty:
        return {
            "type": "empty",
            "column": column,
            "title": f"Distribution for {column}",
            "message": "Column contains only missing values.",
        }

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        values = numeric.to_numpy(dtype=float)
        if values.size == 0:
            return {
                "type": "empty",
                "column": column,
                "title": f"Distribution for {column}",
                "message": "Column contains only missing values.",
            }
        bins = int(max(5, min(30, np.sqrt(values.size))))
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        image_data, alt_text = _render_distribution_image(
            column,
            kind="hist",
            values=values,
        )
        return {
            "type": "hist",
            "column": column,
            "title": f"Distribution for {column}",
            "bins": [float(x) for x in centers.tolist()],
            "counts": [int(x) for x in counts.tolist()],
            "x_label": "Value",
            "y_label": "Count",
            "image_data": image_data,
            "alt_text": alt_text,
        }

    value_counts = non_null.astype(str).value_counts().sort_index()
    labels = [str(label) for label in value_counts.index.tolist()]
    counts = [int(x) for x in value_counts.tolist()]
    image_data, alt_text = _render_distribution_image(
        column,
        kind="bar",
        labels=labels,
        counts=counts,
    )
    return {
        "type": "bar",
        "column": column,
        "title": f"Distribution for {column}",
        "labels": labels,
        "counts": counts,
        "x_label": "Category",
        "y_label": "Count",
        "image_data": image_data,
        "alt_text": alt_text,
    }


def _render_distribution_image(
    column: str,
    *,
    kind: str,
    values: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    counts: Optional[List[int]] = None,
) -> tuple[str, str]:
    """Render a distribution plot and return its base64 representation and alt text."""

    figure = Figure(figsize=(6, 4), dpi=110)
    _ = FigureCanvasAgg(figure)
    ax = figure.subplots()

    if kind == "hist" and values is not None:
        sns.histplot(values, kde=True, stat="count", ax=ax, color="#4c72b0")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        alt_text = f"Histogram with KDE for column {column}"
    elif kind == "bar" and labels is not None and counts is not None:
        repeated = np.repeat(labels, counts)
        sns.countplot(x=repeated, order=labels, ax=ax, color="#4c72b0")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        alt_text = f"Category count plot for column {column}"
    else:
        raise ValueError("Invalid arguments for distribution rendering.")

    ax.set_title(f"Distribution for {column}")
    figure.tight_layout()

    buffer = BytesIO()
    try:
        figure.savefig(buffer, format="png", bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        buffer.close()
        figure.clear()

    return f"data:image/png;base64,{encoded}", alt_text

def _normalise_schema_spec(
    spec: Optional[MutableMapping[str, object]] | Dict[str, object] | None,
) -> Optional[Dict[str, object]]:
    if spec is None:
        return None
    return json.loads(json.dumps(spec))


def _summarise_schema_changes(
    initial: Dict[str, Dict[str, object]],
    current: Dict[str, MutableMapping[str, object]],
) -> tuple[List[Dict[str, object]], OrderedDict[str, Dict[str, object]]]:
    changes: List[Dict[str, object]] = []
    changed_specs: OrderedDict[str, Dict[str, object]] = OrderedDict()
    for column, spec in current.items():
        after = _normalise_schema_spec(spec)
        before = _normalise_schema_spec(initial.get(column))
        if before != after:
            changes.append({"column": column, "before": before, "after": after})
            if after is not None:
                changed_specs[column] = after
    return changes, changed_specs


def _format_python_update(
    changed_specs: OrderedDict[str, Dict[str, object]],
) -> str:
    if not changed_specs:
        return ""
    lines = ["schema.update({"]
    for column, spec in changed_specs.items():
        lines.append(f"    {column!r}: {spec!r},")
    lines.append("})")
    return "\n".join(lines)


def _coerce_optional_positive_int(value: object) -> Optional[int]:
    """Convert value to int when provided, ensuring it is positive."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid integers.")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected a positive integer.") from error
    if coerced <= 0:
        raise ValueError("Expected a positive integer.")
    return coerced


def _summarise_series(series: pd.Series) -> Dict[str, object]:
    """Return a compact summary of series suitable for JSON transport."""

    non_null = series.dropna()
    nunique = int(non_null.nunique()) if len(non_null) else 0
    missing = int(series.isna().sum()) if hasattr(series, "isna") else 0
    sample = [_to_python(value) for value in series.head(5)]
    return {
        "dtype": str(series.dtype),
        "nunique": nunique,
        "missing": missing,
        "sample": sample,
    }


def _to_python(value: object) -> object:
    """Convert numpy scalars to plain Python values for JSON encoding."""

    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


__all__ = ["launch_schema_builder"]
