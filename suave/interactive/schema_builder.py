"""Browser-based helper for interactive schema editing.

The utilities in this module are optional and only activated when the
Flask dependency is available. They provide a lightweight web interface
that runs locally, making it possible to review and adjust the schema
suggested by the SchemaInferencer.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..schema_inference import (
    SchemaInferenceMode,
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
        }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SUAVE Schema Builder</title>
<style>
body { font-family: Arial, sans-serif; margin: 16px; }
h1 { font-size: 1.4rem; }
table { border-collapse: collapse; width: 100%; margin-top: 16px; }
th, td { border: 1px solid #ddd; padding: 8px; }
th { background-color: #f4f4f4; text-align: left; }
select, input { width: 100%; box-sizing: border-box; }
small { color: #555; }
button { margin-top: 16px; padding: 8px 16px; font-size: 1rem; }
.message { margin-top: 8px; padding: 8px; background: #eef; border: 1px solid #ccd; }
.error { background: #fee; border-color: #f99; }
</style>
</head>
<body>
<h1>SUAVE Schema Builder</h1>
<p>Review the automatically inferred schema, adjust column types as needed, and click <strong>Save schema</strong> when finished.</p>
<div id="messages"></div>
<table>
<thead>
<tr>
<th>Column</th>
<th>Type</th>
<th>n_classes</th>
<th>y_dim</th>
<th>Distribution</th>
<th>Summary</th>
</tr>
</thead>
<tbody id="schema-body"></tbody>
</table>
<button id="finalize">Save schema</button>
<script>
const TYPE_OPTIONS = ["real", "pos", "count", "cat", "ordinal"];

function renderMessages(messages) {
    const container = document.getElementById("messages");
    container.innerHTML = "";
    messages.forEach(function(msg) {
        const div = document.createElement("div");
        div.className = "message";
        div.textContent = msg;
        container.appendChild(div);
    });
}

function showDistribution(column) {
    fetch("/api/distribution?column=" + encodeURIComponent(column))
        .then(function(response) {
            if (!response.ok) {
                throw new Error('Failed to load distribution');
            }
            return response.json();
        })
        .then(function(payload) {
            if (payload.error) {
                alert(payload.error);
                return;
            }
            openDistributionWindow(column, payload);
        })
        .catch(function(err) {
            alert('Failed to load distribution: ' + err);
        });
}

function openDistributionWindow(column, payload) {
    const win = window.open('', '_blank', 'width=720,height=520');
    if (!win) {
        alert('Unable to open a new window for the distribution plot.');
        return;
    }
    const payloadJson = JSON.stringify(payload);
    const docLines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="utf-8" />',
        '<title>Distribution for ' + column + '</title>',
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        '<style>body { font-family: Arial, sans-serif; margin: 16px; } #plot { width: 100%; height: 90vh; }</style>',
        '</head>',
        '<body>',
        '<h2>Distribution for ' + column + '</h2>',
        '<div id="plot"></div>',
        '<script>',
        'const payload = ' + payloadJson + ';',
        'if (payload.type === "empty") {',
        "  document.getElementById('plot').innerText = payload.message || 'No data available.';",
        '} else {',
        "  const layout = {title: payload.title || 'Distribution', xaxis: {title: payload.x_label || 'Value'}, yaxis: {title: payload.y_label || 'Count'}};",
        '  let trace;',
        '  if (payload.type === "hist") {',
        "    trace = {type: 'bar', x: payload.bins, y: payload.counts, marker: {color: '#4c72b0'}};",
        '  } else {',
        "    trace = {type: 'bar', x: payload.labels, y: payload.counts, marker: {color: '#4c72b0'}};",
        '  }',
        "  Plotly.newPlot('plot', [trace], layout);",
        '}',
        '</script>',
        '</body>',
        '</html>',
    ];
    win.document.write(docLines.join(''));
    win.document.close();
}

function renderTable(columns) {
    const body = document.getElementById("schema-body");
    body.innerHTML = "";
    columns.forEach(function(column) {
        const row = document.createElement("tr");

        const nameCell = document.createElement("td");
        nameCell.textContent = column.name;
        row.appendChild(nameCell);

        const typeCell = document.createElement("td");
        const select = document.createElement("select");
        TYPE_OPTIONS.forEach(function(option) {
            const choice = document.createElement("option");
            choice.value = option;
            choice.textContent = option;
            if (option === column.type) {
                choice.selected = true;
            }
            select.appendChild(choice);
        });
        select.addEventListener("change", function() {
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        typeCell.appendChild(select);
        row.appendChild(typeCell);

        const nClassesCell = document.createElement("td");
        const nClassesInput = document.createElement("input");
        nClassesInput.type = "number";
        nClassesInput.min = "2";
        nClassesInput.placeholder = "required for cat/ordinal";
        nClassesInput.value = column.n_classes === null ? "" : column.n_classes;
        nClassesInput.addEventListener("change", function() {
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        nClassesCell.appendChild(nClassesInput);
        row.appendChild(nClassesCell);

        const yDimCell = document.createElement("td");
        const yDimInput = document.createElement("input");
        yDimInput.type = "number";
        yDimInput.min = "1";
        yDimInput.placeholder = "optional";
        yDimInput.value = column.y_dim === null ? "" : column.y_dim;
        yDimInput.addEventListener("change", function() {
            sendUpdate(column.name, select.value, nClassesInput.value, yDimInput.value);
        });
        yDimCell.appendChild(yDimInput);
        row.appendChild(yDimCell);

        const distributionCell = document.createElement("td");
        const distributionButton = document.createElement("button");
        distributionButton.type = "button";
        distributionButton.textContent = "View";
        distributionButton.addEventListener("click", function() {
            showDistribution(column.name);
        });
        distributionCell.appendChild(distributionButton);
        row.appendChild(distributionCell);

        const summaryCell = document.createElement("td");
        const summary = document.createElement("small");
        const sampleText = JSON.stringify(column.sample);
        summary.textContent = 'dtype=' + column.dtype + ', nunique=' + column.nunique + ', missing=' + column.missing + ', sample=' + sampleText;
        if (column.note) {
            const note = document.createElement("div");
            note.textContent = column.note;
            summary.appendChild(document.createElement("br"));
            summary.appendChild(note);
        }
        summaryCell.appendChild(summary);
        row.appendChild(summaryCell);

        body.appendChild(row);
    });
}

function loadState() {
    fetch("/api/state")
        .then(function(response) {
            if (!response.ok) {
                throw new Error('Failed to load state');
            }
            return response.json();
        })
        .then(function(payload) {
            renderMessages(payload.messages || []);
            renderTable(payload.columns || []);
        })
        .catch(function(err) {
            showError('Unable to load initial state: ' + err);
        });
}

function preparePayload(column, type, nClasses, yDim) {
    const payload = { column: column, type: type };
    if (nClasses !== '' && nClasses !== null) {
        payload.n_classes = Number(nClasses);
    }
    if (yDim !== '' && yDim !== null) {
        payload.y_dim = Number(yDim);
    }
    return payload;
}

function sendUpdate(column, type, nClasses, yDim) {
    const payload = preparePayload(column, type, nClasses, yDim);
    fetch("/api/schema", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    }).then(function(response) {
        return response.json();
    }).then(function(data) {
        if (data.error) {
            alert(data.error);
        } else {
            loadState();
        }
    }).catch(function(err) {
        alert('Failed to update column: ' + err);
    });
}

document.getElementById("finalize").addEventListener("click", function() {
    fetch("/api/finalize", { method: "POST" })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            if (data.error) {
                alert(data.error);
            } else {
                alert('Schema saved. You can now close this tab.');
            }
        })
        .catch(function(err) {
            alert('Failed to save schema: ' + err);
        });
});

function showError(message) {
    const container = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message error";
    div.textContent = message;
    container.appendChild(div);
}

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
    mode: SchemaInferenceMode = SchemaInferenceMode.INFO,
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
        Inference mode forwarded to SchemaInferencer. INFO works well for
        interactive use because it includes diagnostic messages.
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
    >>> from SUAVE.interactive import launch_schema_builder
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
    result = inferencer.infer(df, feature_columns, mode=mode)
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
        self._notes = dict(inference.column_notes)
        self._messages = list(inference.messages)
        if not self._messages:
            self._messages = ["Schema inferred automatically; adjust columns as needed."]
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
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
            threading.Thread(target=self._launch_browser, args=(url,), daemon=True).start()

        self._stop_event.wait()
        server.shutdown()
        server_thread.join(timeout=2.0)

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
                    "columns": [self._column_state(name).to_dict() for name in self._schema_dict],
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
                    return jsonify(
                        {
                            "error": (
                                f"Column '{column}' is '{new_type}' and requires 'n_classes' > 1."
                            )
                        }
                    ), 400
                updated_spec["n_classes"] = int(n_classes)
            elif n_classes is not None:
                return jsonify({"error": "'n_classes' only applies to categorical/ordinal types."}), 400

            if y_dim is not None:
                updated_spec["y_dim"] = int(y_dim)

            with self._lock:
                self._schema_dict[column] = updated_spec
                payload = self._column_state(column).to_dict()
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
                self._stop_event.set()
            return jsonify({"status": "ok"})

    def _column_state(self, column: str) -> _ColumnState:
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
        return {
            "type": "hist",
            "column": column,
            "title": f"Distribution for {column}",
            "bins": [float(x) for x in centers.tolist()],
            "counts": [int(x) for x in counts.tolist()],
            "x_label": "Value",
            "y_label": "Count",
        }

    value_counts = non_null.astype(str).value_counts().sort_index()
    labels = [str(label) for label in value_counts.index.tolist()]
    return {
        "type": "bar",
        "column": column,
        "title": f"Distribution for {column}",
        "labels": labels,
        "counts": [int(x) for x in value_counts.tolist()],
        "x_label": "Category",
        "y_label": "Count",
    }


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
