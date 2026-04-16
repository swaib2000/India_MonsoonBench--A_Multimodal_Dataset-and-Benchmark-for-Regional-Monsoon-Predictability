function parseCSV(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (quoted) {
      if (char === '"' && next === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        value += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(value);
      value = "";
    } else if (char === "\n") {
      row.push(value);
      rows.push(row);
      row = [];
      value = "";
    } else if (char !== "\r") {
      value += char;
    }
  }

  if (value.length || row.length) {
    row.push(value);
    rows.push(row);
  }

  return rows.filter((r) => r.some((cell) => cell.trim().length));
}

function formatCell(value) {
  const trimmed = value.trim();
  if (trimmed === "") return "";
  const numeric = Number(trimmed);
  if (!Number.isNaN(numeric) && trimmed.match(/^-?\d+(\.\d+)?$/)) {
    if (Math.abs(numeric) >= 1000 && Number.isInteger(numeric)) {
      return numeric.toLocaleString("en-IN");
    }
    if (Math.abs(numeric) < 1 && numeric !== 0) return numeric.toFixed(3);
    if (!Number.isInteger(numeric)) return numeric.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  return trimmed.replaceAll("_", " ");
}

async function renderCSVTable(container) {
  const src = container.dataset.csv;
  const maxRows = Number(container.dataset.maxRows || "0");
  const response = await fetch(src);
  const text = await response.text();
  const rows = parseCSV(text);
  if (!rows.length) return;

  const headers = rows[0];
  const body = maxRows > 0 ? rows.slice(1, maxRows + 1) : rows.slice(1);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");

  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header.replaceAll("_", " ");
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  body.forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((_, index) => {
      const td = document.createElement("td");
      td.textContent = formatCell(row[index] || "");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-csv]").forEach((container) => {
    renderCSVTable(container).catch((error) => {
      container.innerHTML = `<p class="note">Could not load table: ${error.message}</p>`;
    });
  });
});
