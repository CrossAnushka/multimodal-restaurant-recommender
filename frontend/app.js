"use strict";

const PRICE_LABELS = { 1: "$", 2: "$$", 3: "$$$", 4: "$$$$" };
const TOP_CUISINES = 24; // synthetic data has 12; show all comfortably

const state = {
  selectedCuisines: new Set(),
  price: 2,
};

const $ = (sel) => document.querySelector(sel);

async function init() {
  let opts;
  try {
    opts = await fetch("/api/options").then((r) => r.json());
  } catch (e) {
    $("#hint").textContent = "Could not reach the API. Is the server running?";
    return;
  }
  renderCuisines(opts.cuisines);
  renderPrices(opts.price_bands);
  renderCities(opts.cities);

  $("#onboarding").addEventListener("submit", onSubmit);
  $("#userBtn").addEventListener("click", onUserSubmit);
  $("#userId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); onUserSubmit(); }
  });
}

function renderCuisines(cuisines) {
  const box = $("#cuisines");
  box.innerHTML = "";
  cuisines.slice(0, TOP_CUISINES).forEach(({ name }) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = name;
    btn.addEventListener("click", () => {
      if (state.selectedCuisines.has(name)) {
        state.selectedCuisines.delete(name);
        btn.classList.remove("active");
      } else {
        state.selectedCuisines.add(name);
        btn.classList.add("active");
      }
      updateHint();
    });
    box.appendChild(btn);
  });
}

function renderPrices(bands) {
  const box = $("#price");
  box.innerHTML = "";
  bands.forEach((b) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "price-btn" + (b === state.price ? " active" : "");
    btn.textContent = PRICE_LABELS[b];
    btn.addEventListener("click", () => {
      state.price = b;
      box.querySelectorAll(".price-btn").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
    });
    box.appendChild(btn);
  });
}

function renderCities(cities) {
  const sel = $("#city");
  cities.forEach((c) => {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c;
    sel.appendChild(o);
  });
}

function updateHint() {
  const n = state.selectedCuisines.size;
  $("#hint").textContent = n === 0
    ? "Pick one or more cuisines to get started."
    : `${n} cuisine${n > 1 ? "s" : ""} selected.`;
}

async function onSubmit(e) {
  e.preventDefault();
  if (state.selectedCuisines.size === 0) {
    $("#hint").textContent = "Please pick at least one cuisine.";
    return;
  }
  const city = $("#city").value;
  const payload = {
    cuisines: [...state.selectedCuisines],
    price: state.price,
    city: city || null,
    k: 9,
  };
  const subtitle = `Cold-start picks · ${payload.cuisines.join(", ")} · ${PRICE_LABELS[payload.price]}` +
    (city ? ` · ${city}` : "");
  await runQuery("/api/recommend/cold", payload, "Fresh picks for you", subtitle);
}

async function onUserSubmit() {
  const id = parseInt($("#userId").value, 10);
  if (Number.isNaN(id)) { $("#userId").focus(); return; }
  await runQuery("/api/recommend/user", { user_id: id, k: 9 },
    `Picks for user #${id}`, "Ranked by the hybrid model from this user's taste profile");
}

async function runQuery(url, payload, title, subtitle) {
  showResults();
  $("#resultsTitle").textContent = title;
  $("#resultsSub").textContent = subtitle;
  $("#grid").innerHTML = `<div class="state"><div class="spinner"></div>Ranking restaurants…</div>`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderResults(data.results, data.cold_user);
  } catch (err) {
    $("#grid").innerHTML = `<div class="state">⚠️ ${err.message}</div>`;
  }
}

function showResults() {
  $("#resultsSection").hidden = false;
  $("#resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResults(results, coldUser) {
  const grid = $("#grid");
  if (!results || results.length === 0) {
    grid.innerHTML = `<div class="state">No matches found. Try different cuisines.</div>`;
    return;
  }
  if (coldUser) {
    $("#resultsSub").textContent += " · ❄️ cold-start user (no history)";
  }
  grid.innerHTML = "";
  results.forEach((r) => grid.appendChild(card(r)));
}

function card(r) {
  const el = document.createElement("article");
  el.className = "card";

  const img = document.createElement("img");
  img.className = "card-img";
  img.loading = "lazy";
  img.src = r.image_url;
  img.alt = `${r.name} — ${r.cuisine}`;
  img.onerror = () => { img.style.visibility = "hidden"; };

  const body = document.createElement("div");
  body.className = "card-body";
  body.innerHTML = `
    <span class="card-cuisine">${esc(r.cuisine)}</span>
    <h3 class="card-name">${esc(r.name)}</h3>
    <span class="card-meta">${r.price} · ${esc(r.city)}</span>
    <div class="badges">
      <span class="badge badge-score">match ${(r.score).toFixed(2)}</span>
      ${r.cold_item ? '<span class="badge badge-new">New</span>' : ""}
    </div>
    ${whyBars(r.why)}
  `;

  el.appendChild(img);
  el.appendChild(body);
  return el;
}

function whyBars(why) {
  const rows = [
    ["text", "Review text"],
    ["image", "Food image"],
    ["structured", "Attributes"],
  ];
  const bars = rows.map(([key, label]) => {
    const pct = Math.round((why[key] || 0) * 100);
    return `
      <div class="why-row">
        <span class="why-label">${label}</span>
        <span class="why-track"><span class="why-fill ${key}" style="width:${pct}%"></span></span>
      </div>`;
  }).join("");
  return `<div class="why">${bars}</div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

init();
