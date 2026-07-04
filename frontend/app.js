"use strict";

const PRICE_LABELS = { 1: "$", 2: "$$", 3: "$$$", 4: "$$$$" };
const TOP_CUISINES = 24; // synthetic data has 12; show all comfortably

const LS = {
  favorites: "rrs:favorites",
  lastSearch: "rrs:lastSearch",
  theme: "rrs:theme",
};

// Human-readable explanation of what each modality contributes — used in the
// interactive "why matched" tooltips.
const WHY_INFO = {
  text: ["Review text", "DistilBERT read this place's reviews and menu, matching their meaning against your taste."],
  image: ["Food image", "ResNet18 turned the food photos into a visual-style vector and compared it to what you like."],
  structured: ["Attributes", "The structured encoder matched cuisine, price band and location to your preferences."],
};

const state = {
  selectedCuisines: new Set(),
  price: 2,
  allResults: [],          // raw results from the last query
  lastQuery: null,         // {title, subtitle} for restoring after Saved view
  view: "results",         // "results" | "saved"
};

const favorites = loadFavorites();   // Map<id, cardObject>

const $ = (sel) => document.querySelector(sel);

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //
async function init() {
  applyTheme(localStorage.getItem(LS.theme) ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#themeToggle").addEventListener("click", toggleTheme);

  let opts;
  try {
    opts = await fetch("/api/options").then((r) => r.json());
  } catch (e) {
    toast("Could not reach the API. Is the server running?", "error");
    $("#hint").textContent = "Could not reach the API. Is the server running?";
    return;
  }
  renderCuisines(opts.cuisines);
  renderPrices(opts.price_bands);
  renderCities(opts.cities);
  restoreLastSearch();
  renderGallery();
  updateSavedCount();

  $("#onboarding").addEventListener("submit", onSubmit);
  $("#userBtn").addEventListener("click", onUserSubmit);
  $("#userId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); onUserSubmit(); }
  });

  // Toolbar: re-render on any control change.
  ["#searchBox", "#sortBy", "#minRating", "#filterVeg", "#filterOpen"]
    .forEach((sel) => {
      const ev = sel === "#searchBox" ? "input" : "change";
      $(sel).addEventListener(ev, renderVisible);
    });

  // Saved view toggle.
  $("#savedBtn").addEventListener("click", toggleSavedView);

  // Modal close (button, backdrop click, Escape).
  $("#modalClose").addEventListener("click", closeModal);
  $("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
}

// --------------------------------------------------------------------------- //
// Onboarding form rendering
// --------------------------------------------------------------------------- //
function renderCuisines(cuisines) {
  const box = $("#cuisines");
  box.innerHTML = "";
  cuisines.slice(0, TOP_CUISINES).forEach(({ name }) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = name;
    btn.dataset.cuisine = name;
    btn.addEventListener("click", () => {
      if (state.selectedCuisines.has(name)) {
        state.selectedCuisines.delete(name);
        btn.classList.remove("active");
      } else {
        state.selectedCuisines.add(name);
        btn.classList.add("active");
      }
      updateHint();
      saveLastSearch();
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
    btn.dataset.band = b;
    btn.addEventListener("click", () => {
      state.price = b;
      box.querySelectorAll(".price-btn").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      saveLastSearch();
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
  sel.addEventListener("change", saveLastSearch);
}

function updateHint() {
  const n = state.selectedCuisines.size;
  $("#hint").textContent = n === 0
    ? "Pick one or more cuisines to get started."
    : `${n} cuisine${n > 1 ? "s" : ""} selected.`;
}

// --------------------------------------------------------------------------- //
// Persist last search
// --------------------------------------------------------------------------- //
function saveLastSearch() {
  const payload = {
    cuisines: [...state.selectedCuisines],
    price: state.price,
    city: $("#city").value || "",
  };
  try { localStorage.setItem(LS.lastSearch, JSON.stringify(payload)); } catch (e) {}
}

function restoreLastSearch() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem(LS.lastSearch)); } catch (e) { return; }
  if (!saved) return;

  (saved.cuisines || []).forEach((name) => {
    state.selectedCuisines.add(name);
    const chip = document.querySelector(`.chip[data-cuisine="${CSS.escape(name)}"]`);
    if (chip) chip.classList.add("active");
  });
  if (saved.price) {
    state.price = saved.price;
    $("#price").querySelectorAll(".price-btn").forEach((x) =>
      x.classList.toggle("active", Number(x.dataset.band) === saved.price));
  }
  if (saved.city) {
    const opt = $("#city").querySelector(`option[value="${CSS.escape(saved.city)}"]`);
    if (opt) $("#city").value = saved.city;
  }
  updateHint();
}

// --------------------------------------------------------------------------- //
// Queries
// --------------------------------------------------------------------------- //
async function onSubmit(e) {
  e.preventDefault();
  if (state.selectedCuisines.size === 0) {
    toast("Please pick at least one cuisine.", "error");
    $("#hint").textContent = "Please pick at least one cuisine.";
    return;
  }
  saveLastSearch();
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
  if (Number.isNaN(id)) { $("#userId").focus(); toast("Enter a numeric user id.", "error"); return; }
  await runQuery("/api/recommend/user", { user_id: id, k: 9 },
    `Picks for user #${id}`, "Ranked by the hybrid model from this user's taste profile");
}

async function runQuery(url, payload, title, subtitle) {
  state.view = "results";
  updateSavedBtn();
  showResults();
  $("#resultsTitle").textContent = title;
  $("#resultsSub").textContent = subtitle;
  $("#toolbar").hidden = true;
  $("#grid").innerHTML = skeletons(payload.k || 9);

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
    if (data.cold_user) {
      subtitle += " · ❄️ cold-start user (no history)";
      $("#resultsSub").textContent = subtitle;
    }
    state.allResults = data.results || [];
    state.lastQuery = { title, subtitle };
    renderVisible();
  } catch (err) {
    state.allResults = [];
    $("#toolbar").hidden = true;
    $("#grid").innerHTML = `<div class="state">⚠️ ${esc(err.message)}</div>`;
    toast(err.message, "error");
  }
}

function showResults() {
  $("#resultsSection").hidden = false;
  $("#resultsSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

// --------------------------------------------------------------------------- //
// Filtering / sorting / rendering
// --------------------------------------------------------------------------- //
function sourceList() {
  return state.view === "saved" ? [...favorites.values()] : state.allResults;
}

function getVisible() {
  let list = [...sourceList()];
  const q = $("#searchBox").value.trim().toLowerCase();
  const minRating = parseFloat($("#minRating").value) || 0;
  const veg = $("#filterVeg").checked;
  const open = $("#filterOpen").checked;

  if (q) {
    list = list.filter((r) =>
      [r.name, r.cuisine, r.city, r.neighborhood]
        .some((f) => String(f || "").toLowerCase().includes(q)));
  }
  if (minRating > 0) list = list.filter((r) => (r.avg_rating || 0) >= minRating);
  if (veg) list = list.filter((r) => r.veg_friendly);
  if (open) list = list.filter((r) => r.open_now);

  const sort = $("#sortBy").value;
  const cmp = {
    score: (a, b) => b.score - a.score,
    rating: (a, b) => (b.avg_rating || 0) - (a.avg_rating || 0),
    price_asc: (a, b) => a.price_band - b.price_band,
    price_desc: (a, b) => b.price_band - a.price_band,
    name: (a, b) => a.name.localeCompare(b.name),
  }[sort];
  if (cmp) list.sort(cmp);
  return list;
}

function renderVisible() {
  const grid = $("#grid");
  const list = getVisible();
  const total = sourceList().length;

  $("#toolbar").hidden = total === 0;

  if (state.view === "saved" && favorites.size === 0) {
    $("#resultCount").textContent = "";
    grid.innerHTML = `<div class="state">No saved restaurants yet. Tap the ♥ on any card to save it.</div>`;
    return;
  }
  if (list.length === 0) {
    $("#resultCount").textContent = `0 of ${total}`;
    grid.innerHTML = `<div class="state">No matches for these filters. Try loosening them.</div>`;
    return;
  }
  $("#resultCount").textContent = list.length === total
    ? `${total} place${total !== 1 ? "s" : ""}`
    : `${list.length} of ${total}`;

  grid.innerHTML = "";
  list.forEach((r, i) => {
    const el = card(r);
    el.style.animationDelay = `${Math.min(i, 8) * 40}ms`;
    grid.appendChild(el);
  });
}

function skeletons(n) {
  return Array.from({ length: n }, () =>
    `<div class="card skeleton">
       <div class="sk-img"></div>
       <div class="sk-body">
         <div class="sk-line sk-sm"></div>
         <div class="sk-line sk-lg"></div>
         <div class="sk-line sk-md"></div>
       </div>
     </div>`).join("");
}

// --------------------------------------------------------------------------- //
// Card
// --------------------------------------------------------------------------- //
function card(r) {
  const el = document.createElement("article");
  el.className = "card card-in";
  el.tabIndex = 0;
  el.setAttribute("role", "button");

  const media = document.createElement("div");
  media.className = "card-media";

  const img = document.createElement("img");
  img.className = "card-img";
  img.loading = "lazy";
  img.src = r.image_url;
  img.alt = `${r.name} — ${r.cuisine}`;
  img.onerror = () => { img.style.visibility = "hidden"; };
  media.appendChild(img);
  media.appendChild(favButton(r));

  const body = document.createElement("div");
  body.className = "card-body";
  body.innerHTML = `
    <span class="card-cuisine">${esc(r.cuisine)}</span>
    <h3 class="card-name">${esc(r.name)}</h3>
    <span class="card-meta">${esc(r.price)} · ${esc(r.city)}${ratingChip(r)}</span>
    <div class="badges">
      <span class="badge badge-score">match ${(r.score).toFixed(2)}</span>
      ${r.veg_friendly ? '<span class="badge badge-veg">Veg-friendly</span>' : ""}
      ${r.open_now ? '<span class="badge badge-open">Open now</span>' : ""}
      ${r.cold_item ? '<span class="badge badge-new">New</span>' : ""}
    </div>
    ${whyBars(r.why)}
  `;

  el.appendChild(media);
  el.appendChild(body);

  const open = (e) => {
    if (e.target.closest(".fav-btn")) return; // heart handles its own click
    openModal(r);
  };
  el.addEventListener("click", open);
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModal(r); }
  });
  return el;
}

function ratingChip(r) {
  if (r.avg_rating == null) return "";
  return ` · ★ ${r.avg_rating.toFixed(1)} <span class="rc-count">(${r.num_reviews})</span>`;
}

function favButton(r) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "fav-btn" + (favorites.has(r.restaurant_id) ? " active" : "");
  btn.setAttribute("aria-label", "Save restaurant");
  btn.innerHTML = favorites.has(r.restaurant_id) ? "♥" : "♡";
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleFavorite(r, btn);
  });
  return btn;
}

function whyBars(why) {
  const rows = [
    ["text", "Review text"],
    ["image", "Food image"],
    ["structured", "Attributes"],
  ];
  const bars = rows.map(([key, label]) => {
    const pct = Math.round((why[key] || 0) * 100);
    const [, tip] = WHY_INFO[key];
    return `
      <div class="why-row" tabindex="0" data-tip="${esc(tip)}">
        <span class="why-label">${label}</span>
        <span class="why-track"><span class="why-fill ${key}" style="width:${pct}%"></span></span>
        <span class="why-pct">${pct}%</span>
      </div>`;
  }).join("");
  return `<div class="why">${bars}</div>`;
}

// --------------------------------------------------------------------------- //
// Favorites
// --------------------------------------------------------------------------- //
function loadFavorites() {
  try {
    const raw = JSON.parse(localStorage.getItem(LS.favorites)) || {};
    return new Map(Object.entries(raw).map(([k, v]) => [Number(k), v]));
  } catch (e) { return new Map(); }
}

function saveFavorites() {
  const obj = {};
  favorites.forEach((v, k) => { obj[k] = v; });
  try { localStorage.setItem(LS.favorites, JSON.stringify(obj)); } catch (e) {}
}

function toggleFavorite(r, btn) {
  if (favorites.has(r.restaurant_id)) {
    favorites.delete(r.restaurant_id);
    toast(`Removed “${r.name}” from saved.`);
  } else {
    favorites.set(r.restaurant_id, r);
    toast(`Saved “${r.name}”. ♥`);
  }
  saveFavorites();
  updateSavedCount();
  if (btn) {
    const active = favorites.has(r.restaurant_id);
    btn.classList.toggle("active", active);
    btn.innerHTML = active ? "♥" : "♡";
  }
  if (state.view === "saved") renderVisible();
}

function updateSavedCount() {
  $("#savedCount").textContent = favorites.size;
}

function toggleSavedView() {
  state.view = state.view === "saved" ? "results" : "saved";
  updateSavedBtn();

  if (state.view === "saved") {
    showResults();
    $("#resultsTitle").textContent = "Saved restaurants";
    $("#resultsSub").textContent = "Your favorites, stored on this device.";
    renderVisible();
  } else if (state.lastQuery) {
    $("#resultsTitle").textContent = state.lastQuery.title;
    $("#resultsSub").textContent = state.lastQuery.subtitle;
    renderVisible();
  } else {
    $("#resultsSection").hidden = true;
  }
}

function updateSavedBtn() {
  const on = state.view === "saved";
  $("#savedBtn").classList.toggle("active", on);
  $("#savedBtn").setAttribute("aria-pressed", String(on));
}

// --------------------------------------------------------------------------- //
// Card detail modal
// --------------------------------------------------------------------------- //
function openModal(r) {
  const saved = favorites.has(r.restaurant_id);
  $("#modalBody").innerHTML = `
    <div class="modal-media">
      <img src="${esc(r.image_url)}" alt="${esc(r.name)}" onerror="this.style.visibility='hidden'"/>
      <button type="button" class="fav-btn modal-fav ${saved ? "active" : ""}" id="modalFav" aria-label="Save restaurant">${saved ? "♥" : "♡"}</button>
    </div>
    <div class="modal-info">
      <span class="card-cuisine">${esc(r.cuisine)}</span>
      <h3 id="modalName">${esc(r.name)}</h3>
      <p class="modal-meta">${esc(r.price)} · ${esc(r.neighborhood || r.city)}, ${esc(r.city)}${
        r.avg_rating != null ? ` · ★ ${r.avg_rating.toFixed(1)} (${r.num_reviews} reviews)` : ""}</p>
      <div class="badges">
        <span class="badge badge-score">match ${r.score.toFixed(2)}</span>
        ${r.veg_friendly ? '<span class="badge badge-veg">Veg-friendly</span>' : ""}
        ${r.open_now ? '<span class="badge badge-open">Open now</span>' : ""}
        ${r.cold_item ? '<span class="badge badge-new">New</span>' : ""}
      </div>
      ${r.description ? `<p class="modal-desc">${esc(r.description)}</p>` : ""}
      ${r.review ? `<blockquote class="modal-review">“${esc(r.review)}”</blockquote>` : ""}
      <h4 class="modal-why-h">Why this matched you</h4>
      ${whyBars(r.why)}
      <button type="button" class="btn-primary modal-more" id="modalMore">More like this</button>
    </div>
  `;
  $("#modalFav").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleFavorite(r, e.currentTarget);
    // keep grid hearts in sync
    renderVisible();
  });
  $("#modalMore").addEventListener("click", () => moreLikeThis(r));
  $("#modal").hidden = false;
  document.body.classList.add("modal-open");
}

function closeModal() {
  $("#modal").hidden = true;
  document.body.classList.remove("modal-open");
}

function moreLikeThis(r) {
  closeModal();
  // Reset onboarding UI to this restaurant's cuisine + price, then query.
  state.selectedCuisines = new Set([r.cuisine]);
  state.price = r.price_band;
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.dataset.cuisine === r.cuisine));
  $("#price").querySelectorAll(".price-btn").forEach((b) =>
    b.classList.toggle("active", Number(b.dataset.band) === r.price_band));
  updateHint();
  saveLastSearch();
  const payload = { cuisines: [r.cuisine], price: r.price_band, city: null, k: 9 };
  runQuery("/api/recommend/cold", payload,
    `More like ${r.name}`, `${r.cuisine} · ${PRICE_LABELS[r.price_band]}`);
}

// --------------------------------------------------------------------------- //
// Sample-image gallery
// --------------------------------------------------------------------------- //
function renderGallery() {
  const box = $("#gallery");
  if (!box) return;
  // A spread of restaurant ids (images ship as /images/rest_XXXX.png).
  const ids = [3, 27, 61, 90, 132, 178, 210, 258];
  box.innerHTML = ids.map((id) => {
    const src = `/images/rest_${String(id).padStart(4, "0")}.png`;
    return `<div class="gallery-item"><img loading="lazy" src="${src}" alt="Sample food photo"
              onerror="this.parentElement.style.display='none'"/></div>`;
  }).join("");
}

// --------------------------------------------------------------------------- //
// Theme
// --------------------------------------------------------------------------- //
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const icon = $("#themeToggle")?.querySelector(".theme-icon");
  if (icon) icon.textContent = theme === "dark" ? "☀️" : "🌙";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem(LS.theme, next); } catch (e) {}
}

// --------------------------------------------------------------------------- //
// Toasts
// --------------------------------------------------------------------------- //
function toast(message, type = "info") {
  const host = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  host.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, type === "error" ? 4200 : 2600);
}

// --------------------------------------------------------------------------- //
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

init();
