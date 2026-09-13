function byId(id) { return document.getElementById(id); }

/* ── Mobile nav ── */
function toggleNav() {
  byId("nav-links")?.classList.toggle("open");
  document.querySelector(".nav-toggle")?.classList.toggle("is-open");
}
function closeNav() {
  byId("nav-links")?.classList.remove("open");
  document.querySelector(".nav-toggle")?.classList.remove("is-open");
}

/* ── Scroll progress / nav shrink / back-to-top ── */
function updateScrollProgress() {
  const bar = byId("scrollProgress");
  if (!bar) return;
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  bar.style.width = (docHeight > 0 ? (scrollTop / docHeight) * 100 : 0) + "%";
}
function updateNavScroll() {
  byId("siteNav")?.classList.toggle("scrolled", window.scrollY > 60);
}
function updateBackToTop() {
  byId("backToTop")?.classList.toggle("is-visible", window.scrollY > 400);
}
function onScroll() {
  updateScrollProgress();
  updateNavScroll();
  updateBackToTop();
}
window.addEventListener("scroll", onScroll, { passive: true });

/* ── Scroll-spy nav highlighting ── */
function initScrollSpy() {
  const navAnchors = document.querySelectorAll(".nav-links a[href^='#']");
  const sectionMap = [];
  navAnchors.forEach((a) => {
    const target = document.getElementById(a.getAttribute("href").slice(1));
    if (target) sectionMap.push({ el: target, link: a });
  });
  if (!sectionMap.length) return;

  function update() {
    const scrollY = window.scrollY + 120;
    let current = null;
    for (let i = sectionMap.length - 1; i >= 0; i--) {
      if (sectionMap[i].el.offsetTop <= scrollY) { current = sectionMap[i]; break; }
    }
    navAnchors.forEach((a) => a.classList.remove("is-active"));
    if (current) current.link.classList.add("is-active");
  }
  window.addEventListener("scroll", update, { passive: true });
  update();
  navAnchors.forEach((a) => a.addEventListener("click", closeNav));
}

/* ── Animated counters ── */
function initAnimatedCounters() {
  const counters = document.querySelectorAll(".metric-card strong[data-count], .stat-num[data-count]");
  if (!counters.length) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      if (el.dataset.counted) return;
      el.dataset.counted = "1";
      const target = parseFloat(el.dataset.count);
      const suffix = el.dataset.suffix || "";
      const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals, 10) : 0;
      const duration = 1200;
      const start = performance.now();
      function tick(now) {
        const p = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = (target * eased).toFixed(decimals) + suffix;
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.4 });
  counters.forEach((c) => observer.observe(c));
}

/* ── Reveal on scroll ── */
function initReveal() {
  const targets = document.querySelectorAll(
    ".hero-sidecard, .metric-card, .card, .resource-card, .figure-card, .chart-card, .paper-sheet, .policy-card, .profile-card"
  );
  if (!targets.length) return;
  if (!("IntersectionObserver" in window)) { targets.forEach((el) => el.classList.add("revealed")); return; }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) { entry.target.classList.add("revealed"); observer.unobserve(entry.target); }
    });
  }, { threshold: 0.14, rootMargin: "200px 0px -10px 0px" });
  targets.forEach((el, idx) => {
    el.classList.add("is-reveal");
    el.style.transitionDelay = `${Math.min(idx * 15, 180)}ms`;
    observer.observe(el);
  });

  // Safety net: a very fast scroll (keyboard End, an aggressive flick, or
  // assistive tooling that doesn't fire granular scroll/intersection
  // events) can jump past a section before it ever intersects the
  // viewport, leaving it stuck at opacity:0 forever. Force-reveal anything
  // still hidden after a short delay so content is never permanently lost.
  setTimeout(() => {
    document.querySelectorAll(".is-reveal:not(.revealed)").forEach((el) => el.classList.add("revealed"));
  }, 2500);
}

/* ── Chart lightbox ── */
function initLightbox() {
  const lightbox = byId("lightbox");
  const lightboxImg = byId("lightbox-img");
  if (!lightbox || !lightboxImg) return;
  document.querySelectorAll(".chart-card").forEach((card) => {
    card.addEventListener("click", () => {
      const img = card.querySelector("img");
      if (!img) return;
      lightboxImg.src = img.src;
      lightboxImg.alt = img.alt;
      lightbox.classList.add("open");
    });
  });
  lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox || e.target.classList.contains("lightbox-close")) {
      lightbox.classList.remove("open");
      lightboxImg.src = "";
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { lightbox.classList.remove("open"); lightboxImg.src = ""; }
  });
}

/* ── Init ── */
document.addEventListener("DOMContentLoaded", () => {
  initScrollSpy();
  initAnimatedCounters();
  initReveal();
  initLightbox();
  onScroll();

  const yearEl = byId("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
});
