// openodke.dev: the page head (section eyebrow, title, page menu).
// Every page is also published as Markdown at the same path plus ".md"
// (scripts/docs_extras.py writes those, and /llms.txt), so a reader can hand
// a page to a coding agent without copying it out of the browser.

(() => {
  const icon = (paths) =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;
  const ICONS = {
    copy: icon('<rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'),
    check: icon('<path d="M20 6 9 17l-5-5"/>'),
    chevron: icon('<path d="m6 9 6 6 6-6"/>'),
    file: icon('<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 13H8"/><path d="M16 17H8"/>'),
    external: icon('<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'),
    pencil: icon('<path d="M21.17 6.81a1 1 0 0 0-3.99-3.99L3.84 16.17a2 2 0 0 0-.5.83l-1.32 4.35a.5.5 0 0 0 .62.62l4.35-1.32a2 2 0 0 0 .83-.5z"/>'),
  };

  // The page's Markdown, on this host and on the published site.
  const root = () => new URL(window.__md_scope || "/", location.href);
  const relative = () => {
    let path = location.pathname.slice(root().pathname.length).replace(/index\.html$/, "");
    return (path.replace(/\/$/, "") || "index") + ".md";
  };
  // openodke.dev serves the site from its root, so the published copy is there.
  const canonical = () => {
    const link = document.querySelector('link[rel="canonical"]');
    return new URL(relative(), link ? new URL("/", link.href) : root()).href;
  };

  const ask = (prompt) => encodeURIComponent(prompt);

  function build() {
    const article = document.querySelector(".md-content__inner");
    const h1 = article && article.querySelector("h1");
    if (!h1 || article.querySelector(".oo-page-head")) return;

    const head = document.createElement("div");
    head.className = "oo-page-head";
    const titles = document.createElement("div");
    const tab = document.querySelector(".md-tabs__item--active .md-tabs__link");
    if (tab) {
      const eyebrow = document.createElement("p");
      eyebrow.className = "oo-eyebrow";
      eyebrow.textContent = tab.textContent.trim();
      titles.append(eyebrow);
    }
    h1.replaceWith(head);
    titles.append(h1);

    const md = new URL(relative(), root()).href;
    const published = canonical();
    const prompt = `Read ${published} so I can ask questions about it.`;
    const edit = document.querySelector('a.md-content__button[href*="/edit/"]');

    const actions = document.createElement("div");
    actions.className = "oo-actions";
    actions.innerHTML = `
      <button type="button" class="oo-copy" title="Copy this page as Markdown">${ICONS.copy}<span>Copy page</span></button>
      <button type="button" class="oo-more" aria-haspopup="menu" aria-label="More page options">${ICONS.chevron}</button>
      <div class="oo-menu" role="menu">
        <button type="button" class="oo-copy" role="menuitem">${ICONS.copy}<div><strong>Copy page</strong><span>As Markdown, for a coding agent</span></div></button>
        <a role="menuitem" href="${md}" target="_blank" rel="noopener">${ICONS.file}<div><strong>View as Markdown</strong><span>This page as plain text</span></div></a>
        <a role="menuitem" href="https://claude.ai/new?q=${ask(prompt)}" target="_blank" rel="noopener">${ICONS.external}<div><strong>Open in Claude</strong><span>Ask questions about this page</span></div></a>
        <a role="menuitem" href="https://chatgpt.com/?q=${ask(prompt)}" target="_blank" rel="noopener">${ICONS.external}<div><strong>Open in ChatGPT</strong><span>Ask questions about this page</span></div></a>
        ${edit ? `<a role="menuitem" href="${edit.href}" target="_blank" rel="noopener">${ICONS.pencil}<div><strong>Edit on GitHub</strong><span>Suggest a change to this page</span></div></a>` : ""}
      </div>`;
    head.append(titles, actions);

    const more = actions.querySelector(".oo-more");
    more.addEventListener("click", (event) => {
      event.stopPropagation();
      actions.toggleAttribute("data-open");
    });
    document.addEventListener("click", () => actions.removeAttribute("data-open"));

    for (const button of actions.querySelectorAll(".oo-copy")) {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        try {
          const text = await (await fetch(md)).text();
          await navigator.clipboard.writeText(text);
          actions.querySelector(".oo-copy").innerHTML = `${ICONS.check}<span>Copied</span>`;
        } catch {
          window.open(md, "_blank", "noopener");
        }
        actions.removeAttribute("data-open");
        setTimeout(() => {
          actions.querySelector(".oo-copy").innerHTML = `${ICONS.copy}<span>Copy page</span>`;
        }, 2000);
      });
    }
  }

  // Instant navigation swaps the page without a reload; build on every page.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(build);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
