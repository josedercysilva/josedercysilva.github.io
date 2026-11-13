import { POSTS } from './posts-data.js?v=3';
import { UI } from './i18n.js?v=3';

/* =========================================================
   ESTADO GLOBAL DA APLICAÇÃO
   ========================================================= */
let currentLang = localStorage.getItem('idioma') ||
  ((navigator.language && navigator.language.toLowerCase().startsWith('pt')) ? 'pt' : 'en');

let currentTheme = localStorage.getItem('tema');
if (!currentTheme) {
  const prefereEscuro = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  currentTheme = prefereEscuro ? 'escuro' : 'claro';
}

/* =========================================================
   GERENCIAMENTO DE TEMA (Dark Charcoal / Light)
   ========================================================= */
function aplicarTema(tema, salvarManual = false) {
  currentTheme = tema;
  if (salvarManual) {
    localStorage.setItem('tema', tema);
    localStorage.setItem('tema-manual', 'true');
  }

  const toggle = document.getElementById('toggle-tema');

  if (tema === 'claro') {
    document.body.classList.add('tema-claro');
    if (toggle) toggle.checked = false;
  } else {
    document.body.classList.remove('tema-claro');
    if (toggle) toggle.checked = true;
  }
}

if (window.matchMedia) {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('tema-manual')) {
      aplicarTema(e.matches ? 'escuro' : 'claro', false);
    }
  });
}

/* =========================================================
   GERENCIAMENTO DE IDIOMA (i18n)
   ========================================================= */
function aplicarIdiomaDocumento() {
  document.documentElement.lang = currentLang === 'pt' ? 'pt-BR' : 'en';
}

function setIdioma(novoIdioma) {
  if (novoIdioma !== 'pt' && novoIdioma !== 'en') return;
  currentLang = novoIdioma;
  localStorage.setItem('idioma', novoIdioma);
  aplicarIdiomaDocumento();
  atualizarBotoesIdioma();
  renderizarPagina();
}

function atualizarBotoesIdioma() {
  const btnPt = document.getElementById('btn-lang-pt');
  const btnEn = document.getElementById('btn-lang-en');
  if (btnPt) btnPt.classList.toggle('ativo', currentLang === 'pt');
  if (btnEn) btnEn.classList.toggle('ativo', currentLang === 'en');
}

function aplicarTraducoesUI(t) {
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key && t[key]) {
      el.textContent = t[key];
    }
  });
}

/* =========================================================
   UTILITÁRIOS: DATA, MARKDOWN & COMPONENTES COMPARTILHADOS
   ========================================================= */
function formatarData(dataStr, lang, formato = 'long') {
  const [ano, mes, dia] = dataStr.split('-').map(Number);
  const data = new Date(ano, mes - 1, dia);
  const locale = lang === 'pt' ? 'pt-BR' : 'en-US';

  if (formato === 'short') {
    return new Intl.DateTimeFormat(locale, {
      month: 'short',
      day: 'numeric'
    }).format(data);
  }

  return new Intl.DateTimeFormat(locale, {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  }).format(data);
}

function stripFrontmatter(text) {
  return text.replace(/^---[\s\S]*?---\s*/, '');
}

function parseMarkdownFallback(text) {
  let html = text.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/gim, (match, lang, code) => {
    const escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<pre><code class="language-${lang}">${escaped}</code></pre>`;
  });

  html = html.replace(/^---$/gim, '<hr>');
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
  html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/gim, '<ul>$1</ul>');
  html = html.replace(/<\/ul>\s*<ul>/gim, '');
  html = html.replace(/\*\*([^*]+)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/gim, '<em>$1</em>');
  html = html.replace(/`([^`]+)`/gim, '<code>$1</code>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  return html.split(/\n\n+/).map((p) => {
    const trimmed = p.trim();
    if (!trimmed) return '';
    if (
      trimmed.startsWith('<h') ||
      trimmed.startsWith('<pre') ||
      trimmed.startsWith('<ul') ||
      trimmed.startsWith('<blockquote') ||
      trimmed.startsWith('<hr')
    ) {
      return trimmed;
    }
    return `<p>${trimmed}</p>`;
  }).join('\n');
}

function renderMarkdown(mdText) {
  const textoLimpo = stripFrontmatter(mdText);
  if (window.marked && typeof window.marked.parse === 'function') {
    return window.marked.parse(textoLimpo);
  }
  return parseMarkdownFallback(textoLimpo);
}

async function carregarMarkdownNoElemento(filePath, targetEl, loadingText = 'Carregando...') {
  if (!targetEl) return;
  try {
    targetEl.innerHTML = `<p style="color: var(--text-muted); font-family: var(--font-mono);">${loadingText}</p>`;
    const res = await fetch(filePath);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const markdown = await res.text();
    targetEl.innerHTML = renderMarkdown(markdown);
  } catch (err) {
    console.error('Erro ao carregar Markdown:', err);
    targetEl.innerHTML = `<p style="color: #ef4444; font-family: var(--font-mono);">Erro ao carregar conteúdo (${err.message}). Certifique-se de estar rodando via servidor HTTP (ex: python3 -m http.server).</p>`;
  }
}

function renderEmptyPublicationsBox(t) {
  return `
    <div class="empty-publications-box">
      <div class="empty-pub-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
      </div>
      <h4 class="empty-pub-title">${t['section.empty.title']}</h4>
      <p class="empty-pub-desc">${t['section.empty.desc']}</p>
    </div>
  `;
}

/* =========================================================
   RENDERIZAÇÃO DINÂMICA DAS PÁGINAS
   ========================================================= */
function renderizarHome() {
  const t = UI[currentLang];

  // Seção Editorial de Habilidades (4 Colunas)
  const skillsSection = document.getElementById('skills-section');
  if (skillsSection && t.skills) {
    const sData = t.skills;
    const columnsHtml = sData.columns.map(col => `
      <div class="skills-editorial-col">
        <h4 class="skills-editorial-col-title">${col.title}</h4>
        <ul class="skills-editorial-list">
          ${col.items.map(item => `<li>${item}</li>`).join('')}
        </ul>
      </div>
    `).join('');

    skillsSection.innerHTML = `
      <div class="skills-editorial-label">${sData.label}</div>
      <div class="skills-editorial-grid">
        ${columnsHtml}
      </div>
    `;
  }

  // Publicações Recentes
  const postsOrdenados = [...POSTS].sort((a, b) => new Date(b.pubDate) - new Date(a.pubDate));
  const recentPosts = postsOrdenados.slice(0, 10);

  const elCounter = document.getElementById('section-recent-counter');
  const containerPosts = document.getElementById('recent-posts-list');
  const elArchiveContainer = document.querySelector('.archive-action-container');

  if (postsOrdenados.length === 0) {
    if (elCounter) elCounter.textContent = t['section.empty.badge'];
    if (containerPosts) containerPosts.innerHTML = renderEmptyPublicationsBox(t);
    if (elArchiveContainer) elArchiveContainer.style.display = 'none';
    return;
  }

  if (elCounter) {
    elCounter.textContent = t['section.recent.showing']
      .replace('{count}', String(recentPosts.length))
      .replace('{total}', String(postsOrdenados.length));
  }
  if (elArchiveContainer) elArchiveContainer.style.display = '';

  if (containerPosts) {
    containerPosts.innerHTML = recentPosts.map((post) => {
      const info = post[currentLang] || post['pt'];
      const dataFormatada = formatarData(post.pubDate, currentLang);
      const tagsHtml = post.tags && post.tags.length > 0
        ? `<div class="post-tags">${post.tags.map(tag => `<span class="tag">#${tag}</span>`).join('')}</div>`
        : '';

      return `
        <a href="post.html?slug=${encodeURIComponent(post.slug)}" class="post-card">
          <div class="post-card-date">${dataFormatada}</div>
          <div class="post-card-title">${info.title}</div>
          <div class="post-card-desc">${info.description}</div>
          ${tagsHtml}
        </a>
      `;
    }).join('');
  }

  const elArchiveBtn = document.getElementById('archive-cta-btn');
  if (elArchiveBtn) {
    elArchiveBtn.textContent = t['archive.btn'].replace('{count}', String(postsOrdenados.length));
  }
}

function renderizarArquivo() {
  const t = UI[currentLang];
  const postsOrdenados = [...POSTS].sort((a, b) => new Date(b.pubDate) - new Date(a.pubDate));

  const elDesc = document.getElementById('archive-desc');
  const container = document.getElementById('archive-years-container');

  if (postsOrdenados.length === 0) {
    if (elDesc) elDesc.textContent = t['section.empty.desc'];
    if (container) container.innerHTML = renderEmptyPublicationsBox(t);
    return;
  }

  if (elDesc) {
    elDesc.textContent = t['archive.desc'].replace('{count}', String(postsOrdenados.length));
  }

  const postsPorAno = postsOrdenados.reduce((acc, post) => {
    const ano = post.pubDate.split('-')[0];
    if (!acc[ano]) acc[ano] = [];
    acc[ano].push(post);
    return acc;
  }, {});

  const anos = Object.keys(postsPorAno).sort((a, b) => Number(b) - Number(a));

  if (container) {
    container.innerHTML = anos.map((ano) => {
      const listaHtml = postsPorAno[ano].map((post) => {
        const info = post[currentLang] || post['pt'];
        const dataFormatada = formatarData(post.pubDate, currentLang, 'short');
        const tagsHtml = post.tags && post.tags.length > 0
          ? `<div style="margin-top: 0.4rem; display: flex; gap: 0.4rem; flex-wrap: wrap;">
              ${post.tags.map(tag => `<span class="tag" style="font-size: 0.7rem; padding: 0.1rem 0.5rem;">#${tag}</span>`).join('')}
             </div>`
          : '';

        return `
          <a href="post.html?slug=${encodeURIComponent(post.slug)}" class="archive-row">
            <div>
              <span class="archive-row-title">${info.title}</span>
              ${tagsHtml}
            </div>
            <span class="archive-row-date">${dataFormatada}</span>
          </a>
        `;
      }).join('');

      return `
        <div style="margin-bottom: 3rem;">
          <h2 style="font-family: var(--font-mono); color: var(--accent); font-size: 1.3rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem;">
            ${ano}
          </h2>
          <div>
            ${listaHtml}
          </div>
        </div>
      `;
    }).join('');
  }
}

async function renderizarPost() {
  const t = UI[currentLang];
  const params = new URLSearchParams(window.location.search);
  const slug = params.get('slug');

  const post = slug ? POSTS.find(p => p.slug === slug) : POSTS[0];

  const outroIdioma = currentLang === 'pt' ? 'en' : 'pt';
  const elNotice = document.getElementById('post-translation-btn');
  if (elNotice) {
    elNotice.textContent = `${t['post.translationNotice']} →`;
    elNotice.onclick = (e) => {
      e.preventDefault();
      setIdioma(outroIdioma);
    };
  }

  const elDate = document.getElementById('post-date');
  const elTitle = document.getElementById('post-title');
  const elDesc = document.getElementById('post-desc');
  const elTags = document.getElementById('post-tags');
  const elBody = document.getElementById('post-body');

  if (!post) {
    if (elDate) elDate.textContent = '404';
    if (elTitle) elTitle.textContent = t['post.notFoundTitle'];
    if (elDesc) elDesc.textContent = t['post.notFoundDesc'];
    if (elTags) elTags.innerHTML = '';
    if (elBody) {
      elBody.innerHTML = `<p><a href="arquivo.html">${t['archive.title']} →</a></p>`;
    }
    document.title = `${t['post.notFoundTitle']} | José Dercy`;
    return;
  }

  const info = post[currentLang] || post['pt'];

  if (elDate) {
    const [ano, mes, dia] = post.pubDate.split('-').map(Number);
    const dataObj = new Date(ano, mes - 1, dia);
    elDate.textContent = new Intl.DateTimeFormat(currentLang === 'pt' ? 'pt-BR' : 'en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }).format(dataObj);
  }

  if (elTitle) elTitle.textContent = info.title;
  document.title = `${info.title} | José Dercy`;

  if (elDesc) elDesc.textContent = info.description;

  if (elTags) {
    elTags.innerHTML = (post.tags || []).map(tag => `<span class="tag">#${tag}</span>`).join('');
  }

  await carregarMarkdownNoElemento(info.file, elBody, 'Carregando artigo...');
}

async function renderizarSobre() {
  const t = UI[currentLang];
  const outroIdioma = currentLang === 'pt' ? 'en' : 'pt';

  const elNotice = document.getElementById('about-translation-btn');
  if (elNotice) {
    elNotice.textContent = `${t['about.translationNotice']} →`;
    elNotice.onclick = (e) => {
      e.preventDefault();
      setIdioma(outroIdioma);
    };
  }

  document.title = `${t['about.title']} | José Dercy`;

  await carregarMarkdownNoElemento(
    `posts/${currentLang}/sobre.md`,
    document.getElementById('about-body'),
    'Carregando...'
  );
}

function renderizarPagina() {
  const t = UI[currentLang];
  aplicarTraducoesUI(t);

  if (document.getElementById('recent-posts-list')) {
    renderizarHome();
  } else if (document.getElementById('archive-years-container')) {
    renderizarArquivo();
  } else if (document.getElementById('post-body')) {
    renderizarPost();
  } else if (document.getElementById('about-page-container')) {
    renderizarSobre();
  }
}

function atualizarNavAtiva() {
  const path = window.location.pathname;
  const navHome = document.getElementById('nav-home');
  const navAbout = document.getElementById('nav-about');
  const navArchive = document.getElementById('nav-archive');

  [navHome, navAbout, navArchive].forEach((el) => {
    if (el) el.classList.remove('active');
  });

  if (path.includes('arquivo.html')) {
    if (navArchive) navArchive.classList.add('active');
  } else if (path.includes('sobre.html')) {
    if (navAbout) navAbout.classList.add('active');
  } else if (!path.includes('post.html')) {
    if (navHome) navHome.classList.add('active');
  }
}

/* =========================================================
   INICIALIZAÇÃO NO CARREGAMENTO DA PÁGINA
   ========================================================= */
document.addEventListener('DOMContentLoaded', () => {
  aplicarTema(currentTheme);
  aplicarIdiomaDocumento();

  const toggleTema = document.getElementById('toggle-tema');
  if (toggleTema) {
    toggleTema.addEventListener('change', () => {
      aplicarTema(toggleTema.checked ? 'escuro' : 'claro', true);
    });
  }

  const btnPt = document.getElementById('btn-lang-pt');
  if (btnPt) btnPt.addEventListener('click', () => setIdioma('pt'));

  const btnEn = document.getElementById('btn-lang-en');
  if (btnEn) btnEn.addEventListener('click', () => setIdioma('en'));

  atualizarBotoesIdioma();
  atualizarNavAtiva();

  const navBrand = document.getElementById('nav-brand');
  const secPublicacoes = document.getElementById('publicacoes');
  if (navBrand && secPublicacoes) {
    navBrand.addEventListener('click', (e) => {
      e.preventDefault();
      secPublicacoes.scrollIntoView({ behavior: 'smooth' });
      history.pushState(null, '', '#publicacoes');
    });
  }

  renderizarPagina();
});
