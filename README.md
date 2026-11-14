# Portfólio Vanilla (HTML, CSS e JavaScript Puro)

Site/blog pessoal de **José Dercy**, construído **100% em Vanilla Web puro** — **zero build tools, zero bundlers, zero `npm install`**.

**Site no ar:** [josedercysilva.github.io](https://josedercysilva.github.io)

---

## Como Iniciar o Servidor Local

Como os navegadores bloqueiam requisições `fetch()` de arquivos locais (`file://`) por CORS, execute na raiz do projeto qualquer servidor HTTP local:

```bash
# Opção 1: Com Python 3 nativo (Recomendado)
python3 -m http.server 3000

# Opção 2: Com npx live-server (com hot-reload)
npx live-server --port=3000
```

Abra no navegador: `http://localhost:3000`

---

## Arquitetura do Projeto

- **HTML (`*.html`)**: Estrutura das páginas e conteúdo principal da página inicial (`data-lang="pt"` / `data-lang="en"`).
- **Markdown (`posts/{pt,en}/*.md`)**: Conteúdo editorial — artigos e página *Sobre Mim* (`sobre.md`).
- **JavaScript (`js/`)**: Rótulos curtos de interface (`i18n.js`), metadados dos artigos (`posts-data.js`) e lógica de renderização (`main.js`).

```text
josedercysilva.github.io/
├── index.html                  # Página inicial (Hero, Core Skills, Publicações)
├── sobre.html                  # Página Sobre Mim (carrega posts/{pt,en}/sobre.md)
├── arquivo.html                # Índice cronológico completo agrupado por ano
├── post.html                   # Leitor universal de artigos (?slug=...)
├── img/
│   └── foto_josedercysilva.jpg # Foto do perfil
├── css/
│   └── style.css               # Estilos minimalistas + Temas (Dark Charcoal / Light Papel)
├── js/
│   ├── main.js                 # Controlador: tema, i18n via data-i18n, fetch e renderização
│   ├── i18n.js                 # Dicionário enxuto de UI (PT / EN) + dados de Core Skills
│   └── posts-data.js           # Metadados dos artigos publicados
└── posts/
    ├── pt/                     # Conteúdo em Markdown (Português)
    │   └── sobre.md            # Bio e contatos da página Sobre
    └── en/                     # Conteúdo em Markdown (Inglês)
        └── sobre.md            # Bio e contatos da página About
```

---

## Recursos Implementados

1. **Estética Minimalista de Engenharia:**
   - Tema escuro padrão (`#2a2a2a` com destaque `#64ffda`) e tema claro em tom papel (`#f0f0e8` com destaque `#d85d2a`), com tipografia `Inter` e `Fira Code`.
   - Largura contida (`max-width: 800px`) focada em legibilidade.

2. **Interruptor de Tema:**
   - Detecção automática do tema do sistema operacional (`prefers-color-scheme`) e persistência no `localStorage`.

3. **Bilinguismo Nativo (PT / EN):**
   - Seletor de idioma no cabeçalho (`PT | EN`) com detecção inicial pelo `navigator.language` e troca instantânea sem recarregar a página.

4. **Leitor de Markdown Client-Side:**
   - Renderiza os arquivos `.md` diretamente no navegador usando `marked.min.js`, com parser de fallback offline e remoção automática de `frontmatter`.

---

## Como Adicionar um Novo Artigo

1. Crie os arquivos Markdown em `posts/pt/<seu-slug>.md` e `posts/en/<seu-slug>.md`.
2. Em `js/posts-data.js`, adicione uma nova entrada ao array `POSTS`:

```javascript
{
  slug: 'novo-artigo',
  pubDate: '2026-12-01',
  tags: ['c', 'linux', 'kernel'],
  pt: {
    title: 'Título do Artigo em Português',
    description: 'Resumo técnico do artigo.',
    file: 'posts/pt/novo-artigo.md'
  },
  en: {
    title: 'Article Title in English',
    description: 'Technical summary of the article.',
    file: 'posts/en/novo-artigo.md'
  }
}
```

O artigo aparecerá automaticamente na Home, no Arquivo Histórico e em `post.html?slug=novo-artigo`.
