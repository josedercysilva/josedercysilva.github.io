// Metadados dos artigos publicados (adicione aqui conforme publicar novos arquivos em posts/{pt,en}/)
export const POSTS = [
  {
    slug: 'cent-runtime',
    pubDate: '2026-11-19',
    tags: ['c++17', 'pim', 'cxl', 'llm', 'pytorch'],
    pt: {
      title: 'cent_runtime: Runtime C++17 e C ABI para Arquitetura Near-Bank PIM (CENT)',
      description: 'Relatório técnico da implementação de um runtime em C++17 com C ABI nativa para despachar operações GEMV do Llama-2 7B em GDDR6-AiM via CXL 3.0.',
      file: 'posts/pt/cent-runtime.md'
    },
    en: {
      title: 'cent_runtime: C++17 Runtime and C ABI for the CENT Near-Bank PIM Architecture',
      description: 'Technical report on implementing a C++17 runtime with a native C ABI to dispatch Llama-2 7B GEMV layers onto GDDR6-AiM over CXL 3.0.',
      file: 'posts/en/cent-runtime.md'
    }
  }
];
