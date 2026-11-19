// Metadados dos artigos publicados (adicione aqui conforme publicar novos arquivos em posts/{pt,en}/)
export const POSTS = [
  {
    slug: 'cent-runtime',
    pubDate: '2026-11-19',
    tags: ['c++17', 'pim', 'cxl', 'llm', 'pytorch'],
    pt: {
      title: 'cent_runtime: Conectando PyTorch e NumPy à Arquitetura Near-Bank PIM (CENT)',
      description: 'Como desenvolvi um runtime em C++17 com fronteira C ABI para despachar camadas lineares (GEMV) do Llama-2 7B sobre GDDR6-AiM via CXL 3.0.',
      file: 'posts/pt/cent-runtime.md'
    },
    en: {
      title: 'cent_runtime: Bridging PyTorch and NumPy to the CENT Near-Bank PIM Architecture',
      description: 'How I built a C++17 runtime with a native C ABI boundary to dispatch Llama-2 7B linear layers (GEMV) onto GDDR6-AiM over CXL 3.0.',
      file: 'posts/en/cent-runtime.md'
    }
  }
];
