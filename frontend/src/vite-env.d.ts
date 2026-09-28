/// <reference types="vite/client" />

declare module 'plotly.js/dist/plotly-basic.min.js' {
  const Plotly: typeof import('plotly.js');
  export default Plotly;
}
