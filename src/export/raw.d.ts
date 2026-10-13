// Vite `?raw` imports: the file's contents as a string. Used so the exporter
// ships (and verifies) the exact runtime source text that lives in the repo.
declare module "*?raw" {
  const content: string;
  export default content;
}
