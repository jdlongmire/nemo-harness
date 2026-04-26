"""SVG creation and manipulation tools."""

from __future__ import annotations
import json
import re
from pathlib import Path

from tools.sandbox import Sandbox
from tools.registry import ToolDef, TOOL_REGISTRY


def register(sandbox: Sandbox):
    """Register SVG tools with the global registry."""

    def _output_path(filename: str) -> Path:
        out_dir = sandbox.allowed_dirs[0] / 'output'
        out_dir.mkdir(exist_ok=True)
        resolved = (out_dir / filename).resolve()
        sandbox.check_path(str(resolved))
        return resolved

    async def create_svg(filename: str, content: str,
                         width: str = '800', height: str = '600') -> dict:
        """Create an SVG file.

        content can be:
        - Raw SVG markup (elements only, no outer <svg> tag needed)
        - A complete SVG document (detected by leading <svg or <?xml)
        - A JSON object describing the diagram (see structured format below)

        Structured JSON format:
        {
          "background": "#ffffff",
          "elements": [
            {"type": "rect", "x": 10, "y": 10, "width": 100, "height": 50,
             "fill": "#4472C4", "stroke": "#000", "stroke_width": 1, "rx": 5},
            {"type": "circle", "cx": 200, "cy": 100, "r": 40, "fill": "#ED7D31"},
            {"type": "ellipse", "cx": 300, "cy": 100, "rx": 60, "ry": 30, "fill": "#A5A5A5"},
            {"type": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 100,
             "stroke": "#000", "stroke_width": 2},
            {"type": "polyline", "points": "0,0 50,25 100,0", "stroke": "#000",
             "fill": "none", "stroke_width": 2},
            {"type": "polygon", "points": "50,0 100,50 0,50", "fill": "#70AD47"},
            {"type": "text", "x": 50, "y": 50, "text": "Hello",
             "font_size": 16, "font_family": "sans-serif", "fill": "#000",
             "anchor": "middle", "weight": "bold"},
            {"type": "path", "d": "M10 10 L90 90", "stroke": "#000", "fill": "none"},
            {"type": "group", "transform": "translate(100,100)", "elements": [...]},
            {"type": "arrow", "x1": 0, "y1": 0, "x2": 100, "y2": 0,
             "stroke": "#000", "stroke_width": 2}
          ],
          "defs": "<style>...</style>"
        }
        """
        if not filename.endswith('.svg'):
            filename += '.svg'
        out = _output_path(filename)

        stripped = content.strip()

        # Case 1: complete SVG document
        if stripped.startswith('<?xml') or stripped.startswith('<svg'):
            svg_content = stripped
        else:
            # Case 2: try JSON structured format
            try:
                data = json.loads(stripped)
                svg_content = _build_svg_from_json(data, width, height)
            except (json.JSONDecodeError, TypeError):
                # Case 3: raw SVG elements — wrap them
                svg_content = _wrap_svg(stripped, width, height)

        out.write_text(svg_content, encoding='utf-8')
        return {'message': f'Created {filename}', 'path': str(out), 'success': True}

    async def edit_svg(filename: str, find: str, replace: str) -> dict:
        """Find and replace text within an existing SVG file.

        Useful for updating labels, colors, or attributes without regenerating.
        """
        if not filename.endswith('.svg'):
            filename += '.svg'
        out_dir = sandbox.allowed_dirs[0] / 'output'
        path = (out_dir / filename).resolve()
        sandbox.check_path(str(path))

        if not path.exists():
            return {'error': f'File not found: {filename}', 'success': False}

        text = path.read_text(encoding='utf-8')
        if find not in text:
            return {'error': f'Pattern not found in {filename}', 'success': False}

        updated = text.replace(find, replace)
        path.write_text(updated, encoding='utf-8')
        count = text.count(find)
        return {
            'message': f'Replaced {count} occurrence(s) in {filename}',
            'path': str(path),
            'success': True,
        }

    def _wrap_svg(inner: str, w: str, h: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
            f'{inner}\n</svg>'
        )

    def _build_svg_from_json(data: dict, w: str, h: str) -> str:
        w = str(data.get('width', w))
        h = str(data.get('height', h))
        bg = data.get('background', '')
        defs = data.get('defs', '')
        elements = data.get('elements', [])

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        ]

        # Arrowhead marker (included if any arrow elements exist)
        needs_arrow = _has_arrow(elements)
        if defs or needs_arrow:
            parts.append('  <defs>')
            if needs_arrow:
                parts.append(
                    '    <marker id="arrowhead" markerWidth="10" markerHeight="7" '
                    'refX="10" refY="3.5" orient="auto">'
                )
                parts.append('      <polygon points="0 0, 10 3.5, 0 7" fill="context-stroke"/>')
                parts.append('    </marker>')
            if defs:
                parts.append(f'    {defs}')
            parts.append('  </defs>')

        if bg:
            parts.append(f'  <rect width="100%" height="100%" fill="{bg}"/>')

        for elem in elements:
            parts.append(_render_element(elem, indent=2))

        parts.append('</svg>')
        return '\n'.join(parts)

    def _has_arrow(elements: list) -> bool:
        for e in elements:
            if isinstance(e, dict):
                if e.get('type') == 'arrow':
                    return True
                if e.get('type') == 'group' and _has_arrow(e.get('elements', [])):
                    return True
        return False

    def _render_element(elem: dict, indent: int = 2) -> str:
        pad = ' ' * indent
        t = elem.get('type', '')

        if t == 'rect':
            attrs = _attrs(elem, ['x', 'y', 'width', 'height', 'fill', 'stroke',
                                   'stroke_width', 'rx', 'ry', 'opacity', 'transform'])
            return f'{pad}<rect {attrs}/>'

        elif t == 'circle':
            attrs = _attrs(elem, ['cx', 'cy', 'r', 'fill', 'stroke', 'stroke_width',
                                   'opacity', 'transform'])
            return f'{pad}<circle {attrs}/>'

        elif t == 'ellipse':
            attrs = _attrs(elem, ['cx', 'cy', 'rx', 'ry', 'fill', 'stroke',
                                   'stroke_width', 'opacity', 'transform'])
            return f'{pad}<ellipse {attrs}/>'

        elif t == 'line':
            attrs = _attrs(elem, ['x1', 'y1', 'x2', 'y2', 'stroke', 'stroke_width',
                                   'stroke_dasharray', 'opacity', 'transform'])
            return f'{pad}<line {attrs}/>'

        elif t == 'polyline':
            attrs = _attrs(elem, ['points', 'stroke', 'fill', 'stroke_width',
                                   'stroke_dasharray', 'opacity', 'transform'])
            return f'{pad}<polyline {attrs}/>'

        elif t == 'polygon':
            attrs = _attrs(elem, ['points', 'fill', 'stroke', 'stroke_width',
                                   'opacity', 'transform'])
            return f'{pad}<polygon {attrs}/>'

        elif t == 'text':
            text_val = elem.get('text', '')
            attrs = _attrs(elem, ['x', 'y', 'fill', 'font_size', 'font_family',
                                   'anchor', 'weight', 'opacity', 'transform',
                                   'dominant_baseline'])
            return f'{pad}<text {attrs}>{_escape(text_val)}</text>'

        elif t == 'path':
            attrs = _attrs(elem, ['d', 'fill', 'stroke', 'stroke_width',
                                   'stroke_dasharray', 'opacity', 'transform'])
            return f'{pad}<path {attrs}/>'

        elif t == 'arrow':
            attrs = _attrs(elem, ['x1', 'y1', 'x2', 'y2', 'stroke', 'stroke_width',
                                   'opacity', 'transform'])
            return f'{pad}<line {attrs} marker-end="url(#arrowhead)"/>'

        elif t == 'group':
            transform = elem.get('transform', '')
            children = elem.get('elements', [])
            t_attr = f' transform="{transform}"' if transform else ''
            lines = [f'{pad}<g{t_attr}>']
            for child in children:
                lines.append(_render_element(child, indent + 2))
            lines.append(f'{pad}</g>')
            return '\n'.join(lines)

        elif t == 'image':
            attrs = _attrs(elem, ['x', 'y', 'width', 'height', 'href', 'opacity',
                                   'transform'])
            return f'{pad}<image {attrs}/>'

        else:
            return f'{pad}<!-- unknown element type: {t} -->'

    def _attrs(elem: dict, keys: list[str]) -> str:
        """Build SVG attribute string from element dict, mapping underscores to hyphens."""
        attr_map = {
            'stroke_width': 'stroke-width',
            'stroke_dasharray': 'stroke-dasharray',
            'font_size': 'font-size',
            'font_family': 'font-family',
            'anchor': 'text-anchor',
            'weight': 'font-weight',
            'dominant_baseline': 'dominant-baseline',
            'href': 'href',
        }
        parts = []
        for k in keys:
            v = elem.get(k)
            if v is not None:
                svg_attr = attr_map.get(k, k)
                parts.append(f'{svg_attr}="{v}"')
        return ' '.join(parts)

    def _escape(text: str) -> str:
        """Escape XML special characters."""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

    # Register tools
    TOOL_REGISTRY.register(ToolDef(
        name='create_svg',
        description=(
            'Create an SVG file. Content can be: (1) raw SVG elements, '
            '(2) a complete SVG document, or (3) a JSON object with '
            '"elements" array containing shapes like rect, circle, ellipse, '
            'line, text, path, polygon, arrow, group, image. '
            'Use for diagrams, charts, illustrations, and visual output.'
        ),
        parameters={
            'filename': {'type': 'string', 'description': 'Output filename (e.g. "diagram.svg")'},
            'content': {'type': 'string', 'description': (
                'SVG content: raw elements, complete SVG document, or JSON '
                '{"background": "#fff", "elements": [{"type": "rect", "x": 0, ...}], "defs": "..."}'
            )},
            'width': {'type': 'string', 'description': 'SVG width in pixels (default: 800)'},
            'height': {'type': 'string', 'description': 'SVG height in pixels (default: 600)'},
        },
        required=['filename', 'content'],
        handler=create_svg,
    ))

    TOOL_REGISTRY.register(ToolDef(
        name='edit_svg',
        description=(
            'Find and replace text in an existing SVG file in the output/ directory. '
            'Useful for updating labels, colors, or attributes without regenerating.'
        ),
        parameters={
            'filename': {'type': 'string', 'description': 'SVG filename in output/ (e.g. "diagram.svg")'},
            'find': {'type': 'string', 'description': 'Text to find in the SVG'},
            'replace': {'type': 'string', 'description': 'Replacement text'},
        },
        handler=edit_svg,
    ))
