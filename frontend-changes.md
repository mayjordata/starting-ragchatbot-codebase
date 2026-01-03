# Frontend Changes: Theme Toggle Button

## Overview
Added a light/dark mode toggle button to the UI, allowing users to switch between themes with smooth animations and persistent preference storage.

## Files Modified

### 1. `frontend/index.html`
- Added theme toggle button in the `<body>` before the main container
- Button includes SVG icons for both sun (light mode indicator) and moon (dark mode indicator)
- Accessibility attributes: `aria-label`, `title`, and keyboard focusability

### 2. `frontend/style.css`
- Added light theme CSS variables under `[data-theme="light"]` selector
- Added new `--code-bg` variable for code block backgrounds
- Added `.theme-toggle` button styles with:
  - Fixed positioning in top-right corner
  - Circular design (44x44px)
  - Hover and focus states with visual feedback
  - Active state with scale animation
- Added icon transition animations:
  - Sun icon: rotates and scales out when switching to light mode
  - Moon icon: rotates and scales in when switching to light mode
- Added smooth transitions for theme changes on key UI elements

### 3. `frontend/script.js`
- Added `themeToggle` to DOM element references
- Added `initTheme()` function:
  - Loads saved theme from localStorage
  - Falls back to system preference (prefers-color-scheme)
  - Listens for system theme changes
- Added `toggleTheme()` function to switch between themes
- Added `setTheme()` function to apply theme and update aria-label
- Integrated theme toggle into event listeners

## Features

### Design
- Circular button positioned in top-right corner
- Uses sun icon for dark mode, moon icon for light mode
- Matches existing design aesthetic with consistent colors and styling

### Animations
- Smooth icon swap animation with rotation and scale
- 0.3s transition duration for all theme-related changes
- Hover: slight scale up and border color change
- Active: scale down feedback

### Accessibility
- Keyboard navigable (focusable with Tab)
- Visible focus ring using existing `--focus-ring` color
- Dynamic `aria-label` that describes the action (e.g., "Switch to dark mode")
- `title` attribute for tooltip on hover

### Persistence
- Theme preference saved to localStorage
- Respects system preference on first visit
- Listens for system preference changes when no manual preference is set

## Light Theme CSS Variables

The light theme is activated via `[data-theme="light"]` selector on the document root.

### Color Palette Comparison

| Variable | Dark Value | Light Value | Purpose |
|----------|-----------|-------------|---------|
| `--background` | `#0f172a` | `#f8fafc` | Page background |
| `--surface` | `#1e293b` | `#ffffff` | Cards, sidebar, inputs |
| `--surface-hover` | `#334155` | `#f1f5f9` | Hover states for surfaces |
| `--text-primary` | `#f1f5f9` | `#1e293b` | Main text content |
| `--text-secondary` | `#94a3b8` | `#64748b` | Labels, hints, metadata |
| `--border-color` | `#334155` | `#e2e8f0` | Borders and dividers |
| `--assistant-message` | `#374151` | `#f1f5f9` | AI message background |
| `--shadow` | `rgba(0,0,0,0.3)` | `rgba(0,0,0,0.1)` | Box shadows |
| `--focus-ring` | `rgba(37,99,235,0.2)` | `rgba(37,99,235,0.3)` | Focus indicators |
| `--welcome-bg` | `#1e3a5f` | `#eff6ff` | Welcome message background |
| `--code-bg` | `rgba(0,0,0,0.2)` | `rgba(0,0,0,0.05)` | Code block background |

### Accessibility Standards

- **Text Contrast**: Light theme uses `#1e293b` text on `#f8fafc` background (contrast ratio ~14:1, exceeds WCAG AAA)
- **Secondary Text**: `#64748b` on light backgrounds maintains 4.5:1+ contrast (WCAG AA compliant)
- **Primary Color**: Blue `#2563eb` is preserved for consistent branding and sufficient contrast
- **Focus Indicators**: Enhanced focus ring opacity (0.3 vs 0.2) for better visibility on light backgrounds

### Design Decisions

1. **Consistent Primary Colors**: `--primary-color` and `--primary-hover` remain unchanged to maintain brand identity
2. **User Message Color**: Blue user messages stay the same for visual continuity
3. **Softer Shadows**: Reduced shadow opacity (0.1 vs 0.3) for a cleaner light appearance
4. **Slate Color Palette**: Uses Tailwind CSS slate scale for professional, neutral tones
