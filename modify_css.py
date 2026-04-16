import re

with open('frontend/web/styles.css', 'r') as f:
    content = f.read()

# 1. Update Root Variables for Luminescent Theme
new_root = """@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --surface: #0b1326;
  --surface-container-low: #131b2e;
  --surface-container: #171f33;
  --surface-container-high: #222a3d;
  --surface-container-highest: #2d3449;
  --on-surface: #dae2fd;
  --on-surface-variant: #c7c4d7;
  --primary: #c0c1ff;
  --primary-container: #8083ff;
  --secondary: #ddb7ff;
  --secondary-container: #6f00be;
  --tertiary: #ffb0cd;
  --tertiary-container: #f751a1;
  --error: #ffb4ab;
  --error-container: #93000a;
  --outline-variant: #464554;
  --shadow-ambient: 0px 24px 48px -12px rgba(0, 0, 0, 0.5);
  --shadow-glow: 0px 0px 24px rgba(192, 193, 255, 0.4);
  --radius-xl: 0px; 
  --radius-lg: 0px;
  --radius-md: 0px;
  --text: var(--on-surface);
  --muted: var(--on-surface-variant);
  --line: rgba(70, 69, 84, 0.15); /* outline-variant at 15% */
  --cyan: var(--primary);
  --blue: var(--secondary);
  --amber: var(--tertiary);
  --coral: var(--error);
  --green: #4ee1a0;
}
"""
content = re.sub(r"@import.*?:root \{.*?\n\}", new_root, content, flags=re.DOTALL)

# 2. Update Typography
content = content.replace("font-family: 'Sora', sans-serif;", "font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;")
content = content.replace("font-family: 'Space Grotesk', sans-serif;", "font-family: 'Inter', sans-serif;")

# 3. Update body background
new_body = """body {
  margin: 0;
  min-height: 100vh;
  font-family: 'Inter', sans-serif;
  color: var(--text);
  background: var(--surface);
}

body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(192, 193, 255, 0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(192, 193, 255, 0.05) 1px, transparent 1px);
  background-size: 50px 50px;
  mask-image: radial-gradient(circle at center, rgba(0, 0, 0, 0.8), transparent 100%);
  pointer-events: none;
}"""
content = re.sub(r"body \{.*?body::before \{.*?\}", new_body, content, flags=re.DOTALL)

# 4. Update the glass layers
new_glass = """.glass {
  position: relative;
  overflow: hidden;
  border-bottom: 1px solid var(--line);
  background: var(--surface-container-high);
  backdrop-filter: blur(24px);
  box-shadow: var(--shadow-ambient);
  transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass:hover {
  box-shadow: var(--shadow-glow);
  transform: translateY(-2px);
}"""
content = re.sub(r"\.glass \{.*?\}", new_glass, content, flags=re.DOTALL)
content = re.sub(r"\.glass::before \{.*?\}", "", content, flags=re.DOTALL)

# 5. Make buttons neon
new_buttons = """.button {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  padding: 12px 24px;
  border: 0;
  border-radius: 0;
  cursor: pointer;
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), filter 0.3s;
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.button:hover {
  transform: translateY(-2px);
  filter: saturate(1.2);
}

.button-primary {
  color: #0b1326;
  background: linear-gradient(45deg, var(--primary), var(--secondary), var(--tertiary));
  box-shadow: 0 10px 20px rgba(192, 193, 255, 0.2);
}

.button-primary:hover {
  box-shadow: 0 0 20px var(--primary-container);
}

.button-ghost {
  color: var(--text);
  background: rgba(45, 52, 73, 0.6); /* surface-container-highest at 60% */
  backdrop-filter: blur(12px);
  border-bottom: 2px solid var(--primary-container);
}

.button-ghost:hover {
  background: rgba(45, 52, 73, 0.9);
}"""
content = re.sub(r"\.button \{.*\.button-ghost:hover \{.*?\}", new_buttons, content, flags=re.DOTALL)

# 6. Inputs & Selects
new_inputs = """select,
input[type="number"],
input[type="text"] {
  width: 100%;
  padding: 13px 14px;
  border: none;
  border-bottom: 1px solid var(--line);
  border-radius: 0;
  color: var(--text);
  background: transparent;
  outline: none;
  transition: background 0.3s, border-bottom-color 0.3s;
}

select:focus,
input[type="number"]:focus,
input[type="text"]:focus {
  background: var(--surface-container-lowest);
  border-bottom: 2px solid var(--primary);
}"""
content = re.sub(r"select,.*?input\[type=\"text\"\]:focus \{.*?\}", new_inputs, content, flags=re.DOTALL)

# 7. Add slide-up and fade animations for cards
content = content.replace("animation: fade-up 0.55s ease both;", "animation: fade-up 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;")

# 8. Hide borders in cards
content = re.sub(r"border: 1px solid rgba.*?\];", "border-bottom: 1px solid var(--line);", content)

with open('frontend/web/styles.css', 'w') as f:
    f.write(content)

