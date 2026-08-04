import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shutil
import os

# Clear matplotlib cache first
cache_dir = os.path.expanduser('~/.cache/matplotlib')
if os.path.exists(cache_dir):
    print("Clearing matplotlib cache...")
    shutil.rmtree(cache_dir)
    print("Cache cleared successfully!")
else:
    print("No matplotlib cache found to clear")

print()

# Rebuild font cache using the correct method
print("Rebuilding font cache...")
try:
    # Try the newer method first
    fm.fontManager.ttflist = fm.createFontList(fm.findSystemFonts())
    print("Font cache rebuilt using createFontList!")
except:
    try:
        # Try alternative method
        fm._load_fontmanager(try_read_cache=False)
        print("Font cache rebuilt using _load_fontmanager!")
    except:
        # Last resort - just get a fresh FontManager
        fm.fontManager = fm.FontManager()
        print("Font cache rebuilt using new FontManager!")

print()

# Get all available fonts
all_fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))

print("=== SEARCHING FOR COMPUTER MODERN VARIANTS ===")
print()

# Search for Computer Modern variants
cm_variants = []
search_terms = ['cm', 'computer', 'modern', 'cmu', 'lm', 'latin']

for font in all_fonts:
    font_lower = font.lower()
    if any(term in font_lower for term in search_terms):
        cm_variants.append(font)
        print(f"Found: {font}")

print()
print("=== SEARCHING FOR OTHER ACADEMIC FONTS ===")
print()

# Search for other good academic fonts
academic_fonts = {
    'times': [],
    'termes': [],
    'stix': [],
    'liberation': [],
    'nimbus': [],
    'dejavu': []
}

for font in all_fonts:
    font_lower = font.lower()
    for key in academic_fonts:
        if key in font_lower:
            academic_fonts[key].append(font)

for font_type, fonts in academic_fonts.items():
    if fonts:
        print(f"\n{font_type.upper()} variants:")
        for f in fonts:
            print(f"  - {f}")

print()
print("=== FONT FILE LOCATIONS ===")
print()

# Check actual font files
import subprocess
import os

try:
    # Check system font directories
    font_dirs = [
        '/usr/share/fonts/',
        '/usr/local/share/fonts/',
        '~/.fonts/',
        '~/.local/share/fonts/'
    ]
    
    for dir_path in font_dirs:
        expanded_path = os.path.expanduser(dir_path)
        if os.path.exists(expanded_path):
            result = subprocess.run(['find', expanded_path, '-name', '*cm*.ttf', '-o', '-name', '*cm*.otf'], 
                                  capture_output=True, text=True)
            if result.stdout:
                print(f"Computer Modern fonts found in {dir_path}:")
                print(result.stdout)
except:
    pass

print()
print("=== RECOMMENDED FONT TO USE ===")
print()

# Determine best available font
if cm_variants:
    recommended = cm_variants[0]
    print(f"Use this Computer Modern variant: '{recommended}'")
elif academic_fonts['termes']:
    recommended = academic_fonts['termes'][0]
    print(f"Use TeX Gyre Termes (Times clone): '{recommended}'")
elif academic_fonts['liberation']:
    recommended = academic_fonts['liberation'][0]
    print(f"Use Liberation Serif: '{recommended}'")
elif academic_fonts['stix']:
    recommended = academic_fonts['stix'][0]
    print(f"Use STIX: '{recommended}'")
else:
    recommended = 'DejaVu Serif' if 'DejaVu Serif' in all_fonts else 'serif'
    print(f"Use default: '{recommended}'")

print()
print("=== TOTAL FONTS AVAILABLE ===")
print(f"Total number of fonts: {len(all_fonts)}")

# Create a test plot
print()
print("=== CREATING TEST PLOT ===")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [recommended]

fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, f'Test with {recommended}', 
        fontsize=24, ha='center', va='center')
ax.text(0.5, 0.3, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 
        fontsize=16, ha='center', va='center')
ax.text(0.5, 0.2, 'abcdefghijklmnopqrstuvwxyz', 
        fontsize=16, ha='center', va='center')
ax.text(0.5, 0.1, '0123456789', 
        fontsize=16, ha='center', va='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.title(f'Font Test: {recommended}')
plt.tight_layout()
plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Test plot saved as 'font_test.png' using font: {recommended}")