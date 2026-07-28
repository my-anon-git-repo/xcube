import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

def setup_publication_style():
    """
    Set up matplotlib for publication-quality plots.
    Call this function at the beginning of your plotting script.
    """
    
    # Use seaborn paper style as base
    plt.style.use('seaborn-v0_8-paper')
    
    # Configure fonts for publication quality
    try:
        available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
        
        # Preferred serif fonts in order of preference
        serif_fonts = ['CMU Serif', 'CMU Classical Serif', 'CMU Serif Extra',
                       'STIXGeneral', 'Times New Roman',
                       'Nimbus Roman', 'DejaVu Serif', 'serif']
        
        font_to_use = 'DejaVu Serif'  # Default fallback
        for font in serif_fonts:
            if any(font.lower() in avail_font.lower() for avail_font in available_fonts):
                font_to_use = font
                break
        
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = [font_to_use] + serif_fonts
        print(f"Using font: {font_to_use}")
        
        # Use Computer Modern for math
        mpl.rcParams['mathtext.fontset'] = 'cm'
        
    except:
        # Fallback to sans-serif if serif fonts not available
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
        print("Using sans-serif fonts as fallback")
    
    # Font sizes
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10.5
    mpl.rcParams['figure.titlesize'] = 14
    
    # Line and marker properties
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 8
    
    # Grid properties
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linewidth'] = 0.5


def create_figure(figsize=(10, 6.18), dpi=300):
    """
    Create a figure with golden ratio proportions.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size in inches (width, height). Default uses golden ratio.
    dpi : int
        Dots per inch for high-resolution output.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, ax


def style_axes(ax, fig):
    """
    Apply publication-quality styling to axes.
    
    Parameters:
    -----------
    ax : matplotlib axes object
    fig : matplotlib figure object
    """
    # Set background colors
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Configure spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)
    
    # Hide top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=6, width=1.0)


def get_color_palette():
    """
    Get a professional color palette for plots.
    
    Returns:
    --------
    dict : Color codes for different models/categories
    """
    return {
        'primary': '#E74C3C',      # Red
        'secondary': '#3498DB',    # Blue
        'tertiary': '#27AE60',     # Green
        'quaternary': '#8B4513',   # Brown
        'accent1': '#9B59B6',      # Purple
        'accent2': '#F39C12',      # Orange
        'accent3': '#1ABC9C',      # Turquoise
        'neutral': '#7F8C8D'       # Gray
    }


def save_publication_figure(fig, output_path, filename, formats=['png', 'pdf']):
    """
    Save figure in multiple formats for publication.
    
    Parameters:
    -----------
    fig : matplotlib figure object
    output_path : str or Path object
        Directory to save figures
    filename : str
        Base filename without extension
    formats : list
        File formats to save (default: ['png', 'pdf'])
    """
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    for fmt in formats:
        save_path = output_path / f'{filename}.{fmt}'
        fig.savefig(save_path,
                    format=fmt,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"✓ Saved: {save_path}")


def format_legend(ax, loc='best', **kwargs):
    """
    Create a nicely formatted legend.
    
    Parameters:
    -----------
    ax : matplotlib axes object
    loc : str or tuple
        Legend location
    **kwargs : additional legend parameters
    """
    default_params = {
        'frameon': True,
        'fancybox': True,
        'shadow': True,
        'framealpha': 0.95,
        'borderpad': 1.0
    }
    default_params.update(kwargs)
    
    return ax.legend(loc=loc, **default_params)


# Example usage function
def example_plot():
    """Example of how to use the publication style."""
    
    # Set up the style
    setup_publication_style()
    
    # Create figure
    fig, ax = create_figure()
    
    # Get colors
    colors = get_color_palette()
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
    
    # Plot data
    ax.plot(x, y1, label='Model A', color=colors['primary'], linewidth=2.5)
    ax.plot(x, y2, label='Model B', color=colors['secondary'], linewidth=2.5, linestyle='--')
    
    # Labels and title
    ax.set_xlabel('X-axis Label', fontsize=13, fontweight='medium')
    ax.set_ylabel('Y-axis Label', fontsize=13, fontweight='medium')
    ax.set_title('Publication-Quality Plot Example', fontsize=15, fontweight='bold', pad=20)
    
    # Apply styling
    style_axes(ax, fig)
    
    # Add legend
    format_legend(ax, loc='upper right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    save_publication_figure(fig, './publication_figures', 'example_plot')
    
    return fig, ax


# Specific plot styling functions

def style_violin_plot(parts, colors, alpha=0.7):
    """
    Style violin plot parts with publication quality.
    
    Parameters:
    -----------
    parts : dict
        Dictionary of violin plot parts from ax.violinplot()
    colors : list
        List of colors for each violin
    alpha : float
        Transparency level
    """
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(alpha)
        pc.set_edgecolor('#333333')
        pc.set_linewidth(1.5)
    
    for partname in ['cbars', 'cmins', 'cmaxes']:
        if partname in parts:
            parts[partname].set_edgecolor('#333333')
            parts[partname].set_linewidth(1.5)


def style_box_plot(bp, colors, alpha=0.7):
    """
    Style box plot with publication quality.
    
    Parameters:
    -----------
    bp : dict
        Dictionary from ax.boxplot()
    colors : list
        List of colors for each box
    alpha : float
        Transparency level
    """
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='#333333', linewidth=1.5)


def add_statistical_annotations(ax, x_pos, y_pos, text, fontsize=10):
    """
    Add statistical annotations to plot.
    
    Parameters:
    -----------
    ax : matplotlib axes object
    x_pos : float
        X position for text
    y_pos : float
        Y position for text
    text : str
        Annotation text
    fontsize : int
        Font size for annotation
    """
    ax.text(x_pos, y_pos, text,
            ha='center', va='bottom',
            fontweight='bold',
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.3",
                     facecolor='white',
                     edgecolor='gray',
                     alpha=0.8))


# LaTeX integration helpers

def format_for_latex(value, decimals=3):
    """Format numeric value for LaTeX."""
    return f"{value:.{decimals}f}"


def generate_latex_caption(plot_type, statistics):
    """
    Generate a LaTeX caption for the plot.
    
    Parameters:
    -----------
    plot_type : str
        Type of plot (e.g., 'violin', 'box', 'cdf')
    statistics : dict
        Dictionary containing relevant statistics
    
    Returns:
    --------
    str : Formatted LaTeX caption
    """
    base_caption = f"\\caption{{{plot_type} plot showing... "
    
    # Add statistics to caption
    if 'mean' in statistics:
        base_caption += f"Mean values: {statistics['mean']}. "
    
    base_caption += "}}"
    
    return base_caption


if __name__ == "__main__":
    # Run example
    fig, ax = example_plot()
    plt.show()