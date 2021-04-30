# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Generate figures --------------------------------------------------------

from contextlib import contextmanager
from itertools import combinations
import math
import os
import subprocess
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image

THETAS = [0, 2 * math.pi / 3, 4 * math.pi / 3]
# This in inches
WIDTH = 4.8

cmap = matplotlib.cm.get_cmap('viridis', 3)
colors = cmap(np.linspace(0, 1, 3))
black = tuple((colors[0] * 0.25))

STATE_COLOR = {
    0: 'white',
    **{idx: tuple(colors[idx]) for idx in [1, 2]}
}


@contextmanager
def in_temporary_directory():
    '''
    This convenience function is analogous to tempfile.TemporaryDirectory,
    except that it also temporarily changes the working directory to that
    temporary directory's path, then changes it back on exiting the context.
    '''
    # Note the need to use an absolute path here, since ordinarily it will just
    # be '.' to start with. Trying to change back to that when the temp_path no
    # longer exists will cause all kinds of problems, so let's just normalize
    # it. Unfortunately, there's no easy way I can see to ensure a true return
    # to EXACTLY the original curdir IN ITS ORIGINAL FORM if it's a relative
    # path, but this will ensure we're returned to the same original ABSOLUTE
    # path, and I struggle to think of circumstances under which that would make
    # a difference.
    orig_path = os.path.abspath(os.curdir)
    with tempfile.TemporaryDirectory() as temp_path:
        os.chdir(temp_path)
        yield temp_path
        os.chdir(orig_path)


def make_image(ax, states, shift_x=0):
    for theta_1, theta_2 in combinations(THETAS, 2):
        x_1, y_1 = int(128 * math.cos(theta_1)), int(128 * math.sin(theta_1))
        x_2, y_2 = int(128 * math.cos(theta_2)), int(128 * math.sin(theta_2))
        ax.plot([x_1 + shift_x, x_2 + shift_x], [y_1, y_2], color=black)

    for theta, state, name in zip(THETAS, states, 'abc'):
        x, y = int(128 * math.cos(theta)), int(128 * math.sin(theta))
        ax.plot(
            x + shift_x, y, marker='o', fillstyle='full', markersize=48,
            color=STATE_COLOR[state], markeredgecolor=black
        )
        ax.text(
            x + shift_x, y, str(state),
            fontsize=18,
            verticalalignment='center',
            horizontalalignment='center'
        )
        ax.text(
            int(1.3 * x) + shift_x, int(1.3 * y), name, fontsize=12,
            verticalalignment='center', horizontalalignment='center'
        )


def draw_curved_arrow(ax, from_vertex, to_vertex, style='->', shift_x=0):
    theta_1, theta_2 = THETAS[from_vertex], THETAS[to_vertex]
    sign = 1 if (theta_1 < theta_2) else -1

    theta_1 += sign * (math.pi / 12)
    theta_2 -= sign * (math.pi / 12)

    x_1, y_1 = int(128 * math.cos(theta_1)), int(128 * math.sin(theta_1))
    x_2, y_2 = int(128 * math.cos(theta_2)), int(128 * math.sin(theta_2))

    arc = matplotlib.patches.FancyArrowPatch(
        (x_1 + shift_x, y_1), (x_2 + shift_x, y_2),
        connectionstyle='arc3,rad=.5',
        arrowstyle=style,
        mutation_scale=40
    )
    ax.add_patch(arc)


def draw_progression_arrow(ax, shift_x=320, style='->'):
    arc = matplotlib.patches.FancyArrowPatch(
        (-160 + shift_x + 20, 0), (-160 + shift_x + 80, 0),
        arrowstyle=style,
        mutation_scale=40
    )
    ax.add_patch(arc)


fig, ax = plot.subplots(1, 1, figsize=(WIDTH, WIDTH))
plot.axis('off')
ax.set_xlim(-160, 160)
ax.set_ylim(-160, 160)
make_image(ax, [1, 0, 0])
plot.savefig('images/graph_1.png', transparent=True)
plot.close()

fig, ax = plot.subplots(1, 1, figsize=(WIDTH, WIDTH))
plot.axis('off')
ax.set_xlim(-160, 160)
ax.set_ylim(-160, 160)
make_image(ax, [1, 1, 0])
draw_curved_arrow(ax, 0, 1)
plot.savefig('images/graph_2.png', transparent=True)
plot.close()

fig, ax = plot.subplots(1, 1, figsize=(WIDTH, WIDTH))
plot.axis('off')
ax.set_xlim(-160, 160)
ax.set_ylim(-160, 160)
make_image(ax, [0, 1, 0])
draw_curved_arrow(ax, 0, 1, style='<->')
plot.savefig('images/graph_3.png', transparent=True)
plot.close()

fig, ax = plot.subplots(1, 1, figsize=(2 * WIDTH, WIDTH))
plot.axis('off')
ax.set_xlim(-160, 480)
ax.set_ylim(-160, 160)
make_image(ax, [1, 0, 0])
make_image(ax, [1, 1, 0], shift_x=320)
draw_curved_arrow(ax, 0, 1, shift_x=320)
draw_progression_arrow(ax, shift_x=320)
plot.savefig('images/graph_4.png', transparent=True)
plot.close()

fig, ax = plot.subplots(1, 1, figsize=(2 * WIDTH, WIDTH))
plot.axis('off')
ax.set_xlim(-160, 480)
ax.set_ylim(-160, 160)
make_image(ax, [1, 0, 0])
make_image(ax, [0, 1, 0], shift_x=320)
draw_curved_arrow(ax, 0, 1, style='<->', shift_x=320)
draw_progression_arrow(ax, shift_x=320)
plot.savefig('images/graph_5.png', transparent=True)
plot.close()

examples_path = os.path.join(os.path.dirname(__file__), '../../examples/')
script_path = os.path.join(examples_path, 'run_may_and_leonard.py')
force_regenerate = os.getenv('FORCE_REGENERATE', False)

if not os.path.exists(script_path):
    warnings.warn(
        'Could not locate example generating script. Omitting regeneration '
        'of example images.'
    )

else:
    for n, r in ((3, 1), (4, 1), (6, 3)):
        # The conf.py runs from the source directory, and this will be the
        # correct relative path. However, we're going to change directory to a
        # temporary directory below to avoid cluttering up the file system with
        # temporary files in the event of an error, so normalize the path now.
        image_path = os.path.abspath(f'images/may_leonard_{n}_{r}.png')
        if os.path.exists(image_path) and not force_regenerate:
            continue
        command = [
            'python3', script_path,
            '--elapsed-time', '1000',
            '--type', str(n), str(r),
            '--output-path', '.',
        ]

        # Capture stdout to avoid printing redundant information to screen.
        # stderr will still be printed.
        with in_temporary_directory() as temp_path:
            # This will produce a gif as output. Open it and extract the last
            # frame, to save to the target directory.
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            image = Image.open(os.path.join(temp_path, '0.gif'))
            image.seek(image.n_frames - 1)
            image.save(image_path)
        print(f'Regenerated image {os.path.basename(image_path)}')


# -- Path setup --------------------------------------------------------------


# -- Project information -----------------------------------------------------

project = 'Viridicle'
copyright = '2021'
author = 'Mark Lowell'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

try:
    import viridian

    html_theme_options = {}
    html_theme_options.update(viridian.SPHINX_THEME_OPTIONS)

    pygments_style = 'viridian'

except ImportError:
    print('viridian not found. Building with alabaster defaults.')


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Remove source from build output
html_copy_source = False
