# env\_yaml

This directory contains `yaml` files that hold the `conda` environment
settings for this repo. These files are referenced by
`../scripts/setup_main.sb`.

## Important note
 This pipeline was originally developed to use the
`keras2-tf27.yml` environment specification. This spec lists
only a few dependencies, without version numbers. This choice
was intentional, and it allowed new users of the pipeline
to use the most recent versions of all required packages.

Around Jul 1, 2024, we discovered that installing from this
file created an environment that resulted in errors during
training. This means that at least one of the listed packages
has a new version that breaks the pipeline. To fix this,
we exported an existing environment, creating
`keras2-tf27-frozen.yml`, a conda environment
specification that lists the exact version of every installed
package. The setup scripts were also updated to point to this
frozen environment spec.

In the interest of time, we will leave the frozen env spec
because we know that the current pipeline works with this
exact environment. In the future, it would be ideal to
figure out which package(s) broke the non-frozen environment,
and update the non-frozen env with only the required
package versions, i.e. allowing all packages to use their
latest version except for the one(s) that need to have
frozen versions.

\- Heather
