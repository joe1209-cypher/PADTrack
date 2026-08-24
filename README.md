# PADTrack-Prior-guided-Anisotropic-Thermal-Diffusion-for-Thermal-Infrared-Tracking
## Abstract
Existing Thermal Infrared (TIR) trackers generally
perform feature refinement over the entire search region in a
uniform manner and rely on purely data-driven feature learning.
However, such strategies overlook two intrinsic characteristics:
(1) target-related thermal responses are typically spatially sparse,
and (2) TIR observations exhibit spatially structured thermal
patterns, for which structure-preserving diffusion provides a
natural inductive bias. As a result, irrelevant background ac-
tivations may be excessively amplified, while weak yet target-
consistent thermal structures cannot be effectively preserved. To
address these issues, we propose PADTrack, a selective thermal
feature evolution framework that introduces anisotropic thermal
diffusion into TIR tracking. Different from the conventional
feature enhancement that mainly recalibrates feature responses,
our feature evolution refers to an explicit spatial state tran-
sition of features through controlled information propagation.
Specifically, we first design a foreground-aware spatial prior
module to generate a dynamic sparse spatial prior map, which
identifies target-consistent regions and distinguishes them from
background-dominated areas. Based on the generated spatial
semantic prior, we develop a prior-guided anisotropic thermal
diffusion module that adaptively evolves thermal features by
selectively propagating informative responses. This module en-
hances discriminative target structures while suppressing unreli-
able background interference. Furthermore, we introduce a gated
diffusion residual injection mechanism to dynamically control
diffusion updates and integrate reliable evolved features into the
original representation, preventing the accumulation of ambigu-
ous responses during feature evolution. Extensive experiments
on four TIR tracking benchmarks demonstrate that PADTrack
achieves favorable results.
<img width="1000" height="600" alt="NT" src="https://github.com/user-attachments/assets/373db1b6-f89b-4885-bcc6-c17d6049670d" />
## Download
## Install the environment
## Set project paths
