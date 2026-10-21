"""The tag of mpiparticles' own scope-profiler regions.

The regions are opt-in: they record nothing unless the application enables the
tag, for example with ``scope_profiler.setup(enable_tags=["mpiparticles"])``.
"""

#: Tags of every mpiparticles region; registered as opt-in in ``__init__``.
PROFILE_TAGS = ("mpiparticles",)
