######
 Data
######

This module is used to initialise datasets (constructed using
anemoi-datasets) and load data into the model. It performs
validation checks, such as ensuring that the training dataset end date is
before the start date of the validation dataset.

Each training, validation, and test dataset is a map-style dataset:
an index selects one sample, which ``__getitem__`` reads from all data
readers. The datamodule pairs each dataset with a sampler that decides
which samples a rank loads and in which order, and with a dataloader
that records its position so that training can resume in the middle of
an epoch.

Dataset Architecture
====================

The data module provides three types of dataset readers that wrap
anemoi-datasets data:

Native Grid Dataset
-------------------

The ``NativeGridDataset`` class is used for standard atmospheric data
on a native grid. It provides a simple interface for reading data samples
at specified time indices.

Trajectory Dataset
------------------

The ``TrajectoryDataset`` class extends ``NativeGridDataset`` to support
trajectory-based sampling, where data is organized into temporal
trajectories. This is useful for tracking atmospheric features over time
or for specialized training strategies that require trajectory awareness.

Trajectories are defined by:

* **Trajectory start**: The reference datetime from which trajectories begin
* **Trajectory length**: The number of time steps in each trajectory

Each sample in the dataset is associated with a trajectory ID, ensuring
that samples are correctly grouped and that trajectory boundaries are
respected during training.

Composite Native Grid Dataset
-----------------------------

The ``CompositeNativeGridDataset`` class reads a base dataset like
``NativeGridDataset`` and replaces parts of each sample with data from
the datasets listed under ``overrides``. The base dataset provides the
grid, the statistics and the targets. Each override replaces one
``region`` of the grid at some of the time steps of a sample
(``times``), counted from its t_0:

* ``region``: ``interior`` (the first grid of a ``cutout``),
  ``boundary`` (the other grid points) or ``all``
* ``times``: ``inputs`` (up to t_0), ``outputs`` (after t_0),
  ``from_t0`` (from t_0 on) or ``all``

A gridded source is read at the valid time of each time step. A
trajectories source is read from the forecast initialised at t_0, at
the lead time of each time step, so it can only replace ``outputs`` or
``from_t0``. Grid points are matched to the nearest source grid point,
at most ``max_distance_km`` away, and variables by name (all variables
that both datasets have, unless ``variables`` lists them). Overrides are
applied in order, and only the samples for which every override has its
data are used.

The interior after t_0 holds the loss targets, so it always comes from
the base dataset. On the boundary, the values after t_0 are the lateral
boundary conditions of the rollout and are not part of the loss.

This makes it possible to fine-tune a limited area model trained on
``cutout(CERRA, ERA5)`` with the data it gets operationally:

.. code:: yaml

   dataloader:
     training:
       datasets:
         data:
           dataset_config:
             dataset: ${dataloader.dataset}  # cutout(CERRA, ERA5)
             frequency: ${data.frequency}
           start: null
           end: 2020
           overrides:
             # IFS analysis interpolated to the CERRA grid on the interior
             - region: interior
               times: inputs
               dataset_config:
                 dataset: ifs-an-cerra-grid.zarr
             # IFS analysis on the boundary
             - region: boundary
               times: inputs
               dataset_config:
                 dataset: ifs-an-n320.zarr
             # The IFS forecast initialised at t_0 on the boundary.
             # With times: from_t0, the boundary at t_0 is also taken from it.
             - region: boundary
               times: outputs
               dataset_config:
                 dataset: ifs-fc-n320-trajectories.zarr

Multi-Dataset
-------------

The ``MultiDataset`` class provides a higher-level wrapper that can
synchronize and combine multiple datasets (either ``NativeGridDataset``
or ``TrajectoryDataset`` instances). This is the primary interface used
for training and supports:

* Synchronizing samples across multiple datasets with different grids
* Handling grid sharding for distributed training

Shuffling and the split of samples between groups of ranks are done by the
sampler that the datamodule builds, not by the dataset.

.. note::

   Users wishing to change the format of the batch input into the model
   should sub-class ``MultiDataset`` and override the ``get_sample``
   method.

Data Loading Policy
--------------------

The ``DataLoadingPolicy`` pairs the dataset construction above with the
batch sampler that orders, shards and batches its samples, so the two always
stay consistent with each other. It is selected via the
``dataloader.loading_policy`` config entry. The default,
``SynchronizedLoadingPolicy``, builds one synchronized ``MultiDataset`` and
splits the samples of a distributed sampler into batches with torchdata's
``BatchSampler``, which remembers its position so that training can resume
in the middle of an epoch.

API Reference
=============

Dataset Readers
---------------

.. automodule:: anemoi.training.data.data_reader
   :members:
   :no-undoc-members:
   :show-inheritance:

Multi-Dataset API
-----------------

.. automodule:: anemoi.training.data.multidataset
   :members:
   :no-undoc-members:
   :show-inheritance:

Data Loading Policy API
------------------------

.. automodule:: anemoi.training.data.loading
   :members:
   :no-undoc-members:
   :show-inheritance:
