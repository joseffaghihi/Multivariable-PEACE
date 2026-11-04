# Provenance and scope

The original public simulator notebooks are retained at their repository-root paths. Their baseline commit is 70fb6c2c3f51d0a76fd5ec909e53ade4b18b6307. The revised implementations are separate scripts under experiments/.

The customer simulator uses a monthly-usage coefficient of 3 and independent bugs faced, following its structural assignments. A historical coefficient-30 equation is retained only as an explicitly named oracle sensitivity. The current primary analysis uses coefficient 3. Graph-implied nulls are distinguished from observationally estimated nulls.

The revised studies were developed sequentially and are not claimed to be preregistered. Exploratory nonlinear-component selection was introduced after derivative instability was observed; the earlier linear, additive and interaction-capable fits remain reported. Integration and spline-scaling checks led to corrections before the final reference snapshot. Superseded development versions are not alternative final results.

The matched-mechanism design was recorded before its noisy calibration. design-record.json retains the original design and source hashes and adds the published script hash. Publication cleanup reformats code, makes imports explicit, replaces local-project paths, and adds execution and verification tools. It does not change the numerical designs. docs/source-history.json records the source hashes before cleanup.

Only synthetic experimental code, numerical outputs and technical documentation are included. Editorial correspondence, private manuscript sources and local environment paths are not replication dependencies.

The original notebooks have been left unchanged and may contain state-dependent exploratory cells. The revised command-line pipelines are the supported replication path.
