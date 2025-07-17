# CCM

Elixir module that implements Convergent Cross Mapping (CCM) for time series data.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed by adding `ccm` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ccm, "~> 0.1.0"}
  ]
end
```

## Key Features

* Time-delay embedding - Reconstructs the attractor from univariate time series
* Cross-mapping - Predicts one variable from another's reconstructed state space
* Convergence testing - Checks if prediction skill improves with library size
* Bidirectional analysis - Tests causality in both directions
* Bootstrap sampling - Improves statistical reliability

## Main Functions

* CCM.new/3 - Creates a new CCM analysis structure
* cross_map/2 - Performs unidirectional CCM analysis
* bidirectional_ccm/1 - Tests causality in both directions

## Usage

```elixir
# Generate test data
{x_series, y_series} = CCMExample.generate_coupled_logistic_maps(300, 0.15)

# Perform analysis
ccm = CCM.new(x_series, y_series, embedding_dim: 3, tau: 1, num_samples: 50)
results = CCM.bidirectional_ccm(ccm)

# Run the example
CCMExample.run()
```